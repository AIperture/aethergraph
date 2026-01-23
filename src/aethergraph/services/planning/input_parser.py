from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aethergraph.contracts.services.llm import LLMClientProtocol


@dataclass
class ParsedInputs:
    """
    Result of attempting to parse user-provided values for some fields.

    - values: successfully parsed values, keyed by field name.
    - resolved_keys: subset of field names for which we got a non-null value.
    - missing_keys: subset of field names we still lack.
    - errors: human-readable errors that the agent/orchestrator can surface.
    """

    values: dict[str, Any]
    resolved_keys: set[str]
    missing_keys: set[str]
    errors: list[str]


class InputParserError(Exception):
    """Hard failure in the input parser (e.g. LLM/schema problem)."""


@dataclass
class InputParser:
    """
    Generic, LLM-backed input parser.

    Given:
      - a user message
      - a set of expected field names
      - optional skill metadata that describes those fields

    it asks the LLM to extract values and returns a ParsedInputs object.

    This parser is intentionally generic across agents/verticals. The only
    domain-specific hints it uses are the optional `skill.meta["inputs"]`
    entries, if provided.

    Expected skill.meta["inputs"] format (optional):

        inputs:
          - name: dataset_path
            description: Path to the training dataset
            required: true
            example: /data/train.csv
          - name: grid_spec
            description: Grid configuration for evaluation
            required: false

    All fields are permitted to be any JSON type; we do not enforce types here.
    """

    llm: LLMClientProtocol

    async def parse_message_for_fields(
        self,
        *,
        message: str,
        missing_keys: list[str],
        skill: Any | None = None,
    ) -> ParsedInputs:
        """
        Ask the LLM to extract values for the given missing_keys.

        Args:
            message: The user's natural-language reply.
            missing_keys: Field names whose values we want to extract.
            skill: Optional SkillSpec-like object with `.meta` containing
                   an "inputs" list describing each field.

        Returns:
            ParsedInputs with values/resolved_keys/missing_keys/errors.

        Notes:
            - If the LLM cannot confidently determine a field, it should set it
              to null. We then treat that as "missing".
            - If the LLM call fails or returns invalid JSON, we return an object
              with all keys in missing_keys and a populated `errors` list.
        """
        # Build per-field descriptions from skill.meta if available
        field_descriptions = self._build_field_descriptions(missing_keys, skill)

        # Build JSON schema for extraction
        schema = self._build_extraction_schema(missing_keys)

        system_prompt = (
            "You are an input extraction assistant. "
            "Your task is to read a user message and extract values for a fixed "
            "set of fields. You must return ONLY a JSON object that conforms "
            "to the provided JSON schema.\n\n"
            "If a field is not clearly specified in the user's message, or you "
            "are not confident about its value, you MUST set that field to null. "
            "Do not try to guess values that are not present."
        )

        # We include field descriptions as a hint to the LLM
        fields_description_str = "\n".join(
            f"- {name}: {desc or '(no description)'}" for name, desc in field_descriptions.items()
        )

        user_prompt = (
            "User message:\n"
            f"{message}\n\n"
            "You must extract values for the following fields (if present):\n"
            f"{fields_description_str}\n\n"
            "Return ONLY the JSON object, no explanations."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            raw, _usage = await self.llm.chat(
                messages,
                output_format="json",
                json_schema=schema,
                schema_name="ParsedInputs",
                strict_schema=True,
                validate_json=True,
            )
        except Exception as exc:  # noqa: BLE001
            # Hard LLM failure → all fields still missing; surface error to user.
            return ParsedInputs(
                values={},
                resolved_keys=set(),
                missing_keys=set(missing_keys),
                errors=[
                    "I couldn't reliably extract the requested inputs from your reply. "
                    "Please restate the values clearly, for example:\n"
                    + "\n".join(f"- {k} = <value>" for k in missing_keys),
                    f"(Internal parser error: {exc!r})",
                ],
            )

        if not isinstance(raw, dict):
            # Should not happen with strict_schema=True, but guard anyway
            return ParsedInputs(
                values={},
                resolved_keys=set(),
                missing_keys=set(missing_keys),
                errors=[
                    "I couldn't reliably extract the requested inputs from your reply. "
                    "Please restate the values clearly.",
                    f"(Parser expected JSON object, got {type(raw)})",
                ],
            )

        values: dict[str, Any] = {}
        resolved: set[str] = set()
        missing: set[str] = set()

        for key in missing_keys:
            val = raw.get(key, None)
            if val is None:
                missing.add(key)
            else:
                values[key] = val
                resolved.add(key)

        errors: list[str] = []
        if missing:
            errors.append(
                "I still don't have values for the following fields: "
                + ", ".join(sorted(missing))
                + ". Please specify them explicitly, for example:\n"
                + "\n".join(f"- {k} = <value>" for k in sorted(missing))
            )

        return ParsedInputs(
            values=values,
            resolved_keys=resolved,
            missing_keys=missing,
            errors=errors,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_field_descriptions(
        missing_keys: list[str],
        skill: Any | None,
    ) -> dict[str, str | None]:
        """
        Build a mapping { field_name: description_or_None } using optional skill meta.

        We expect (but do not require):

            skill.meta["inputs"] = [
                { "name": "dataset_path", "description": "...", ...},
                ...
            ]
        """
        descs: dict[str, str | None] = {k: None for k in missing_keys}

        meta = getattr(skill, "meta", None) or {}
        inputs_meta = meta.get("inputs") or []
        if not isinstance(inputs_meta, list):
            return descs

        index: dict[str, dict] = {}
        for entry in inputs_meta:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            if not name:
                continue
            index[name] = entry

        for key in missing_keys:
            if key in index:
                desc = index[key].get("description") or index[key].get("help")
                descs[key] = desc

        return descs

    @staticmethod
    def _build_extraction_schema(
        missing_keys: list[str],
    ) -> dict[str, Any]:
        """
        Build a permissive JSON schema for the extraction object.

        Each field is allowed to be any JSON type or null. We rely on the
        LLM + external validation to make sense of the values.

        Schema:

            {
              "type": "object",
              "properties": {
                "<field>": {
                  "anyOf": [
                    {"type": "string"},
                    {"type": "number"},
                    {"type": "integer"},
                    {"type": "boolean"},
                    {"type": "object"},
                    {"type": "array"},
                    {"type": "null"}
                  ]
                },
                ...
              },
              "required": [],
              "additionalProperties": false
            }
        """
        field_schema: dict[str, Any] = {
            "anyOf": [
                {"type": "string"},
                {"type": "number"},
                {"type": "integer"},
                {"type": "boolean"},
                {"type": "object"},
                {"type": "array"},
                {"type": "null"},
            ]
        }

        props: dict[str, Any] = {k: field_schema for k in missing_keys}

        return {
            "type": "object",
            "properties": props,
            # We do NOT require any fields; LLM sets missing ones to null.
            "required": [],
            "additionalProperties": False,
        }
