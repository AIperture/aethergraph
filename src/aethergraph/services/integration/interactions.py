"""Exact open-interaction resolution for canonical ingress."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from aethergraph.contracts.integration import ExternalSessionBinding, IngressEnvelope
from aethergraph.services.channel.choices import normalize_choice_reply
from aethergraph.services.channel.resources import InputResource
from aethergraph.services.continuations.continuation import Continuation


class InteractionResolutionError(RuntimeError):
    """Structured failure raised when ingress cannot select one open interaction."""

    def __init__(
        self,
        *,
        code: Literal[
            "integration.interaction_not_found",
            "integration.interaction_ambiguous",
            "integration.interaction_session_mismatch",
            "integration.interaction_kind_mismatch",
        ],
        message: str,
    ) -> None:
        """Create one stable interaction-resolution failure.

        Resolution never guesses by channel or most-recent continuation.

        Examples:
            Reject an unknown button interaction:
            ```python
            InteractionResolutionError(
                code="integration.interaction_not_found",
                message="The interaction is not open.",
            )
            ```

            Reject ambiguous free text:
            ```python
            InteractionResolutionError(
                code="integration.interaction_ambiguous",
                message="More than one text interaction is open.",
            )
            ```

        Args:
            code: Stable machine-readable interaction failure code.
            message: Human-readable failure explanation.

        Returns:
            None.

        Notes:
            Exact callback identities and bound-session free text are separate policies.
        """
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class ResolvedInteraction:
    """One exact continuation selected for canonical resume."""

    interaction_id: str
    continuation: Continuation


class InteractionResolver:
    """Resolve exact callbacks or one eligible bound-session free-text wait."""

    def __init__(self, continuation_store) -> None:
        """Bind resolution to the Host continuation store.

        The resolver reads open continuation records but never uses correlator,
        channel, prefix, or newest-wait fallbacks.

        Examples:
            Create a resolver:
            ```python
            resolver = InteractionResolver(container.cont_store)
            ```

            Resolve during coordinator acceptance:
            ```python
            result = await resolver.resolve(binding=binding, envelope=envelope)
            ```

        Args:
            continuation_store: Store exposing `list_waits()` and `get_by_token()`.

        Returns:
            None.

        Notes:
            Continuations must carry `_interaction_id` in their persisted payload.
        """
        self.store = continuation_store

    async def resolve(
        self,
        *,
        binding: ExternalSessionBinding,
        envelope: IngressEnvelope,
    ) -> ResolvedInteraction | None:
        """Resolve zero or one exact open interaction for an ingress envelope.

        Choice callbacks require their issued interaction ID. Free text and files
        select by the durable AG session and reject multiple eligible waits.

        Examples:
            Resolve a button callback:
            ```python
            resolved = await resolver.resolve(binding=binding, envelope=choice_envelope)
            ```

            Detect a root turn:
            ```python
            resolved = await resolver.resolve(binding=binding, envelope=text_envelope)
            assert resolved is None
            ```

        Args:
            binding: Durable external-to-AG session binding.
            envelope: Closed canonical ingress envelope.

        Returns:
            ResolvedInteraction | None: Exact open continuation, or `None` for a root turn.

        Notes:
            Structured root input does not resume a text/file interaction implicitly.
        """
        waits = await self.store.list_waits()
        open_waits = [wait for wait in waits if self._is_open(wait)]
        if envelope.choice is not None:
            exact = [
                wait
                for wait in open_waits
                if self._interaction_id(wait) == envelope.choice.interaction_id
            ]
            if not exact:
                raise InteractionResolutionError(
                    code="integration.interaction_not_found",
                    message="The supplied interaction identity is not open.",
                )
            if len(exact) > 1:
                raise InteractionResolutionError(
                    code="integration.interaction_ambiguous",
                    message="The supplied interaction identity is not unique.",
                )
            wait = exact[0]
            if wait.get("session_id") != binding.ag_session_id:
                raise InteractionResolutionError(
                    code="integration.interaction_session_mismatch",
                    message="The interaction does not belong to the bound AG session.",
                )
            if wait.get("kind") not in {"approval", "choice"}:
                raise InteractionResolutionError(
                    code="integration.interaction_kind_mismatch",
                    message="The interaction does not accept a choice response.",
                )
            return await self._load(wait)

        eligible_kinds = self._eligible_kinds(envelope)
        if not eligible_kinds:
            return None
        eligible = [
            wait
            for wait in open_waits
            if wait.get("session_id") == binding.ag_session_id
            and wait.get("kind") in eligible_kinds
        ]
        if not eligible:
            return None
        if len(eligible) > 1:
            raise InteractionResolutionError(
                code="integration.interaction_ambiguous",
                message="More than one eligible interaction is open in the bound session.",
            )
        return await self._load(eligible[0])

    async def _load(self, wait: dict[str, Any]) -> ResolvedInteraction:
        token = str(wait.get("token") or "")
        continuation = await self.store.get_by_token(token)
        if continuation is None or continuation.closed:
            raise InteractionResolutionError(
                code="integration.interaction_not_found",
                message="The selected interaction is no longer open.",
            )
        return ResolvedInteraction(
            interaction_id=self._interaction_id(wait),
            continuation=continuation,
        )

    @staticmethod
    def _interaction_id(wait: dict[str, Any]) -> str:
        payload = wait.get("payload")
        return str(payload.get("_interaction_id") or "") if isinstance(payload, dict) else ""

    @staticmethod
    def _is_open(wait: dict[str, Any]) -> bool:
        if wait.get("closed"):
            return False
        deadline = wait.get("deadline")
        if isinstance(deadline, str):
            deadline = datetime.fromisoformat(deadline)
        return not isinstance(deadline, datetime) or datetime.now(UTC) <= deadline.astimezone(UTC)

    @staticmethod
    def _eligible_kinds(envelope: IngressEnvelope) -> set[str]:
        if envelope.structured_input is not None:
            return set()
        if envelope.attachments and envelope.text is not None:
            return {"user_input_or_files"}
        if envelope.attachments:
            return {"user_files", "user_input_or_files"}
        if envelope.text is not None:
            return {"user_input", "user_input_or_files"}
        return set()


def build_interaction_payload(
    *,
    resolved: ResolvedInteraction,
    envelope: IngressEnvelope,
    resources: tuple[InputResource, ...],
) -> dict[str, Any]:
    """Build the exact resume payload for one resolved interaction.

    The payload is provider-neutral and includes materialized attachment records.

    Examples:
        Build a choice response:
        ```python
        payload = build_interaction_payload(
            resolved=resolved,
            envelope=choice_envelope,
            resources=(),
        )
        ```

        Build a file response:
        ```python
        payload = build_interaction_payload(
            resolved=resolved,
            envelope=file_envelope,
            resources=resources,
        )
        ```

    Args:
        resolved: Exact continuation selected by `InteractionResolver`.
        envelope: Closed canonical ingress envelope.
        resources: Materialized inbound resources.

    Returns:
        dict[str, Any]: Provider-neutral payload consumed by `ResumeRouter`.

    Notes:
        Continuation tokens are not copied into the external payload.
    """
    continuation = resolved.continuation
    attachments = [resource.to_dict() for resource in resources]
    files = [resource.to_display_file() for resource in resources]
    base: dict[str, Any] = {
        "text": envelope.text or "",
        "attachments": attachments,
        "interaction_id": resolved.interaction_id,
    }
    if envelope.choice is not None:
        raw_choice = envelope.choice.option_ids[0]
        normalized = normalize_choice_reply(
            prompt=continuation.prompt,
            raw_choice=raw_choice,
            raw_text=envelope.text or "",
        )
        return {**base, **normalized}
    if continuation.kind in {"user_files", "user_input_or_files"}:
        base["files"] = files
    return base
