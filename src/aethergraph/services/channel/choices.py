from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from aethergraph.contracts.services.channel import ChoiceOption


def build_choice_options(
    options: Iterable[str | ChoiceOption | dict[str, Any]] | None,
) -> list[ChoiceOption]:
    built: list[ChoiceOption] = []
    for _idx, option in enumerate(options or [], start=1):
        if isinstance(option, ChoiceOption):
            aliases = tuple(str(alias) for alias in option.aliases if str(alias).strip())
            built.append(
                ChoiceOption(
                    id=str(option.id),
                    label=str(option.label),
                    aliases=aliases,
                )
            )
            continue

        if isinstance(option, dict):
            label = str(option.get("label") or option.get("name") or option.get("id") or "").strip()
            choice_id = str(option.get("id") or label).strip()
            aliases = tuple(
                str(alias).strip() for alias in (option.get("aliases") or []) if str(alias).strip()
            )
            if choice_id and label:
                built.append(ChoiceOption(id=choice_id, label=label, aliases=aliases))
            continue

        label = str(option).strip()
        if label:
            built.append(ChoiceOption(id=label, label=label, aliases=()))

    if not built:
        return []

    deduped: list[ChoiceOption] = []
    seen: set[str] = set()
    for option in built:
        if option.id in seen:
            continue
        deduped.append(option)
        seen.add(option.id)
    return deduped


def prompt_choices_from_prompt(prompt: Any) -> list[ChoiceOption]:
    if isinstance(prompt, dict):
        raw_choices = prompt.get("choices")
        if isinstance(raw_choices, list):
            choices = build_choice_options(raw_choices)
            if choices:
                return choices
        raw_buttons = prompt.get("buttons") or prompt.get("options") or []
        if isinstance(raw_buttons, list):
            return build_choice_options(raw_buttons)
    if isinstance(prompt, (list, tuple)):
        return build_choice_options(prompt)
    return []


def choice_prompt_payload(
    title: str,
    *,
    options: Iterable[str | ChoiceOption | dict[str, Any]],
) -> dict[str, Any]:
    choices = build_choice_options(options)
    return {
        "title": title,
        "choices": [
            {"id": choice.id, "label": choice.label, "aliases": list(choice.aliases)}
            for choice in choices
        ],
        "buttons": [choice.label for choice in choices],
        "options": [choice.label for choice in choices],
    }


def normalize_choice_reply(
    *,
    prompt: Any,
    raw_choice: Any = None,
    raw_text: Any = None,
) -> dict[str, Any]:
    text = str(raw_text or "")
    choice_text = None if raw_choice is None else str(raw_choice).strip()
    choices = prompt_choices_from_prompt(prompt)

    if not choices:
        return {
            "choice": choice_text or None,
            "choice_label": None,
            "text": text,
            "matched": False,
        }

    by_norm: dict[str, ChoiceOption] = {}
    for idx, choice in enumerate(choices, start=1):
        variants = {
            choice.id,
            choice.label,
            *(choice.aliases or ()),
            str(idx),
        }
        for variant in variants:
            token = str(variant).strip()
            if token:
                by_norm.setdefault(token.lower(), choice)

    candidate = choice_text if choice_text else text.strip()
    if candidate:
        match = by_norm.get(candidate.lower())
        if match is not None:
            return {
                "choice": match.id,
                "choice_label": match.label,
                "text": text,
                "matched": True,
            }

    return {
        "choice": choice_text or None,
        "choice_label": None,
        "text": text,
        "matched": False,
    }
