---
name: docstring-format-enforcer
description: Enforce and apply a consistent Python docstring style for methods and functions using the section order Intro, Examples, Args, Returns, and Notes. Use when creating or updating APIs, facades, mixins, services, contracts, or any code where docstrings must be machine-readable, example-driven, and strictly aligned with real method signatures.
---

# Docstring Format Enforcer

Write or rewrite Python docstrings using this exact section order:
1. Intro
2. Examples
3. Args
4. Returns
5. Notes

If a section has no meaningful content, keep the section and write a short explicit placeholder (for example, `Notes: None.`).

## Required Output Rules

- Use triple double quotes.
- Keep the first line as a one-sentence summary.
- Add a short intro paragraph describing behavior and side effects.
- Provide at least two runnable-style examples for public methods.
- Keep example code in fenced `python` blocks.
- Ensure every argument in `Args` exists in the method signature.
- Do not document removed or nonexistent parameters.
- Ensure `Returns` matches the real return type and behavior.
- Include deprecation behavior in `Notes` when applicable.
- Use consistent terminology with the codebase (Event, facade names, service names).

## Canonical Template

```python
"""
<One-line summary sentence.>

<Intro paragraph explaining behavior, side effects, and key semantics.>

Examples:
    <Example 1 short title>:
        <example code>

    <Example 2 short title>:
        <example code>

Args:
    <arg_name>: <Description aligned to signature and runtime behavior.>


Returns:
    <Type>: <Exact return behavior and edge cases.>

Notes:
    <Compatibility, deprecation, persistence/scope behavior, caveats, or `None.`>
"""
```

## Signature Consistency Checklist

Before finalizing docstrings, verify:
- Example calls only use real parameters.
- Optional parameters are represented correctly.
- Deprecated aliases are labeled in `Notes`.
- Return description covers `None`/empty cases where applicable.
- Scope/persistence flags match implementation names (`use_persistence` vs legacy names).

## Rewrite Workflow

1. Read method signature and implementation first.
2. Draft docstring using the canonical template.
3. Add two examples matching real call paths.
4. Fill `Args` from actual parameters only.
5. Add caveats/deprecations under `Notes`.
6. Confirm `Returns` matches annotations and code branches.
7. Run lightweight syntax validation (for example `python -m py_compile`) when files changed.

## Preferred Style

- Be concrete and specific; avoid vague wording.
- Prefer short paragraphs and readable line lengths.
- Use backticks for identifiers and literals.
- Keep wording stable across similar APIs.
- For facade mixins, document behavior from the caller perspective.

