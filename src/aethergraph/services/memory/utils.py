def _summary_prefix(scope_id: str, summary_tag: str) -> str:
    return f"mem/{scope_id}/summaries/{summary_tag}/"


def _summary_doc_id(scope_id: str, summary_tag: str, ts: str) -> str:
    """
    Build a doc_id for a summary. We assume `ts` is an ISO-ish string
    (e.g. from now_iso()) and rely on lexicographic ordering.
    """
    return f"{_summary_prefix(scope_id, summary_tag)}{ts}"
