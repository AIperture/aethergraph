from fastapi import HTTPException


def decode_cursor(cursor: str | None) -> int:
    """
    Turn an opaque cursor string into an integer offset.

    For now, cursor is just the stringified offest. Later we will
    switch to base64 JSON or keyset pagination without changing
    the endpoints
    """
    if not cursor:
        return 0

    try:
        return int(cursor)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid cursor") from e


def encode_cursor(offset: int) -> str:
    """
    Turn an integer offset into an opaque cursor string.

    For now, cursor is just the stringified offest. Later we will
    switch to base64 JSON or keyset pagination without changing
    the endpoints
    """
    return str(offset)
