from __future__ import annotations

from aethergraph.api.v1.session import _row_to_session_chat_event


def test_session_chat_event_presenter_forwards_attachments() -> None:
    event = _row_to_session_chat_event(
        {
            "id": "event-1",
            "ts": 123.0,
            "payload": {
                "type": "user.message",
                "text": "look at this",
                "files": [
                    {
                        "artifact_id": "display-artifact",
                        "name": "display.png",
                        "mimetype": "image/png",
                    }
                ],
                "attachments": [
                    {
                        "artifact_id": "resource-artifact",
                        "name": "source.png",
                        "mimetype": "image/png",
                        "url": "/api/v1/artifacts/resource-artifact/content",
                    }
                ],
            },
        },
        "session-1",
    )

    assert event.files is not None
    assert event.files[0].artifact_id == "display-artifact"
    assert event.attachments is not None
    assert event.attachments[0].artifact_id == "resource-artifact"
    assert event.attachments[0].url == "/api/v1/artifacts/resource-artifact/content"


def test_session_chat_event_presenter_dedupes_attachment_shown_as_file() -> None:
    """A context ref that is already an inline display file must not also render
    as a duplicate 'context' pill (matched by explicit id or content URL)."""
    event = _row_to_session_chat_event(
        {
            "id": "event-2",
            "ts": 1.0,
            "payload": {
                "type": "user.message",
                "text": "look at this",
                "files": [
                    {"artifact_id": "img-1", "name": "img.png", "mimetype": "image/png"},
                ],
                "attachments": [
                    # Same artifact as the display file -> dropped.
                    {"artifact_id": "img-1", "name": "img.png"},
                    # Same artifact, referenced by content URL only -> dropped.
                    {"url": "/api/v1/artifacts/img-1/content", "name": "img.png"},
                    # A distinct context ref (e.g. a doc) -> kept.
                    {"artifact_id": "doc-9", "name": "notes.txt"},
                ],
                "meta": {
                    "role": "user",
                    "attachments": [
                        {"artifact_id": "img-1", "name": "img.png"},
                        {"artifact_id": "doc-9", "name": "notes.txt"},
                    ],
                },
            },
        },
        "session-2",
    )

    assert event.attachments is not None
    assert [a.artifact_id for a in event.attachments] == ["doc-9"]
    # meta.attachments (read first by the frontend) is deduped the same way,
    # while other meta keys are preserved.
    assert event.meta["role"] == "user"
    assert [a["artifact_id"] for a in event.meta["attachments"]] == ["doc-9"]


def test_session_chat_event_presenter_drops_attachments_when_all_are_files() -> None:
    event = _row_to_session_chat_event(
        {
            "id": "event-3",
            "ts": 2.0,
            "payload": {
                "type": "user.message",
                "text": "",
                "files": [{"artifact_id": "only-1", "name": "a.png"}],
                "attachments": [{"artifact_id": "only-1", "name": "a.png"}],
            },
        },
        "session-3",
    )

    assert event.files is not None
    assert event.attachments is None
