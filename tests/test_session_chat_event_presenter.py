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
