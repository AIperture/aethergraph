from __future__ import annotations

from email.message import Message

import pytest

from aethergraph.services.artifacts import canonical_public


def _public_dns(*_args, **_kwargs):
    return [(None, None, None, None, ("93.184.216.34", 443))]


def test_remote_artifact_url_rejects_credentials_and_private_addresses(monkeypatch) -> None:
    monkeypatch.setattr(canonical_public.socket, "getaddrinfo", _public_dns)
    with pytest.raises(ValueError, match="credentials"):
        canonical_public._validate_remote_url("https://user:secret@example.test/report")

    monkeypatch.setattr(
        canonical_public.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [(None, None, None, None, ("127.0.0.1", 80))],
    )
    with pytest.raises(ValueError, match="non-public"):
        canonical_public._validate_remote_url("http://example.test/report")


def test_remote_artifact_download_is_bounded_and_sanitizes_source(monkeypatch) -> None:
    headers = Message()
    headers["Content-Type"] = "text/csv; charset=utf-8"
    headers["Content-Length"] = "12"
    headers["Content-Disposition"] = 'attachment; filename="report.csv"'

    class _Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def geturl(self):
            return "https://example.test/files/report.csv?signature=secret#fragment"

        def read(self, _size):
            return b"name,value\na,1\n"

        @property
        def headers(self):
            return headers

    class _Opener:
        def open(self, _request, *, timeout):
            assert timeout == 2.0
            return _Response()

    monkeypatch.setattr(canonical_public.socket, "getaddrinfo", _public_dns)
    monkeypatch.setattr(canonical_public, "build_opener", lambda *_args: _Opener())

    payload, mime, filename, source = canonical_public._download_remote_file(
        "https://example.test/files/report.csv?token=secret",
        timeout_s=2.0,
        max_bytes=1_024,
    )

    assert payload == b"name,value\na,1\n"
    assert mime == "text/csv"
    assert filename == "report.csv"
    assert source == "https://example.test/files/report.csv"


def test_remote_artifact_download_rejects_oversized_declared_body(monkeypatch) -> None:
    headers = Message()
    headers["Content-Type"] = "application/octet-stream"
    headers["Content-Length"] = "2048"

    class _Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def geturl(self):
            return "https://example.test/file.bin"

        def read(self, _size):
            raise AssertionError("oversized content must fail before reading")

        @property
        def headers(self):
            return headers

    class _Opener:
        def open(self, _request, *, timeout):
            return _Response()

    monkeypatch.setattr(canonical_public.socket, "getaddrinfo", _public_dns)
    monkeypatch.setattr(canonical_public, "build_opener", lambda *_args: _Opener())

    with pytest.raises(ValueError, match="exceeds max_bytes"):
        canonical_public._download_remote_file(
            "https://example.test/file.bin",
            timeout_s=2.0,
            max_bytes=1_024,
        )
