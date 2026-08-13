from __future__ import annotations

import io

from PIL import Image
import pytest

from aethergraph.services.llm import (
    ImagePart,
    ImagePreparationPolicy,
    MediaPreparationError,
    MultimodalInputPolicy,
    is_accepted_image_mime,
    prepare_image_bytes,
    prepare_image_inputs,
)


def _image_bytes(
    *,
    format_name: str = "PNG",
    size: tuple[int, int] = (8, 8),
    mode: str = "RGB",
) -> bytes:
    image = Image.new(mode, size, (160, 40, 40))
    buffer = io.BytesIO()
    image.save(buffer, format=format_name)
    return buffer.getvalue()


def test_multimodal_policy_projects_once_into_media_preparation() -> None:
    policy = ImagePreparationPolicy.from_multimodal_input(
        MultimodalInputPolicy(
            image_input_enabled=True,
            allow_remote_urls=True,
            max_images=2,
            max_image_bytes=100_000,
            accepted_mime_types=("image/png",),
            resize_max_dimension=512,
            resize_max_pixels=200_000,
            jpeg_quality=82,
            min_jpeg_quality=68,
        )
    )

    assert policy.max_images == 2
    assert policy.image_input_enabled is True
    assert policy.allow_remote_urls is True
    assert policy.max_total_bytes == 200_000
    assert policy.target_bytes == 100_000
    assert policy.max_dimension == 512
    assert policy.max_pixels == 200_000
    assert policy.jpeg_quality == 82
    assert policy.min_jpeg_quality == 68
    assert is_accepted_image_mime("IMAGE/PNG; charset=binary", policy)


def test_whole_request_media_policy_rejects_disabled_and_remote_images() -> None:
    remote = (ImagePart(url="https://example.test/image.png"),)

    with pytest.raises(MediaPreparationError, match="disabled"):
        prepare_image_inputs(
            remote,
            policy=ImagePreparationPolicy(image_input_enabled=False),
        )
    with pytest.raises(MediaPreparationError, match="remote image URLs"):
        prepare_image_inputs(remote, policy=ImagePreparationPolicy())

    assert (
        prepare_image_inputs(
            remote,
            policy=ImagePreparationPolicy(allow_remote_urls=True),
        )
        == remote
    )


def test_whole_request_media_policy_normalizes_inline_images_and_count() -> None:
    original = _image_bytes()
    policy = ImagePreparationPolicy(max_images=1, max_image_bytes=100_000)

    prepared = prepare_image_inputs(
        (ImagePart(data=original, mime_type="image/png"),),
        policy=policy,
    )

    assert prepared[0].data == original
    assert prepared[0].mime_type == "image/png"
    with pytest.raises(MediaPreparationError, match="image count"):
        prepare_image_inputs(
            (
                ImagePart(data=original, mime_type="image/png"),
                ImagePart(data=original, mime_type="image/png"),
            ),
            policy=policy,
        )


def test_canonical_quality_range_projects_without_narrowing() -> None:
    policy = ImagePreparationPolicy.from_multimodal_input(
        MultimodalInputPolicy(jpeg_quality=100, min_jpeg_quality=96)
    )

    assert policy.jpeg_quality == 100
    assert policy.min_jpeg_quality == 96


def test_small_safe_png_is_preserved_and_encoded_only_on_projection() -> None:
    original = _image_bytes()

    prepared = prepare_image_bytes(
        original,
        declared_mime="image/png",
        byte_limit=100_000,
        policy=ImagePreparationPolicy(),
    )

    assert prepared.data == original
    assert prepared.mime_type == "image/png"
    assert prepared.data_url().startswith("data:image/png;base64,")
    assert original.hex() not in repr(prepared)


def test_oversized_raster_is_bounded_and_normalized_to_jpeg() -> None:
    original = _image_bytes(size=(1200, 800))
    policy = ImagePreparationPolicy(
        max_dimension=128,
        max_pixels=16_384,
        max_image_bytes=20_000,
        target_bytes=20_000,
    )

    prepared = prepare_image_bytes(
        original,
        declared_mime="image/png",
        byte_limit=20_000,
        policy=policy,
    )

    assert prepared.mime_type == "image/jpeg"
    assert len(prepared.data) <= 20_000
    with Image.open(io.BytesIO(prepared.data)) as image:
        assert max(image.size) <= 128
        assert image.size[0] * image.size[1] <= 16_384


@pytest.mark.parametrize(
    ("data", "mime", "message"),
    [
        (b"not an image", "image/png", "could not be decoded"),
        (_image_bytes(), "application/octet-stream", "unsupported image format"),
    ],
)
def test_invalid_or_unsupported_media_fails_before_provider_projection(
    data: bytes,
    mime: str,
    message: str,
) -> None:
    with pytest.raises(MediaPreparationError, match=message):
        prepare_image_bytes(
            data,
            declared_mime=mime,
            byte_limit=100_000,
            policy=ImagePreparationPolicy(),
        )


def test_disabled_resize_rejects_payload_over_hard_limit() -> None:
    data = _image_bytes(format_name="BMP", size=(256, 256))

    with pytest.raises(MediaPreparationError, match="exceeds byte limit"):
        prepare_image_bytes(
            data,
            declared_mime="image/bmp",
            byte_limit=1_000,
            policy=ImagePreparationPolicy(resize_enabled=False),
        )
