"""Multipart projection of already prepared canonical image bytes."""

from aethergraph.services.llm.types import ImageInput


def image_edit_files(images: tuple[ImageInput, ...]) -> list[tuple[str, tuple[str, bytes, str]]]:
    """Project validated inline image inputs into ordered multipart file fields.

    Examples:
        `image_edit_files((ImageInput(data=b"bytes", mime_type="image/png"),))`.

    Args:
        images: Canonical images already admitted by AG media preparation.

    Returns:
        Ordered image[] file fields for the existing HTTP client.

    Notes:
        This function does not fetch, resize, authorize, or retry media.
    """
    result = []
    for index, image in enumerate(images):
        if image.data is None or image.mime_type not in {"image/png", "image/jpeg"}:
            raise ValueError("image edit transport requires prepared PNG/JPEG bytes")
        extension = "png" if image.mime_type == "image/png" else "jpg"
        result.append(("image[]", (f"input-{index}.{extension}", image.data, image.mime_type)))
    return result
