"""Provider-neutral raster image admission and bounded preparation."""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
import io
import math
from typing import Any

from .profiles import MultimodalInputPolicy

_PRESERVE_MIME_TYPES = {"image/jpeg", "image/png"}
_SUPPORTED_RASTER_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/gif",
    "image/bmp",
    "image/tiff",
}


class MediaPreparationError(ValueError):
    """Reject unsafe, unsupported, or undecodable model media input."""


@dataclass(frozen=True)
class ImagePreparationPolicy:
    """Central image admission, byte, and raster normalization policy."""

    max_images: int = 4
    max_total_bytes: int = 8_000_000
    max_image_bytes: int | None = None
    accepted_mime_prefixes: tuple[str, ...] = ("image/",)
    accepted_mime_types: tuple[str, ...] = ()
    resize_enabled: bool = True
    max_dimension: int = 1280
    max_pixels: int = 1_500_000
    target_bytes: int | None = None
    jpeg_quality: int = 85
    min_jpeg_quality: int = 70
    shrink_factor: float = 0.85
    max_shrink_steps: int = 6

    def __post_init__(self) -> None:
        """Validate and normalize one image preparation policy.

        Intro:
            Ensures count, byte, geometry, MIME, and bounded recompression
            settings are internally consistent before any image is decoded.

        Examples:
            Validate the default policy:
                ```python
                policy = ImagePreparationPolicy()
                assert policy.max_images == 4
                ```

            Reject an invalid byte limit:
                ```python
                try:
                    ImagePreparationPolicy(max_total_bytes=0)
                except MediaPreparationError:
                    pass
                ```

        Args:
            self: Newly initialized image preparation policy.

        Returns:
            None: Completes after immutable normalized values are stored.

        Notes:
            This validation performs no image decoding and does not import
            Pillow.
        """

        positive = {
            "max_images": self.max_images,
            "max_total_bytes": self.max_total_bytes,
            "max_dimension": self.max_dimension,
            "max_pixels": self.max_pixels,
        }
        for name, value in positive.items():
            if value < 1:
                raise MediaPreparationError(f"image preparation {name} must be at least 1")
        if self.max_image_bytes is not None and self.max_image_bytes < 1:
            raise MediaPreparationError("image preparation max_image_bytes must be at least 1")
        if self.target_bytes is not None and self.target_bytes < 1:
            raise MediaPreparationError("image preparation target_bytes must be at least 1")
        if not (1 <= self.min_jpeg_quality <= self.jpeg_quality <= 95):
            raise MediaPreparationError(
                "image preparation JPEG quality must satisfy 1 <= min <= quality <= 95"
            )
        if not (0 < self.shrink_factor < 1):
            raise MediaPreparationError("image preparation shrink_factor must be between 0 and 1")
        if self.max_shrink_steps < 0:
            raise MediaPreparationError("image preparation max_shrink_steps must be non-negative")
        prefixes = tuple(
            item
            for item in (_normalize_mime(value) for value in self.accepted_mime_prefixes)
            if item
        )
        types = tuple(
            item for item in (_normalize_mime(value) for value in self.accepted_mime_types) if item
        )
        if len(prefixes) != len(set(prefixes)) or len(types) != len(set(types)):
            raise MediaPreparationError("accepted image MIME values must be unique")
        object.__setattr__(self, "accepted_mime_prefixes", prefixes)
        object.__setattr__(self, "accepted_mime_types", types)

    @classmethod
    def from_multimodal_input(
        cls,
        policy: MultimodalInputPolicy,
    ) -> ImagePreparationPolicy:
        """Project one canonical multimodal profile into media preparation.

        Intro:
            Converts the profile-owned image limits exactly once so consumers
            do not repeat defaulting, MIME, resize, or byte-limit decisions.

        Examples:
            Project canonical defaults:
                ```python
                prepared = ImagePreparationPolicy.from_multimodal_input(
                    MultimodalInputPolicy()
                )
                ```

            Project an explicit per-image limit:
                ```python
                prepared = ImagePreparationPolicy.from_multimodal_input(
                    MultimodalInputPolicy(max_images=2, max_image_bytes=100_000)
                )
                assert prepared.max_total_bytes == 200_000
                ```

        Args:
            cls: Image preparation policy class.
            policy: Canonical immutable multimodal input policy.

        Returns:
            ImagePreparationPolicy: Closed preparation policy with resolved
            defaults and total byte admission.

        Notes:
            Model capability resolution remains separate from application-owned
            media admission policy.
        """

        if not isinstance(policy, MultimodalInputPolicy):
            raise TypeError("media preparation requires MultimodalInputPolicy")
        max_images = policy.max_images or 4
        max_total_bytes = (
            policy.max_image_bytes * max_images if policy.max_image_bytes is not None else 8_000_000
        )
        return cls(
            max_images=max_images,
            max_total_bytes=max_total_bytes,
            max_image_bytes=policy.max_image_bytes,
            accepted_mime_prefixes=policy.accepted_mime_prefixes,
            accepted_mime_types=policy.accepted_mime_types,
            resize_enabled=policy.resize_enabled,
            max_dimension=policy.resize_max_dimension,
            max_pixels=policy.resize_max_pixels,
            target_bytes=policy.max_image_bytes,
            jpeg_quality=policy.jpeg_quality,
            min_jpeg_quality=policy.min_jpeg_quality,
        )


@dataclass(frozen=True)
class PreparedImage:
    """One validated raster payload ready for provider wire projection."""

    data: bytes = field(repr=False)
    mime_type: str

    def __post_init__(self) -> None:
        """Validate and detach one prepared image result.

        Intro:
            Ensures prepared payloads remain immutable bytes with a supported,
            normalized raster MIME type before provider projection.

        Examples:
            Validate a prepared PNG:
                ```python
                image = PreparedImage(data=b"png", mime_type="image/png")
                assert image.mime_type == "image/png"
                ```

            Reject an unsupported type:
                ```python
                try:
                    PreparedImage(data=b"svg", mime_type="image/svg+xml")
                except MediaPreparationError:
                    pass
                ```

        Args:
            self: Newly initialized prepared image.

        Returns:
            None: Completes after detached normalized fields are stored.

        Notes:
            Pixel validity is established by `prepare_image_bytes`; this closed
            value validates only its representation contract.
        """

        if not isinstance(self.data, bytes):
            raise TypeError("prepared image data must be bytes")
        if not self.data:
            raise MediaPreparationError("prepared image data must not be empty")
        mime_type = _normalize_image_mime(self.mime_type)
        if mime_type not in _SUPPORTED_RASTER_MIME_TYPES:
            raise MediaPreparationError(f"unsupported prepared image MIME type: {self.mime_type}")
        object.__setattr__(self, "mime_type", mime_type)

    def data_url(self) -> str:
        """Encode the prepared raster as one detached base64 data URL.

        Intro:
            Produces a provider-neutral inline representation only after byte
            validation and normalization have completed.

        Examples:
            Encode a PNG payload:
                ```python
                image = PreparedImage(data=b"png", mime_type="image/png")
                assert image.data_url().startswith("data:image/png;base64,")
                ```

            Confirm deterministic output:
                ```python
                image = PreparedImage(data=b"same", mime_type="image/jpeg")
                assert image.data_url() == image.data_url()
                ```

        Args:
            self: Prepared image payload.

        Returns:
            str: MIME-qualified base64 data URL.

        Notes:
            The returned string contains the image bytes and must not be written
            to ordinary logs or trace metadata.
        """

        encoded = base64.b64encode(self.data).decode("ascii")
        return f"data:{self.mime_type};base64,{encoded}"


def is_accepted_image_mime(mime: str, policy: ImagePreparationPolicy) -> bool:
    """Return whether one declared MIME type passes canonical admission.

    Intro:
        Applies exact MIME allowlisting before prefix matching using normalized,
        case-insensitive media types.

    Examples:
        Accept a default image type:
            ```python
            assert is_accepted_image_mime("image/png", ImagePreparationPolicy())
            ```

        Reject unrelated content:
            ```python
            assert not is_accepted_image_mime("text/plain", ImagePreparationPolicy())
            ```

    Args:
        mime: Declared attachment MIME type.
        policy: Validated image preparation policy.

    Returns:
        bool: `True` when the exact type or one accepted prefix matches.

    Notes:
        Actual raster decoding separately verifies that accepted declarations
        contain a supported image payload.
    """

    normalized = _normalize_mime(mime)
    return normalized in policy.accepted_mime_types or any(
        normalized.startswith(prefix) for prefix in policy.accepted_mime_prefixes
    )


def prepare_image_bytes(
    data: bytes,
    *,
    declared_mime: str,
    byte_limit: int,
    policy: ImagePreparationPolicy,
) -> PreparedImage:
    """Validate, orient, bound, and normalize one raster image payload.

    Intro:
        Decodes one supported raster lazily, applies EXIF orientation, preserves
        safe in-policy PNG/JPEG bytes, and otherwise performs bounded JPEG
        resizing and recompression.

    Examples:
        Prepare a small PNG:
            ```python
            prepared = prepare_image_bytes(
                png_bytes,
                declared_mime="image/png",
                byte_limit=100_000,
                policy=ImagePreparationPolicy(),
            )
            ```

        Reject a non-image payload:
            ```python
            try:
                prepare_image_bytes(
                    b"not an image",
                    declared_mime="image/png",
                    byte_limit=100_000,
                    policy=ImagePreparationPolicy(),
                )
            except MediaPreparationError:
                pass
            ```

    Args:
        data: Detached encoded raster bytes.
        declared_mime: Declared attachment MIME type.
        byte_limit: Hard maximum bytes permitted for the prepared output.
        policy: Validated image preparation policy.

    Returns:
        PreparedImage: Validated bytes and their actual output MIME type.

    Notes:
        Pillow is imported only when this function is called. Install the
        `aethergraph[media]` extra when AG prepares raster input directly.
    """

    if not isinstance(data, bytes):
        raise TypeError("image preparation expects bytes")
    if byte_limit < 1:
        raise MediaPreparationError("image preparation byte_limit must be at least 1")
    normalized_mime = _normalize_image_mime(declared_mime)
    if normalized_mime not in _SUPPORTED_RASTER_MIME_TYPES:
        raise MediaPreparationError(f"unsupported image format: {declared_mime}")

    Image, ImageOps, UnidentifiedImageError = _load_pillow()
    try:
        with Image.open(io.BytesIO(data)) as image:
            source_format = str(image.format or "").upper()
            image.load()
            image = ImageOps.exif_transpose(image)
            if not isinstance(image.size, tuple) or len(image.size) != 2:
                raise MediaPreparationError("image attachment has invalid dimensions")
            width, height = image.size
            if width < 1 or height < 1:
                raise MediaPreparationError("image attachment has invalid dimensions")
            target = _target_bytes(byte_limit=byte_limit, policy=policy)
            needs_resize = (
                width > policy.max_dimension
                or height > policy.max_dimension
                or (width * height) > policy.max_pixels
                or len(data) > target
            )
            if (
                not needs_resize
                and normalized_mime in _PRESERVE_MIME_TYPES
                and _format_matches_mime(source_format, normalized_mime)
            ):
                return PreparedImage(data=data, mime_type=normalized_mime)
            if not policy.resize_enabled:
                if len(data) > byte_limit:
                    raise MediaPreparationError(
                        f"image attachment exceeds byte limit: {len(data)} > {byte_limit}"
                    )
                return PreparedImage(data=data, mime_type=normalized_mime)
            encoded = _resize_and_encode_jpeg(image, policy=policy, byte_limit=byte_limit)
            return PreparedImage(data=encoded, mime_type="image/jpeg")
    except MediaPreparationError:
        raise
    except UnidentifiedImageError as exc:
        raise MediaPreparationError("image attachment could not be decoded as an image") from exc
    except OSError as exc:
        raise MediaPreparationError("image attachment could not be decoded as an image") from exc


def _resize_and_encode_jpeg(
    image: Any,
    *,
    policy: ImagePreparationPolicy,
    byte_limit: int,
) -> bytes:
    target = _target_bytes(byte_limit=byte_limit, policy=policy)
    working = _resize_to_policy_bounds(image, policy)
    qualities = _dedupe_ints(
        (
            policy.jpeg_quality,
            max(policy.min_jpeg_quality, min(policy.jpeg_quality, 78)),
            policy.min_jpeg_quality,
        )
    )
    last_size = 0
    for _step in range(policy.max_shrink_steps + 1):
        rgb = _jpeg_ready_image(working)
        for quality in qualities:
            encoded = _encode_jpeg(rgb, quality=quality)
            last_size = len(encoded)
            if last_size <= target:
                return encoded
        next_size = (
            max(1, int(working.size[0] * policy.shrink_factor)),
            max(1, int(working.size[1] * policy.shrink_factor)),
        )
        if next_size == working.size:
            break
        working = working.resize(next_size, resample=_resample_filter())
    raise MediaPreparationError(
        f"image attachment exceeds byte limit after resize: {last_size} > {target}"
    )


def _resize_to_policy_bounds(image: Any, policy: ImagePreparationPolicy) -> Any:
    width, height = image.size
    scale = min(
        1.0,
        policy.max_dimension / max(width, height),
        math.sqrt(policy.max_pixels / (width * height)),
    )
    if scale >= 1:
        return image.copy()
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(new_size, resample=_resample_filter())


def _jpeg_ready_image(image: Any) -> Any:
    if image.mode in {"RGBA", "LA"} or (
        image.mode == "P" and "transparency" in getattr(image, "info", {})
    ):
        rgba = image.convert("RGBA")
        background = _new_rgb_image(rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    return image if image.mode == "RGB" else image.convert("RGB")


def _encode_jpeg(image: Any, *, quality: int) -> bytes:
    buffer = io.BytesIO()
    try:
        image.save(buffer, format="JPEG", quality=quality, optimize=True)
    except OSError as exc:
        raise MediaPreparationError("image attachment could not be encoded as JPEG") from exc
    return buffer.getvalue()


def _dedupe_ints(values: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(dict.fromkeys(int(value) for value in values))


def _target_bytes(*, byte_limit: int, policy: ImagePreparationPolicy) -> int:
    target = policy.target_bytes if policy.target_bytes is not None else byte_limit
    return min(target, byte_limit)


def _normalize_mime(mime: str) -> str:
    return str(mime or "").strip().lower().split(";", 1)[0]


def _normalize_image_mime(mime: str) -> str:
    normalized = _normalize_mime(mime)
    return "image/jpeg" if normalized == "image/jpg" else normalized


def _format_matches_mime(source_format: str, mime: str) -> bool:
    if mime == "image/jpeg":
        return source_format in {"JPEG", "JPG"}
    return mime == "image/png" and source_format == "PNG"


def _load_pillow() -> tuple[Any, Any, type[Exception]]:
    try:
        from PIL import Image, ImageOps, UnidentifiedImageError
    except ImportError as exc:
        raise MediaPreparationError(
            "image preparation requires the optional aethergraph[media] dependency"
        ) from exc
    return Image, ImageOps, UnidentifiedImageError


def _resample_filter() -> Any:
    Image, _ImageOps, _UnidentifiedImageError = _load_pillow()
    return Image.Resampling.LANCZOS


def _new_rgb_image(size: tuple[int, int], color: tuple[int, int, int]) -> Any:
    Image, _ImageOps, _UnidentifiedImageError = _load_pillow()
    return Image.new("RGB", size, color)


__all__ = [
    "ImagePreparationPolicy",
    "MediaPreparationError",
    "PreparedImage",
    "is_accepted_image_mime",
    "prepare_image_bytes",
]
