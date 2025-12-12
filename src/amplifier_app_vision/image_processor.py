"""Image processing and optimization for vision API."""

import base64
import hashlib
import io
import logging
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from PIL import Image

logger = logging.getLogger(__name__)

# Quality presets: (max_dimension, jpeg_quality, estimated_tokens_per_image)
QUALITY_PRESETS = {
    "quick": (512, 75, 500),      # Fast analysis, low tokens
    "normal": (1024, 85, 1500),   # Balanced (default)
    "detailed": (1568, 90, 4000), # High detail, more tokens
    "full": (None, 95, 8000),     # No resize, max quality
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
IMAGE_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


class ImageProcessor:
    """Processes images for vision API with token-efficient optimization."""

    def __init__(
        self,
        quality: str = "normal",
        max_dimension: Optional[int] = None,
        jpeg_quality: Optional[int] = None,
    ):
        """Initialize image processor.
        
        Args:
            quality: Quality preset - "quick", "normal", "detailed", "full"
            max_dimension: Override max image dimension in pixels
            jpeg_quality: JPEG compression quality 1-100
        """
        preset = QUALITY_PRESETS.get(quality, QUALITY_PRESETS["normal"])
        
        self.quality = quality
        self.max_dimension = max_dimension or preset[0]
        self.jpeg_quality = jpeg_quality or preset[1]
        self.tokens_per_image = preset[2]
        
        # Cache for deduplication
        self._cache: dict[str, tuple[str, str, dict]] = {}

    def process_file(self, file_path: Path) -> dict:
        """Process a local image file.
        
        Args:
            file_path: Path to image file
            
        Returns:
            Dict with base64 data, media_type, and metadata
        """
        path = Path(file_path).expanduser().resolve()
        
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {file_path}")
        
        if not path.is_file():
            raise ValueError(f"Not a file: {file_path}")
        
        ext = path.suffix.lower()
        if ext not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image format: {ext}")
        
        # Read and hash for caching
        file_bytes = path.read_bytes()
        file_hash = hashlib.md5(file_bytes).hexdigest()
        
        # Check cache
        if file_hash in self._cache:
            logger.debug(f"Using cached image: {path.name}")
            cached = self._cache[file_hash]
            return {
                "data": cached[0],
                "media_type": cached[1],
                "metadata": cached[2],
                "cached": True,
            }
        
        # Process image
        optimized_bytes, media_type = self._optimize_image(file_bytes, ext)
        b64_data = base64.standard_b64encode(optimized_bytes).decode("utf-8")
        
        # Calculate metadata
        original_size = len(file_bytes)
        optimized_size = len(optimized_bytes)
        compression = (1 - optimized_size / original_size) * 100 if original_size else 0
        estimated_tokens = self._estimate_tokens(optimized_size)
        
        metadata = {
            "filename": path.name,
            "original_size": original_size,
            "optimized_size": optimized_size,
            "compression_percent": compression,
            "estimated_tokens": estimated_tokens,
        }
        
        # Cache result
        self._cache[file_hash] = (b64_data, media_type, metadata)
        
        logger.info(
            f"Processed: {path.name} "
            f"({original_size:,} -> {optimized_size:,} bytes, {compression:.0f}% smaller)"
        )
        
        return {
            "data": b64_data,
            "media_type": media_type,
            "metadata": metadata,
            "cached": False,
        }

    def process_url(self, url: str) -> dict:
        """Process an image URL.
        
        Args:
            url: Image URL
            
        Returns:
            Dict with URL and metadata
        """
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"Invalid URL: {url}")
        
        # Determine media type from extension
        path = parsed.path.lower()
        media_type = "image/jpeg"  # default
        for ext, mtype in IMAGE_MEDIA_TYPES.items():
            if path.endswith(ext):
                media_type = mtype
                break
        
        return {
            "url": url,
            "media_type": media_type,
            "metadata": {
                "source": "url",
                "estimated_tokens": self.tokens_per_image,
            },
        }

    def _optimize_image(self, image_bytes: bytes, original_ext: str) -> tuple[bytes, str]:
        """Resize and compress image for token efficiency.
        
        Returns: (optimized_bytes, media_type)
        """
        img = Image.open(io.BytesIO(image_bytes))
        original_size = img.size
        
        # Resize if needed
        if self.max_dimension and max(img.size) > self.max_dimension:
            ratio = self.max_dimension / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            logger.debug(f"Resized: {original_size} -> {new_size}")
        
        # Determine output format
        has_transparency = img.mode in ("RGBA", "LA") or (
            img.mode == "P" and "transparency" in img.info
        )
        
        output = io.BytesIO()
        
        if has_transparency:
            # Keep as PNG for transparency
            if img.mode == "P":
                img = img.convert("RGBA")
            img.save(output, format="PNG", optimize=True)
            media_type = "image/png"
        else:
            # Convert to JPEG for better compression
            if img.mode in ("RGBA", "LA", "P"):
                img = img.convert("RGB")
            img.save(output, format="JPEG", quality=self.jpeg_quality, optimize=True)
            media_type = "image/jpeg"
        
        return output.getvalue(), media_type

    def _estimate_tokens(self, byte_size: int) -> int:
        """Estimate token count based on image byte size."""
        base_tokens = 85
        size_tokens = byte_size // 150
        return min(base_tokens + size_tokens, 10000)

    def get_image_info(self, file_path: Path) -> dict:
        """Get information about an image without processing.
        
        Args:
            file_path: Path to image file
            
        Returns:
            Dict with image information
        """
        path = Path(file_path).expanduser().resolve()
        
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {file_path}")
        
        file_bytes = path.read_bytes()
        img = Image.open(io.BytesIO(file_bytes))
        
        return {
            "filename": path.name,
            "format": img.format,
            "mode": img.mode,
            "size": img.size,
            "width": img.size[0],
            "height": img.size[1],
            "file_size": len(file_bytes),
        }
