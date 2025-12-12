"""Vision analysis with multi-provider support."""

import logging
from pathlib import Path
from typing import Optional

from .image_processor import ImageProcessor
from .providers.base import VisionRequest
from .providers.router import ProviderRouter

logger = logging.getLogger(__name__)


class VisionAnalyzer:
    """Analyzes images using vision-capable AI models.
    
    Supports multiple providers (OpenAI, Anthropic, Google) with automatic
    detection based on model name.
    
    Examples:
        # Auto-detect provider from model
        analyzer = VisionAnalyzer(model="gpt-4o")        # OpenAI
        analyzer = VisionAnalyzer(model="claude-sonnet-4-20250514")  # Anthropic
        analyzer = VisionAnalyzer(model="gemini-1.5-flash")  # Google
        
        # Use default (OpenAI gpt-4o)
        analyzer = VisionAnalyzer()
        
        # Analyze an image
        result = analyzer.analyze("photo.jpg", prompt="What's in this image?")
        print(result["text"])
    """

    DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant that analyzes images. 
Provide clear, detailed descriptions and answer questions accurately based on what you see.
Be concise but thorough."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        quality: str = "normal",
        provider: Optional[str] = None,
    ):
        """Initialize vision analyzer.
        
        Args:
            api_key: API key (auto-detects which provider based on model)
            model: Model identifier - auto-detects provider from name:
                   - gpt-* → OpenAI
                   - claude-* → Anthropic
                   - gemini-* → Google
            quality: Image quality preset (quick, normal, detailed, full)
            provider: Force specific provider (overrides auto-detection)
        """
        self.model = model
        self.processor = ImageProcessor(quality=quality)
        
        # Use router to get appropriate provider
        self._router = ProviderRouter()
        self._provider = self._router.get_provider(
            model=model,
            provider=provider,
            api_key=api_key,
        )

    def _process_image(self, image_source: str | Path) -> dict:
        """Process image source into normalized format.
        
        Args:
            image_source: File path or URL
            
        Returns:
            Dict with image data and metadata
        """
        source_str = str(image_source)
        
        if source_str.startswith(("http://", "https://")):
            return self.processor.process_url(source_str)
        else:
            return self.processor.process_file(Path(image_source))

    def _image_to_request_format(self, image_data: dict) -> dict:
        """Convert processed image to request format.
        
        Args:
            image_data: Processed image data from ImageProcessor
            
        Returns:
            Dict in format expected by VisionRequest.images
        """
        if "url" in image_data:
            return {"url": image_data["url"]}
        else:
            return {
                "data": image_data["data"],
                "media_type": image_data["media_type"],
            }

    def analyze(
        self,
        image_source: str | Path,
        prompt: str = "What's in this image?",
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
    ) -> dict:
        """Analyze an image with a prompt.
        
        Args:
            image_source: File path or URL to image
            prompt: Question or instruction about the image
            system_prompt: Optional system prompt override
            max_tokens: Maximum response tokens
            
        Returns:
            Dict with:
                - text: Analysis result
                - model: Model used
                - usage: Token usage stats
                - image_metadata: Image processing info
        """
        # Process image
        image_data = self._process_image(image_source)
        
        # Build request
        request = VisionRequest(
            images=[self._image_to_request_format(image_data)],
            prompt=prompt,
            system_prompt=system_prompt or self.DEFAULT_SYSTEM_PROMPT,
            max_tokens=max_tokens,
        )
        
        # Delegate to provider
        response = self._provider.analyze(request)
        
        # Return in expected format
        return {
            "text": response.text,
            "model": response.model,
            "usage": response.usage,
            "image_metadata": image_data.get("metadata", {}),
        }

    def analyze_multiple(
        self,
        image_sources: list[str | Path],
        prompt: str = "Describe these images and their relationship.",
        system_prompt: Optional[str] = None,
        max_tokens: int = 2048,
    ) -> dict:
        """Analyze multiple images together.
        
        Args:
            image_sources: List of file paths or URLs
            prompt: Question or instruction about the images
            system_prompt: Optional system prompt override
            max_tokens: Maximum response tokens
            
        Returns:
            Dict with:
                - text: Analysis result
                - model: Model used
                - image_count: Number of images
                - usage: Token usage stats
                - images_metadata: List of image processing info
        """
        images = []
        all_metadata = []
        
        # Process each image
        for source in image_sources:
            image_data = self._process_image(source)
            images.append(self._image_to_request_format(image_data))
            all_metadata.append(image_data.get("metadata", {}))
        
        # Build request
        request = VisionRequest(
            images=images,
            prompt=prompt,
            system_prompt=system_prompt or self.DEFAULT_SYSTEM_PROMPT,
            max_tokens=max_tokens,
        )
        
        # Delegate to provider
        response = self._provider.analyze(request)
        
        # Return in expected format
        return {
            "text": response.text,
            "model": response.model,
            "image_count": len(image_sources),
            "usage": response.usage,
            "images_metadata": all_metadata,
        }

    def describe(self, image_source: str | Path) -> str:
        """Get a detailed description of an image.
        
        Args:
            image_source: File path or URL
            
        Returns:
            Text description
        """
        result = self.analyze(
            image_source,
            prompt="Describe this image in detail. Include what you see, any text, colors, and important elements.",
        )
        return result["text"]

    def extract_text(self, image_source: str | Path) -> str:
        """Extract text from an image (OCR-like).
        
        Args:
            image_source: File path or URL
            
        Returns:
            Extracted text
        """
        result = self.analyze(
            image_source,
            prompt="Extract and transcribe all text visible in this image. Format it cleanly, preserving structure where possible.",
        )
        return result["text"]

    def answer_question(self, image_source: str | Path, question: str) -> str:
        """Answer a specific question about an image.
        
        Args:
            image_source: File path or URL
            question: Question to answer
            
        Returns:
            Answer text
        """
        result = self.analyze(image_source, prompt=question)
        return result["text"]
