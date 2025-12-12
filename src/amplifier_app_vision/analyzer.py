"""OpenAI Vision API integration for image analysis."""

import logging
import os
from pathlib import Path
from typing import Optional

from openai import OpenAI

from .image_processor import ImageProcessor

logger = logging.getLogger(__name__)


class VisionAnalyzer:
    """Analyzes images using OpenAI's GPT-4 Vision API."""

    DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant that analyzes images. 
Provide clear, detailed descriptions and answer questions accurately based on what you see.
Be concise but thorough."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        quality: str = "normal",
    ):
        """Initialize vision analyzer.
        
        Args:
            api_key: OpenAI API key (or uses OPENAI_API_KEY env var)
            model: OpenAI model with vision capabilities
            quality: Image quality preset
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.model = model
        self.processor = ImageProcessor(quality=quality)
        self._client = OpenAI(api_key=self.api_key)

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
            Dict with analysis result and metadata
        """
        # Process image
        image_source_str = str(image_source)
        
        if image_source_str.startswith(("http://", "https://")):
            image_data = self.processor.process_url(image_source_str)
            image_content = {
                "type": "image_url",
                "image_url": {"url": image_data["url"]},
            }
        else:
            image_data = self.processor.process_file(Path(image_source))
            image_content = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{image_data['media_type']};base64,{image_data['data']}"
                },
            }
        
        # Build messages
        messages = [
            {"role": "system", "content": system_prompt or self.DEFAULT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    image_content,
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        
        # Call API
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
            )
            
            result_text = response.choices[0].message.content
            
            return {
                "text": result_text,
                "model": self.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
                "image_metadata": image_data.get("metadata", {}),
            }
            
        except Exception as e:
            logger.error(f"Vision analysis failed: {e}")
            raise

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
            Dict with analysis result and metadata
        """
        content = []
        all_metadata = []
        
        # Process each image
        for source in image_sources:
            source_str = str(source)
            
            if source_str.startswith(("http://", "https://")):
                image_data = self.processor.process_url(source_str)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": image_data["url"]},
                })
            else:
                image_data = self.processor.process_file(Path(source))
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{image_data['media_type']};base64,{image_data['data']}"
                    },
                })
            
            all_metadata.append(image_data.get("metadata", {}))
        
        # Add text prompt
        content.append({"type": "text", "text": prompt})
        
        # Build messages
        messages = [
            {"role": "system", "content": system_prompt or self.DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]
        
        # Call API
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
            )
            
            result_text = response.choices[0].message.content
            
            return {
                "text": result_text,
                "model": self.model,
                "image_count": len(image_sources),
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
                "images_metadata": all_metadata,
            }
            
        except Exception as e:
            logger.error(f"Multi-image analysis failed: {e}")
            raise

    def describe(self, image_source: str | Path) -> str:
        """Get a description of an image.
        
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
