"""Anthropic Claude Vision provider."""

import logging
from typing import Optional

from anthropic import Anthropic

from .base import VisionProvider, VisionRequest, VisionResponse

logger = logging.getLogger(__name__)


class AnthropicProvider(VisionProvider):
    """Anthropic Claude Vision implementation."""
    
    provider_name = "anthropic"
    
    SUPPORTED_MODELS = [
        "claude-sonnet-4-20250514",
        "claude-3-5-sonnet-latest",
        "claude-3-5-sonnet-20241022",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
    ]
    
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514") -> None:
        """Initialize Anthropic provider.
        
        Args:
            api_key: Anthropic API key
            model: Model identifier (default: claude-sonnet-4-20250514)
        """
        self.api_key = api_key
        self.model = model
        self._client = Anthropic(api_key=api_key)
    
    @property
    def supported_models(self) -> list[str]:
        return self.SUPPORTED_MODELS
    
    def analyze(self, request: VisionRequest) -> VisionResponse:
        """Analyze images using Anthropic's vision API.
        
        Args:
            request: Normalized vision request
            
        Returns:
            Normalized vision response
            
        Note:
            Anthropic requires base64 images - URLs are not directly supported.
            The image processor should convert URLs to base64 before calling.
        """
        # Build content array with images and prompt
        content = []
        
        for image in request.images:
            if "url" in image:
                # Anthropic doesn't support URL images directly
                # Should have been converted to base64 by image processor
                raise ValueError(
                    "Anthropic doesn't support URL images directly. "
                    "Use a local file path instead."
                )
            else:
                # Base64 encoded image - Anthropic format
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": image["media_type"],
                        "data": image["data"],
                    },
                })
        
        content.append({"type": "text", "text": request.prompt})
        
        # Build messages (system prompt is separate in Anthropic API)
        messages = [{"role": "user", "content": content}]
        
        # Call API
        try:
            response = self._client.messages.create(
                model=self.model,
                system=request.system_prompt or "",
                messages=messages,
                max_tokens=request.max_tokens,
            )
            
            # Extract text from response
            result_text = ""
            for block in response.content:
                if block.type == "text":
                    result_text += block.text
            
            return VisionResponse(
                text=result_text,
                model=self.model,
                usage={
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
                },
                raw_response=response.model_dump() if hasattr(response, 'model_dump') else None,
            )
            
        except Exception as e:
            logger.error(f"Anthropic vision request failed: {e}")
            raise
