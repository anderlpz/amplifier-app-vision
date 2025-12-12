"""OpenAI GPT-4 Vision provider."""

import logging
from typing import Optional

from openai import OpenAI

from .base import VisionProvider, VisionRequest, VisionResponse

logger = logging.getLogger(__name__)


class OpenAIProvider(VisionProvider):
    """OpenAI GPT-4 Vision implementation."""
    
    provider_name = "openai"
    
    SUPPORTED_MODELS = [
        "gpt-4o",
        "gpt-4o-mini", 
        "gpt-4-turbo",
        "gpt-4-turbo-2024-04-09",
        "gpt-4-vision-preview",
    ]
    
    def __init__(self, api_key: str, model: str = "gpt-4o") -> None:
        """Initialize OpenAI provider.
        
        Args:
            api_key: OpenAI API key
            model: Model identifier (default: gpt-4o)
        """
        self.api_key = api_key
        self.model = model
        self._client = OpenAI(api_key=api_key)
    
    @property
    def supported_models(self) -> list[str]:
        return self.SUPPORTED_MODELS
    
    def analyze(self, request: VisionRequest) -> VisionResponse:
        """Analyze images using OpenAI's vision API.
        
        Args:
            request: Normalized vision request
            
        Returns:
            Normalized vision response
        """
        # Build content array with images and prompt
        content = []
        
        for image in request.images:
            if "url" in image:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": image["url"]},
                })
            else:
                # Base64 encoded image
                data_url = f"data:{image['media_type']};base64,{image['data']}"
                content.append({
                    "type": "image_url",
                    "image_url": {"url": data_url},
                })
        
        content.append({"type": "text", "text": request.prompt})
        
        # Build messages
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": content})
        
        # Call API
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=request.max_tokens,
            )
            
            return VisionResponse(
                text=response.choices[0].message.content,
                model=self.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
                raw_response=response.model_dump() if hasattr(response, 'model_dump') else None,
            )
            
        except Exception as e:
            logger.error(f"OpenAI vision request failed: {e}")
            raise
