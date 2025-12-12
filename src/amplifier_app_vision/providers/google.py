"""Google Gemini Vision provider."""

import base64
import logging
from typing import Optional

import google.generativeai as genai

from .base import VisionProvider, VisionRequest, VisionResponse

logger = logging.getLogger(__name__)


class GoogleProvider(VisionProvider):
    """Google Gemini Vision implementation."""
    
    provider_name = "google"
    
    SUPPORTED_MODELS = [
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "gemini-2.0-flash-exp",
        "gemini-1.5-pro-latest",
        "gemini-1.5-flash-latest",
    ]
    
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash") -> None:
        """Initialize Google provider.
        
        Args:
            api_key: Google API key
            model: Model identifier (default: gemini-1.5-flash)
        """
        self.api_key = api_key
        self.model = model
        
        # Configure the SDK with API key
        genai.configure(api_key=api_key)
        self._model = None  # Lazy init with system prompt
    
    @property
    def supported_models(self) -> list[str]:
        return self.SUPPORTED_MODELS
    
    def _get_model(self, system_prompt: Optional[str] = None) -> genai.GenerativeModel:
        """Get or create model instance with system prompt.
        
        Args:
            system_prompt: Optional system instruction
            
        Returns:
            Configured GenerativeModel
        """
        return genai.GenerativeModel(
            model_name=self.model,
            system_instruction=system_prompt,
        )
    
    def analyze(self, request: VisionRequest) -> VisionResponse:
        """Analyze images using Google's Gemini API.
        
        Args:
            request: Normalized vision request
            
        Returns:
            Normalized vision response
        """
        # Build content array with images and prompt
        content = []
        
        for image in request.images:
            if "url" in image:
                # Google supports URLs but let's use base64 for consistency
                raise ValueError(
                    "Google provider expects base64 images. "
                    "Use a local file path instead."
                )
            else:
                # Base64 encoded image - Google format
                content.append({
                    "mime_type": image["media_type"],
                    "data": base64.b64decode(image["data"]),
                })
        
        content.append(request.prompt)
        
        # Get model with system prompt
        model = self._get_model(request.system_prompt)
        
        # Call API
        try:
            response = model.generate_content(
                content,
                generation_config=genai.GenerationConfig(
                    max_output_tokens=request.max_tokens,
                ),
            )
            
            # Extract usage metadata
            usage = {}
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                usage = {
                    "prompt_tokens": response.usage_metadata.prompt_token_count,
                    "completion_tokens": response.usage_metadata.candidates_token_count,
                    "total_tokens": response.usage_metadata.total_token_count,
                }
            
            return VisionResponse(
                text=response.text,
                model=self.model,
                usage=usage,
                raw_response=None,  # Google response isn't easily serializable
            )
            
        except Exception as e:
            logger.error(f"Google vision request failed: {e}")
            raise
