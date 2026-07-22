"""
llm.py — Station 6a: a provider-agnostic connection to the AI.

The rest of the project calls llm.complete(system, user) and doesn't
care WHICH AI answers. Right now we use Gemini (free tier). To switch
providers later, you'd add a branch here and change config.yaml — no
other file needs to change. That's clean, swappable design.
"""

from google import genai
from google.genai import types
from src.config import config


class LLMClient:
    def __init__(self):
        self.provider = config.llm_provider

        if self.provider == "gemini":
            # Create a Gemini client with our secret key (loaded from .env).
            self._client = genai.Client(api_key=config.gemini_api_key)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def complete(self, system_prompt, user_prompt):
        """
        Send instructions (system) + the actual request (user) to the AI
        and return its text answer.
        """
        if self.provider == "gemini":
            # Gemini takes one combined prompt, so we join system + user.
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            response = self._client.models.generate_content(
                model=config.llm_model,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    temperature=config.temperature,
                    max_output_tokens=config.max_tokens,
                ),
            )
            return response.text

        raise ValueError(f"Unknown provider: {self.provider}")


# One shared LLM client the whole project uses.
llm = LLMClient()