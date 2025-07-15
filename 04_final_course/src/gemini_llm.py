from typing import List, Dict, Any, Optional
import google.generativeai as genai
import time


class GeminiLLM:
    def __init__(
        self,
        api_key: str,
        model: str,
    ):
        genai.configure(api_key=api_key)
        self.model_name = model
        self.model = genai.GenerativeModel(model)

    def _convert_messages_to_gemini_format(
        self, messages: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """Convert standard message format to Gemini-specific format.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys

        Returns:
            List of Gemini-formatted messages
        """
        gemini_messages = []

        # Gemini uses "user" and "model" for roles
        role_mapping = {
            "user": "user",
            "assistant": "model",
            "system": "user",  # Gemini doesn't have a system role, handle specifically
        }

        # Check if first message is a system message
        if messages and messages[0].get("role") == "system":
            # For system message, we'll add it as a user message with a prefix
            system_content = messages[0].get("content", "")
            if system_content:
                # Add the rest of the messages
                system_prompt = f"{system_content}"
                gemini_messages.append({"role": "user", "parts": system_prompt})
                for message in messages[1:]:
                    role = role_mapping.get(message.get("role", "user"), "user")
                    content = message.get("content", "")
                    gemini_messages.append({"role": role, "parts": content})
        else:
            # No system message, just convert roles
            for message in messages:
                role = role_mapping.get(message.get("role", "user"), "user")
                content = message.get("content", "")
                gemini_messages.append({"role": role, "parts": content})

        return gemini_messages

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_new_tokens: Optional[int] = None,
    ):
        """Generate a response using Gemini API with rate limiting.

        Args:
            messages: List of message dictionaries
            temperature: Sampling temperature
            max_output_tokens: Maximum tokens to generate

        Returns:
            Generated response text
        """
        # Format messages for Gemini API
        gemini_messages = self._convert_messages_to_gemini_format(messages)

        # Check if we need to start a chat or just generate
        if len(gemini_messages) > 1:
            # Start a chat with history
            history = gemini_messages[:-1]  # All but the last message
            last_message = gemini_messages[-1]  # The last message to send

            chat = self.model.start_chat(
                history=history,
            )
            # Send the last message to get a response
            response = chat.send_message(
                last_message.get("parts", ""),
                stream=True,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_new_tokens if max_new_tokens else None,
                },
            )
            for chunk in response:
                if chunk.parts:
                    time.sleep(0.05)
                    yield chunk.text
        else:
            # Single message, use generate_content
            content = gemini_messages[0].get("parts", "") if gemini_messages else ""

            response = self.model.generate_content(
                content,
                stream=True,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_new_tokens if max_new_tokens else None,
                },
            )
            for chunk in response:
                if chunk.parts:
                    time.sleep(0.05)
                    yield chunk.text
