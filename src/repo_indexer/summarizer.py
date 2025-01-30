"""Text summarization optimized for semantic embeddings."""

import os
from typing import Optional
from litellm import completion

# Default prompt optimized for semantic embeddings
DEFAULT_SUMMARIZER_PROMPT = """Analyze and summarize the following text with a focus on semantic meaning and key concepts. Your summary should:

1. Preserve the core concepts, technical terms, and domain-specific vocabulary
2. Maintain important relationships between concepts
3. Include key entities and their attributes
4. Retain critical context that affects meaning
5. Exclude redundant examples or repetitive phrasings
6. Focus on factual content over stylistic elements

Aim for a concise summary that would enable accurate semantic embeddings while preserving the essential meaning and relationships in the text.

Text to summarize:"""


def summarize_content(
    text_content: str,
    model: str = "openai/gpt-4o",
    api_key: Optional[str] = None,
    custom_prompt: Optional[str] = None,
    max_tokens: int = 500,
    temperature: float = 0.3,
) -> str:
    """Summarize text content optimized for semantic embeddings.

    Args:
        text_content: The text to summarize
        model: The model identifier (e.g., 'openai/gpt-4o', 'anthropic/claude-2')
        api_key: Optional API key. If not provided, uses environment variables
        custom_prompt: Optional custom prompt to override the default
        max_tokens: Maximum tokens in the summary
        temperature: Temperature for generation (lower = more focused)

    Returns:
        Summarized text optimized for semantic embeddings

    Raises:
        Exception: If the API call fails
    """
    # Set API key if provided
    if api_key:
        provider = model.split("/")[0] if "/" in model else "openai"
        os.environ[f"{provider.upper()}_API_KEY"] = api_key

    # Prepare prompt
    prompt = custom_prompt if custom_prompt else DEFAULT_SUMMARIZER_PROMPT
    messages = [{
        "role": "user",
        "content": f"{prompt}\n\n{text_content}"
    }]

    try:
        # Make API call
        response = completion(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )

        # Extract and return summary
        return response.choices[0].message.content.strip()

    except Exception as e:
        raise Exception(f"Summarization failed: {str(e)}")
