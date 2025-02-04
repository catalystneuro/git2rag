"""Text summarization optimized for semantic embeddings."""

import os
from typing import Optional, List, Union
from litellm import completion, batch_completion

# Default prompt optimized for semantic embeddings
DEFAULT_SUMMARIZER_PROMPT = """Analyze and summarize the following text with a focus on semantic meaning and key concepts. Your summary must:

1. Preserve the core concepts, technical terms, and domain-specific vocabulary
2. Maintain important relationships between concepts
3. Include key entities and their attributes
4. Retain critical context that affects meaning
5. Exclude redundant examples or repetitive phrasings
6. Focus on factual content over stylistic elements
7. Not exceed {max_tokens} tokens in length

Aim for a concise summary that would enable accurate semantic embeddings while preserving the essential meaning and relationships in the text.

Text to summarize:"""


def summarize_content(
    text_content: List[str],
    model: str = "openai/gpt-4o-mini",
    custom_prompt: Optional[str] = None,
    max_tokens: int = 100,
    batch_size: int = 20,
) -> List[str]:
    """Summarize text content optimized for semantic embeddings.

    Args:
        text_content: List of texts to summarize
        model: The model identifier (e.g., 'openai/gpt-4o', 'anthropic/claude-2')
        custom_prompt: Optional custom prompt to override the default
        max_tokens: Maximum tokens in the summary
        batch_size: Size of batches when processing multiple texts (default: 20)

    Returns:
        List of summarized texts

    Raises:
        Exception: If the API call fails
    """
    # Prepare prompt with max_tokens
    prompt = (
        custom_prompt if custom_prompt else DEFAULT_SUMMARIZER_PROMPT.format(max_tokens=max_tokens)
    )

    # Prepare messages for batch processing
    messages_list = [[{"role": "user", "content": f"{prompt}\n\n{text}"}] for text in text_content]

    try:
        # Process in batches
        summaries = []
        total_chunks = len(messages_list)
        for i in range(0, total_chunks, batch_size):
            batch = messages_list[i : i + batch_size]
            responses = batch_completion(
                model=model,
                messages=batch,
            )
            summaries.extend([r.choices[0].message.content.strip() for r in responses])
            print(f"Summarized chunks {i + len(batch)}/{total_chunks}")
        return summaries

    except Exception as e:
        raise Exception(f"Batch summarization failed: {str(e)}")
