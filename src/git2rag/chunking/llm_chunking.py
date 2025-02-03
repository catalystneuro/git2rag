"""
LLM-assisted chunking using instructor and litellm.

This module provides a function llm_chunking that uses an LLM to extract segmentation boundaries
from the provided content, aimed at splitting the text into chunks that are efficient for semantic embeddings.
"""

import instructor
from litellm import completion
from pydantic import BaseModel
from typing import List

class ExtractionResult(BaseModel):
    boundaries: List[int]

def llm_chunking(
    content: str,
    file_type: str,
    model: str = "openai/gpt-4o-mini",
    strip_source_code: bool = True
) -> List[int]:
    """
    Uses an LLM to extract optimal chunk segmentation boundaries from the content based on the file type.
    The goal is to break the original text into semantically coherent segments that are efficient for embeddings.

    Additional Arguments:
        model (str): Specifies which LLM model to use.
        strip_source_code (bool): For code content, if True, instructs the LLM to include only function/class names,
                                    argument types and docstrings in the chunks.

    Args:
        content (str): The text content to analyze.
        file_type (str): The type of content; expected values are "code" or "documentation".

    Returns:
        List[int]: A list of integer indices indicating where each new chunk should start.
    """
    if file_type == "code":
        if strip_source_code:
            prompt = (
                "Given the following source code, determine optimal split points to divide the text into "
                "semantically coherent chunks suitable for efficient semantic embeddings. "
                "Only consider the function and class names, argument types, and docstrings; exclude implementation details. "
                "Each chunk should capture a complete logical unit and be of balanced length. "
                "Return a list of integer indices representing the start positions of each chunk."
            )
        else:
            prompt = (
                "Given the following source code, determine optimal split points to divide the text into "
                "semantically coherent chunks suitable for efficient semantic embeddings. Each chunk should "
                "capture a complete logical unit (for example, a function, class, or coherent block of code) "
                "and be of balanced length. Return a list of integer indices representing the start positions of each chunk."
            )
    else:
        prompt = (
            "Given the following documentation text, determine optimal split points that divide the content into "
            "semantically coherent chunks for efficient semantic embeddings. Each chunk should capture a complete "
            "idea or section and be of balanced length suitable for embedding. Return a list of integer indices "
            "indicating the start positions of each segment."
        )

    messages = [
        {
            "role": "user",
            "content": f"{prompt}\nContent:\n{content}",
        }
    ]
    client = instructor.from_litellm(completion)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        response_model=ExtractionResult,
    )
    return resp.boundaries
