import instructor
from litellm import completion
from pydantic import BaseModel
from typing import List


PROMPT_CODE = """Analyze the following Python source code and segment it into semantically coherent chunks that represent distinct logical sections.
You should include in the chunks:
- Only original code segments, do not write anything that is not present in the code.
- Individual function definitions (capturing the function signature and docstring)
- Class definitions (including the class name, constructor, methods with their signatures and docstrings)
- Full docstrings for functions and classes, do not split or truncate them.
You should exclude:
- Module header (e.g., shebang and module-level docstring)
- Imports and configuration blocks, e.g. loggers and constants
- Main execution blocks

For each chunk, include only high-level details and avoid deep implementation details if the code within the chunk is extensive.
Return a JSON object with a key 'chunks' that maps to a list of strings, where each string is one of the extracted segments.
---
Example:
Content:
```python
import os
import sys
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_sum(a: int, b: int) -> int:
    '''
    Calculate the sum of two integers.

    Args:
        a (int): The first integer.
        b (int): The second integer.

    Returns:
        int: The sum of a and b.
    '''
    result = a + b
    logger.debug("Calculated sum: %d + %d = %d", a, b, result)
    return result


class Person:
    '''
    A class representing a person.

    Attributes:
        name (str): The person's name.
        age (int): The person's age.
    '''

    def __init__(self, name: str, age: int) -> None:
        '''
        Initialize a new Person instance.

        Args:
            name (str): The person's name.
            age (int): The person's age.
        '''
        self.name = name
        self.age = age
        logger.info("Created Person: %s, Age: %d", self.name, self.age)

    def greet(self, other: Optional[str] = None) -> str:
        '''
        Generate a greeting message.

        Args:
            other (Optional[str]): Optional name of another person to greet.

        Returns:
            str: A greeting message.
        '''
        if other:
            message = f"Hello {other}, I am {self.name}."
        else:
            message = f"Hello, my name is {self.name}."
        logger.info("Greeting message: %s", message)
        return message

    def have_birthday(self) -> None:
        '''
        Increment the person's age by one year.
        '''
        self.age += 1
        logger.info("%s is now %d years old.", self.name, self.age)
```
Your response should be:
```json
{
    "chunks": [
        "calculate_sum(a: int, b: int) -> int: '''Calculate the sum of two integers. Args: a (int): The first integer. b (int): The second integer. Returns: int: The sum of a and b.'''",
        "Person: '''A class representing a person. Attributes: name (str): The person's name. age (int): The person's age.''' def __init__(self, name: str, age: int) -> None: '''Initialize a new Person instance. Args: name (str): The person's name. age (int): The person's age.'''
def greet(self, other: Optional[str] = None) -> str: ''' Generate a greeting message. Args: other (Optional[str]): Optional name of another person to greet. Returns: str: A greeting message.'''
def have_birthday(self) -> None: '''Increment the person's age by one year.'''",

    ]
}
```
"""

PROMPT_DOC = """Given the following documentation text, determine the optimal segmentation to break the content into semantically coherent chunks
suitable for semantic embeddings. Each chunk should capture a complete idea or section.
Return a JSON object with a key 'chunks' that maps to a list of strings, each representing a chunk from the input content.
"""


class ExtractionResult(BaseModel):
    chunks: List[str]


def llm_chunking(
    content: str,
    file_type: str,
    model: str = "openai/o3-mini",
    strip_source_code: bool = True,
) -> List[str]:
    """
    Uses an LLM to extract semantically optimal chunked segments from the content based on the file type.
    The goal is to break the original text into semantically coherent chunks that are efficient for semantic embeddings.

    Additional Arguments:
        model (str): Specifies which LLM model to use.
        strip_source_code (bool): For code content, if True, instructs the LLM to include only high-level details
                                    (function/class names, argument types, docstrings, etc.) and exclude internal implementation details.

    Args:
        content (str): The text content to analyze.
        file_type (str): The type of content; expected values are "code" or "documentation".

    Returns:
        List[str]: A list of semantically coherent chunks extracted from the input content.
    """
    if file_type == "code":
        prompt = PROMPT_CODE
    else:
        prompt = PROMPT_DOC

    messages = [
        {
            "role": "user",
            "content": f"{PROMPT}\nContent:\n{content}",
        }
    ]
    client = instructor.from_litellm(completion)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        response_model=ExtractionResult,
    )
    return resp.chunks
