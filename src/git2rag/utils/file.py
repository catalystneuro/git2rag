from typing import Literal


def get_file_type(file_path: str) -> Literal["code", "docs"]:
    """
    Get the file type of a file.
    """
    ext = file_path.split('.')[-1]
    if ext in {".py", ".js", ".java", ".cpp", ".h", ".hpp"}:
        return "code"
    elif ext in {".md", ".rst", ".txt", ".ipynb"}:
        return "docs"
    else:
        return "docs"