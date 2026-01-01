from dataclasses import dataclass
from typing import Optional


@dataclass
class CodeChunk:
    content: str
    file_path: str
    filename: str
    language: str
    category: str
    chunk_type: str
    embedding: Optional[list[float]] = None
    def __init__(self, content: str, file_path: str, filename: str, language: str, category: str, chunk_type: str):
        self.content = content
        self.file_path = file_path
        self.filename = filename
        self.language = language
        self.category = category
        self.chunk_type = chunk_type