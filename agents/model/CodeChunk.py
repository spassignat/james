from dataclasses import dataclass


@dataclass
class CodeChunk:
    content: str
    file_path: str
    filename: str
    language: str
    category: str
    chunk_type: str
