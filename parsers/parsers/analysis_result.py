from dataclasses import dataclass
from pathlib import Path
from typing import List

from parsers.code_chunk import CodeChunk


@dataclass(slots=True)
class AnalysisResult:
    """
    Résultat structuré de l'analyse d'un fichier.
    """

    # Identification
    language: str
    file_path: Path
    file_part: List[str]
    file_name: str
    last_modified: float
    name_parts: List[str]
    # Résultats principaux
    chunks: List[CodeChunk]
    # Informations secondaires
    imports: List[str]
    symbols: List[str]

    # Diagnostics
    errors: List[str]

    def __init__(self, language: str):
        self.imports = []
        self.symbols = []
        self.errors = []
        self.chunks = []
        self.language = language
