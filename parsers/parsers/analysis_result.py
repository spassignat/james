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

    # Résultats principaux
    chunks: List[CodeChunk]

    # Informations secondaires
    imports: List[str]
    symbols: List[str]

    # Diagnostics
    errors: List[str]

    def __init__(self, language: str):
        super().__init__()
        imports = []
        symbols = []
        errors = []
        chunks = []
        self.language = language
