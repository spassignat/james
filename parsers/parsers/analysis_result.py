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

    # Résultats principaux
    chunks: List[CodeChunk]

    # Informations secondaires
    imports: List[str]
    symbols: List[str]

    # Diagnostics
    errors: List[str]
