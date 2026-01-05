from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class CodeChunk:
    """
    Représente un fragment de code ou de contenu analysé.
    """

    # Métadonnées générales
    language: str
    name: str

    # Contenu
    content: str

    # Localisation
    file_path: str | Path
    start_line: int
    end_line: int
