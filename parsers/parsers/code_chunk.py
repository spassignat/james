from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Any

import numpy as np


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
    name_parts: List[str] = field(default_factory=list)
