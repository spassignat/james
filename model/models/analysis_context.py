from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from models.code_chunk import CodeChunk
from models.project_structure import ProjectStructure


@dataclass
class AnalysisContext:
    """
    Contexte complet pour l'analyse et la génération.
    Contient la structure du projet, les chunks, patterns, et config.
    """
    project_structure: ProjectStructure
    chunks: List[CodeChunk]
    file_patterns: Dict[str, Any]
    project_config: Dict[str, Any]
    project_name: str

    def __init__(
            self,
            project_structure: Optional[ProjectStructure] = None,
            chunks: Optional[List[CodeChunk]] = None,
            file_patterns: Optional[Dict[str, Any]] = None,
            project_config: Optional[Dict[str, Any]] = None,
            project_name: str = "A Project"
    ):
        self.project_structure = project_structure
        self.chunks = chunks or []
        self.file_patterns = file_patterns or {}
        self.project_config = project_config or {}
        self.project_name = project_name
