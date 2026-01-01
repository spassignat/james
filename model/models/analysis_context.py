from dataclasses import dataclass
from typing import List, Dict

from .code_chunk import CodeChunk
from .project_structure import ProjectStructure


@dataclass
class AnalysisContext:
    def __init__(self, project_structure: ProjectStructure, chunks: List[CodeChunk], config: dict):
        self.project_structure = project_structure
        self.chunks = chunks
        self.config = config

    project_structure: ProjectStructure
    chunks: List[CodeChunk]
    config: Dict
