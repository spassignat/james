from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ProjectStructure:
    root: str
    folders: Dict[str, List[str]]
    files: List[str]
    patterns: Dict[str, List[str]]
    def __init__(self, **kwargs):
        # kwargs contient les infos renvoyées par ProjectAnalyzer
        for k, v in kwargs.items():
            setattr(self, k, v)