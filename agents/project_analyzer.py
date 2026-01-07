# src/project_analyzer.py
import logging
from pathlib import Path
from typing import List, Dict, Any, Generator

from file.file_scanner import FileScanner
from models.project_structure import ProjectStructure

logger = logging.getLogger(__name__)

class ProjectAnalyzer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.root_path = Path(config.get('project', {}).get('root_directory', '.'))
        self.scanner = FileScanner(config.get('project', {}))

    def analyze_project_structure(self) -> ProjectStructure:
        """Retourne la structure complète du projet"""
        files = list((file_info.path or '') for file_info in self.scanner.scan_project() )
        patterns_identified = self._identify_patterns(files)
        modules = self._identify_modules(files)

        return ProjectStructure(
            project_name=self.root_path.name,
            files=files,
            patterns_identified=patterns_identified,
            modules=modules
        )

    def _identify_patterns(self, files: List[str]) -> List[str]:
        patterns = set()
        for file_path in files:
            for pattern in self.config.get('project', {}).get('file_patterns', []):
                if pattern in file_path:
                    patterns.add(pattern)
        return list(patterns)

    def _identify_modules(self, files: List[str]) -> List[str]:
        modules = set()
        for file_path in files:
            path = Path(file_path)
            if path.parent != self.root_path:
                modules.add(str(path.parent.relative_to(self.root_path)))
        return sorted(modules)

    def scan_project_files(self) -> Generator[Dict[str, Any], None, None]:
        """Renvoie les fichiers un par un via FileScanner"""
        yield from self.scanner.scan_project()
