# src/agents/agent_manager.py (ajouts et modifications)
import os
import re
from pathlib import Path
from typing import Dict, Any, List

# Ajouter l'import
import logging

from config.file_scanner import FileScanner

logger = logging.getLogger(__name__)

class ProjectAnalyzer:
    """Analyseur de projet qui utilise FileScanner"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.project_config = config.get('project', {})

    def analyze_project_structure(self) -> Dict[str, Any]:
        """Analyse la structure du projet"""
        logger.info("📁 Analyse de la structure du projet...")

        # Initialiser le scanner
        scanner = FileScanner(self.project_config)

        # Collecter les fichiers
        files_by_extension = {}
        total_files = 0
        total_size = 0

        for file_info in scanner.scan_project():
            total_files += 1
            total_size += file_info.get('size', 0)

            extension = file_info['extension']
            if extension not in files_by_extension:
                files_by_extension[extension] = []
            files_by_extension[extension].append(file_info)

        # Analyser la structure des répertoires
        directory_structure = self._analyze_directory_structure(scanner)

        # Identifier les patterns de fichiers
        file_patterns = self._identify_file_patterns(files_by_extension)

        return {
            'total_files': total_files,
            'total_size_bytes': total_size,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'files_by_extension': files_by_extension,
            'directory_structure': directory_structure,
            'file_patterns': file_patterns,
            'extensions_found': list(files_by_extension.keys()),
            'file_count_by_extension': {ext: len(files) for ext, files in files_by_extension.items()}
        }

    def _analyze_directory_structure(self, scanner: FileScanner) -> Dict[str, Any]:
        """Analyse la structure des répertoires"""
        root_dir = self.project_config.get('root_directory', '.')

        # Utiliser os.walk pour analyser la structure
        dir_structure = {}

        for root, dirs, files in os.walk(root_dir):
            relative_root = os.path.relpath(root, root_dir)
            if relative_root == '.':
                relative_root = '/'

            # Compter les fichiers par type
            file_types = {}
            for file in files:
                ext = Path(file).suffix.lower()
                file_types[ext] = file_types.get(ext, 0) + 1

            dir_structure[relative_root] = {
                'subdirectories': dirs,
                'files_count': len(files),
                'file_types': file_types,
                'has_source_code': any(ext in ['.java', '.js', '.py', '.ts'] for ext in file_types.keys())
            }

        return dir_structure

    def _identify_file_patterns(self, files_by_extension: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Identifie les patterns de fichiers"""
        patterns = {
            'controllers': [],
            'services': [],
            'repositories': [],
            'models': [],
            'tests': [],
            'configs': [],
            'utils': []
        }

        # Patterns pour chaque type
        pattern_mappings = {
            'controllers': [r'.*controller\.', r'.*ctrl\.', r'.*handler\.'],
            'services': [r'.*service\.', r'.*manager\.', r'.*processor\.'],
            'repositories': [r'.*repository\.', r'.*dao\.', r'.*dataaccess\.'],
            'models': [r'.*model\.', r'.*entity\.', r'.*dto\.', r'.*vo\.'],
            'tests': [r'.*test\.', r'.*spec\.', r'.*\.spec\.'],
            'configs': [r'.*config\.', r'.*settings\.', r'.*\.conf'],
            'utils': [r'.*util\.', r'.*helper\.', r'.*tool\.']
        }

        for extension, files in files_by_extension.items():
            for file_info in files:
                filename = file_info['filename'].lower()

                for pattern_type, patterns_list in pattern_mappings.items():
                    for pattern in patterns_list:
                        if re.search(pattern, filename):
                            patterns[pattern_type].append(file_info['relative_path'])
                            break

        # Filtrer les listes vides
        return {k: v for k, v in patterns.items() if v}


