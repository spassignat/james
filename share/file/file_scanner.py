# src/file_scanner.py
import logging
import os
import re
from pathlib import Path
from typing import List, Dict, Generator

from file.file_info import FileInfo

logger = logging.getLogger(__name__)


class FileScanner:
    def __init__(self, config: Dict):
        self.config = config

    def scan_project(self) -> Generator[FileInfo, None, None]:
        """Scanne récursivement le projet et yield les fichiers valides"""
        root_dir = self.config.get('root_directory', '.')
        excluded_dirs = self.config.get('excluded_directories', [])
        excluded_files = self.config.get('excluded_files', [])
        included_extensions = self.config.get('included_extensions', [])
        max_file_size = self.config.get('max_file_size', 5242880)

        logger.info(f"Début du scan du projet: {root_dir}")
        logger.info(f"Extensions incluses: {included_extensions}")

        for root, dirs, files in os.walk(root_dir):
            # Filtrer les répertoires à exclure
            dirs[:] = [d for d in dirs if not self._should_exclude_directory(d, excluded_dirs)]

            for filename in files:
                file_path = os.path.join(root, filename)

                if self._is_file_valid(file_path, included_extensions, excluded_files, max_file_size):
                    yield FileInfo(
                        path=file_path,
                        relative_path=os.path.relpath(file_path, root_dir),
                        filename=filename,
                        extension=Path(file_path).suffix.lower(),
                        directory=root,
                        size=os.path.getsize(file_path)
                    )

    def _should_exclude_directory(self, directory: str, excluded_patterns: List[str]) -> bool:
        """Vérifie si un répertoire doit être exclu"""
        for pattern in excluded_patterns:
            if re.search(pattern, directory):
                logger.debug(f"Répertoire exclu: {directory} (pattern: {pattern})")
                return True
        return False

    def _is_file_valid(self, file_path: str, included_extensions: List[str],
                       excluded_files: List[str], max_file_size: int) -> bool:
        """Vérifie si un fichier doit être inclus"""
        try:
            # Vérifier la taille
            file_size = os.path.getsize(file_path)
            if file_size > max_file_size:
                return False
            if file_size == 0:
                return False

            # Vérifier l'extension
            file_extension = Path(file_path).suffix.lower()
            if file_extension not in included_extensions:
                return False

            # Vérifier les patterns d'exclusion
            filename = os.path.basename(file_path)
            for pattern in excluded_files:
                if re.search(pattern, filename):
                    return False

            return True

        except OSError:
            return False

    def get_file_count(self) -> int:
        """Compte le nombre total de fichiers valides"""
        return sum(1 for _ in self.scan_project())
