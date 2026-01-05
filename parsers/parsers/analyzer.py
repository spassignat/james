# parsers/analyzer.py
import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

from file.file_name_parser import FilenameParser
from parsers.analysis_result import AnalysisResult

logger = logging.getLogger(__name__)


class Analyzer(ABC):
    """Analyseur de code de base retournant des AnalysisResult"""

    language: str

    def __init__(self, language: str):
        self.filename_parser = FilenameParser()
        self.language = language

    def analyze_file(self, file_path: str) -> AnalysisResult:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Mesurer le temps de traitement
            start_time = datetime.now()

            # Analyser le contenu
            result = self.analyze_content(content, file_path)

            # Mettre à jour les résultats
            self._update_result(result, Path(file_path))

            # Calculer le temps de traitement
            result.processing_time_ms = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )

            return result

        except Exception as e:
            logger.error(f"Error analyzing properties file {file_path}: {e}")
            return self._create_error_result(Path(file_path), str(e))

    @abstractmethod
    def analyze_content(self, content: str, file_path: str) -> AnalysisResult:
        """Analyse du contenu (méthode interne pour compatibilité)"""
        pass

    def _update_result(self, result: AnalysisResult, file_path: Path) -> AnalysisResult:
        """Crée un AnalysisResult de base avec métadonnées"""
        getsize = os.path.getsize(file_path)
        lm = file_path.stat().st_mtime
        result.file_path = file_path
        result.filename = file_path.name,
        result.file_size = getsize,
        result.processing_time_ms = 0,
        result.last_modified = lm,
        result.language = self.language
        return result

    def _create_error_result(self, file_path: Path, error: str) -> AnalysisResult:
        """Crée un résultat d'erreur"""
        result = AnalysisResult(self.language)
        self._update_result(result, file_path)
        result.errors.append(error)
        return result
