# parsers/analyzer.py
import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from file.file_name_parser import FilenameParser
from parsers.analysis_result import AnalysisResult, FileType, FileNamingAnalysis, AnalysisStatus, FileMetrics

logger = logging.getLogger(__name__)


class Analyzer(ABC):
    """Analyseur de code de base retournant des AnalysisResult"""

    def __init__(self):
        self.filename_parser = FilenameParser()
        self.file_type = FileType.UNKNOWN

    @abstractmethod
    def analyze_file(self, file_path: str) -> AnalysisResult:
        """Analyse un fichier et retourne un AnalysisResult"""
        pass

    @abstractmethod
    def analyze_content(self, content: str, file_path: str) -> Dict[str, Any]:
        """Analyse du contenu (méthode interne pour compatibilité)"""
        pass

    def _create_base_result(self, file_path: str) -> AnalysisResult:
        """Crée un AnalysisResult de base avec métadonnées"""
        path_obj = Path(file_path)
        getsize = os.path.getsize(file_path)
        analysis = self._get_naming_analysis(file_path)
        lm = path_obj.stat().st_mtime
        result = AnalysisResult(file_path=str(file_path),
                                filename=path_obj.name,
                                file_type=self.file_type,
                                file_size=getsize,
                                processing_time_ms=0,
                                last_modified=lm,
                                status=AnalysisStatus.PENDING,
                                naming_analysis=analysis)
        return result

    def _get_naming_analysis(self, file_path: str) -> Optional[FileNamingAnalysis]:
        """Analyse le nom de fichier"""
        try:
            filename = Path(file_path).name
            analysis = self.filename_parser.extract_semantic_info(filename)

            # Convertir FileType string en enum
            file_type_str = analysis['deduced']['file_type']
            file_type = FileType.UNKNOWN
            for ft in FileType:
                if ft.value == file_type_str:
                    file_type = ft
                    break

            return FileNamingAnalysis(
                original_name=filename,
                stem=analysis['parsed']['stem'],
                extension=analysis['parsed']['extension'],
                convention=analysis['parsed']['convention'],
                tokens=analysis['parsed']['tokens'],
                suggested_name=analysis['parsed'].get('suggested_name'),
                is_test_file=analysis['parsed']['metadata']['is_test_file'],
                is_config_file=analysis['parsed']['metadata']['is_config_file'],
                is_main_file=analysis['parsed']['metadata']['is_main_file'],
                file_type=file_type,
                domain=analysis['deduced']['domain'],
                layer=analysis['deduced']['layer'],
                purpose=analysis['deduced']['purpose']
            )
        except Exception as e:
            logger.warning(f"Erreur analyse nom fichier {file_path}: {e}")
            return None

    def _create_error_result(self, file_path: str, error: str) -> AnalysisResult:
        """Crée un résultat d'erreur"""
        result = self._create_base_result(file_path)
        result.status = AnalysisStatus.ERROR
        result.errors.append(error)
        return result

    def _calculate_metrics(self, content: str) -> FileMetrics:
        """Calcule les métriques de base"""
        lines = content.split('\n')

        return FileMetrics(
            total_lines=len(lines),
            code_lines=len([l for l in lines if l.strip() and not l.strip().startswith('//')]),
            comment_lines=len([l for l in lines if l.strip().startswith('//')]),
            blank_lines=len([l for l in lines if not l.strip()]),
            file_size_bytes=len(content.encode('utf-8')),
            average_line_length=sum(len(l) for l in lines) / len(lines) if lines else 0
        )
