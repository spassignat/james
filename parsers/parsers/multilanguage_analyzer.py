# multi_language_analyzer.py
import logging
import os
import traceback
from datetime import datetime
from typing import Dict, Any, Optional

from parsers.analysis_result import AnalysisResult, FileType
from parsers.analyzer import Analyzer
from parsers.generic_analyzer import GenericAnalyzer
from parsers.lang.css_analyzer import CSSAnalyzer
from parsers.lang.html_analyzer import HTMLAnalyzer
from parsers.lang.java_analyzer import JavaAnalyzer
from parsers.lang.javascript_analyzer import JavaScriptAnalyzer
from parsers.lang.json_analyzer import JSONAnalyzer
from parsers.lang.properties_analyzer import PropertiesAnalyzer
from parsers.lang.python_analyzer import PythonAnalyzer
from parsers.lang.sql_analyzer import SQLAnalyzer
from parsers.lang.vuejs_analyzer import VueJSAnalyzer
from parsers.lang.xml_analyzer import XMLAnalyzer

logger = logging.getLogger(__name__)


class MultiLanguageAnalyzer(Analyzer):
    """Analyseur multi-langages qui route vers l'analyseur approprié"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.config = config or {}
        self.file_type = FileType.UNKNOWN

        # Initialiser tous les analyseurs
        self.analyzers = {
            '.java': JavaAnalyzer(),
            '.js': JavaScriptAnalyzer(),
            '.ts': JavaScriptAnalyzer(),  # TypeScript utilise le même analyseur
            '.vue': VueJSAnalyzer(),
            '.sql': SQLAnalyzer(),
            '.json': JSONAnalyzer(),
            '.yaml': JSONAnalyzer(),  # YAML utilise JSONAnalyzer (simplifié)
            '.yml': JSONAnalyzer(),  # YML utilise JSONAnalyzer (simplifié)
            '.properties': PropertiesAnalyzer(),
            '.ini': PropertiesAnalyzer(),  # INI utilise PropertiesAnalyzer
            '.cfg': PropertiesAnalyzer(),  # CFG utilise PropertiesAnalyzer
            '.xml': XMLAnalyzer(),
            '.html': HTMLAnalyzer(),
            '.css': CSSAnalyzer(),
            '.py': PythonAnalyzer(),
            '.md': GenericAnalyzer(),  # Markdown
            '.txt': GenericAnalyzer(),  # Texte brut
            '.sh': GenericAnalyzer(),  # Shell script
            '.dockerfile': GenericAnalyzer(),  # Dockerfile
            '.gitignore': GenericAnalyzer(),  # Gitignore
        }

        # Mapper les extensions aux FileType
        self.extension_to_filetype = {
            '.java': FileType.JAVA,
            '.js': FileType.JAVASCRIPT,
            '.ts': FileType.TYPESCRIPT,
            '.vue': FileType.VUE,
            '.sql': FileType.SQL,
            '.json': FileType.JSON,
            '.yaml': FileType.YAML,
            '.yml': FileType.YAML,
            '.properties': FileType.YAML,  # Approximatif
            '.ini': FileType.YAML,  # Approximatif
            '.cfg': FileType.YAML,  # Approximatif
            '.xml': FileType.XML,
            '.html': FileType.HTML,
            '.css': FileType.CSS,
            '.py': FileType.PYTHON,
            '.md': FileType.MARKDOWN,
            '.txt': FileType.UNKNOWN,
            '.sh': FileType.SHELL,
            '.dockerfile': FileType.DOCKERFILE,
            '.gitignore': FileType.UNKNOWN,
        }

    def analyze_file(self, file_path: str) -> AnalysisResult:
        """Analyse un fichier avec l'analyseur approprié et retourne un AnalysisResult"""
        try:
            # Mesurer le temps de traitement total
            start_time = datetime.now()

            # Déterminer l'extension et l'analyseur
            ext = self._get_file_extension(file_path)
            analyzer = self._get_analyzer_for_extension(ext)

            # Analyser le fichier avec l'analyseur approprié
            result = analyzer.analyze_file(file_path)

            # Ajouter des métadonnées spécifiques au multi-langage
            self._enhance_multi_language_result(result, file_path, ext)

            # Calculer le temps de traitement total
            total_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            result.processing_time_ms = total_time_ms
            result.analyzer_name = "MultiLanguageAnalyzer"

            return result

        except Exception as e:
            print(traceback.format_exc())
            logger.error(f"Error in MultiLanguageAnalyzer analyzing {file_path}: {e}")
            return self._create_error_result(file_path, f"Multi-language analysis error: {str(e)}")

    def analyze_content(self, content: str, file_path: str) -> Dict[str, Any]:
        """Analyse du contenu (méthode interne pour compatibilité)"""
        # Cette méthode est principalement pour la compatibilité avec l'interface parente
        # En pratique, analyze_file() est la méthode principale
        try:
            # Simuler une analyse basique
            ext = self._get_file_extension(file_path)
            file_type = self.extension_to_filetype.get(ext, FileType.UNKNOWN)

            return {
                'file_type': file_type.value if file_type else 'unknown',
                'file_path': file_path,
                'extension': ext,
                'content_preview': content[:500],
                'line_count': len(content.split('\n'))
            }
        except Exception as e:
            logger.error(f"Error analyzing content for {file_path}: {e}")
            return {
                'file_type': 'unknown',
                'file_path': file_path,
                'error': str(e)
            }

    def _get_file_extension(self, file_path: str) -> str:
        """Obtient l'extension du fichier avec gestion des cas spéciaux"""
        filename = os.path.basename(file_path).lower()

        # Cas spéciaux pour les fichiers sans extension standard
        if filename == 'dockerfile' or filename.endswith('.dockerfile'):
            return '.dockerfile'
        elif filename == 'makefile':
            return '.sh'  # Traiter comme shell
        elif filename == '.gitignore':
            return '.gitignore'

        # Extension normale
        ext = os.path.splitext(filename)[1].lower()

        # Gestion des extensions multiples (comme .test.js, .spec.ts, etc.)
        if ext in ['.js', '.ts', '.jsx', '.tsx', '.vue']:
            # Vérifier s'il y a une deuxième extension (comme .test.js)
            name_without_ext = os.path.splitext(filename)[0]
            second_ext = os.path.splitext(name_without_ext)[1].lower()
            if second_ext in ['.test', '.spec', '.e2e', '.unit']:
                return ext  # Garder l'extension principale

        return ext

    def _get_analyzer_for_extension(self, ext: str) -> Analyzer:
        """Obtient l'analyseur approprié pour une extension"""
        if ext in self.analyzers:
            return self.analyzers[ext]
        else:
            logger.debug(f"No specific analyzer for extension {ext}, using GenericAnalyzer")
            return GenericAnalyzer()

    def _enhance_multi_language_result(self, result: AnalysisResult, file_path: str, ext: str) -> None:
        """Améliore le résultat avec des informations multi-langages"""

        # Mettre à jour le type de fichier basé sur l'extension
        file_type = self.extension_to_filetype.get(ext, FileType.UNKNOWN)
        if result.file_type == FileType.UNKNOWN and file_type != FileType.UNKNOWN:
            result.file_type = file_type

        # Ajouter des métadonnées sur l'analyseur utilisé
        if not hasattr(result, 'analyzer_used'):
            result.analyzer_used = type(self._get_analyzer_for_extension(ext)).__name__

        # Détecter les fichiers de test
        self._detect_test_file(result, file_path)

        # Détecter les fichiers de configuration
        self._detect_config_file(result, file_path)

        # Ajouter des diagnostics multi-langages
        self._add_multi_language_diagnostics(result, file_path)

    def _detect_test_file(self, result: AnalysisResult, file_path: str) -> None:
        """Détecte si le fichier est un fichier de test"""
        filename = os.path.basename(file_path).lower()
        path_lower = file_path.lower()

        test_patterns = [
            'test', 'spec', 'mock', 'stub', 'fake',
            '__test__', '__tests__', 'test_', '_test.',
            '.test.', '.spec.', '.e2e.', '.unit.'
        ]

        is_test = any(pattern in filename for pattern in test_patterns) or \
                  f'/test/' in path_lower or '/tests/' in path_lower or \
                  f'\\test\\' in path_lower or '\\tests\\' in path_lower

        if is_test:
            result.language_specific['is_test_file'] = True
            if 'test' not in result.patterns.architecture_hints:
                result.patterns.architecture_hints.append('test_file')

            # Mettre à jour l'analyse de nommage si elle existe
            if result.naming_analysis:
                result.naming_analysis.is_test_file = True

    def _detect_config_file(self, result: AnalysisResult, file_path: str) -> None:
        """Détecte si le fichier est un fichier de configuration"""
        filename = os.path.basename(file_path).lower()

        config_patterns = [
            'config', 'configuration', 'settings', 'properties',
            '.env', '.config.', 'application.', 'bootstrap.',
            'web.xml', 'pom.xml', 'build.gradle', 'package.json',
            'docker-compose', 'dockerfile', '.gitignore',
            'makefile', 'cmakelists.txt'
        ]

        is_config = any(pattern in filename for pattern in config_patterns)

        if is_config:
            result.language_specific['is_config_file'] = True
            if 'configuration' not in result.patterns.architecture_hints:
                result.patterns.architecture_hints.append('configuration_file')

            # Mettre à jour l'analyse de nommage si elle existe
            if result.naming_analysis:
                result.naming_analysis.is_config_file = True

    def _add_multi_language_diagnostics(self, result: AnalysisResult, file_path: str) -> None:
        """Ajoute des diagnostics spécifiques au multi-langage"""

        # Vérifier la taille du fichier
        if result.file_size > 1024 * 1024:  # > 1MB
            result.warnings.append(f"Large file size ({result.file_size / 1024 / 1024:.1f} MB)")

        # Vérifier le nombre de lignes
        if result.metrics.total_lines > 1000:
            result.notes.append(f"Long file with {result.metrics.total_lines} lines")

        # Vérifier la complexité
        if result.metrics.complexity_score > 10:
            result.notes.append(f"High complexity score: {result.metrics.complexity_score:.1f}")

        # Vérifier les dépendances
        if len(result.dependencies.external_deps) > 20:
            result.notes.append(f"Many external dependencies: {len(result.dependencies.external_deps)}")

        # Recommandations basées sur le type de fichier
        if result.file_type == FileType.HTML:
            if result.metrics.complexity_score > 5:
                result.recommendations.append("Consider splitting complex HTML into components")

        elif result.file_type == FileType.JAVASCRIPT:
            if result.metrics.function_count > 20:
                result.recommendations.append("Consider breaking down large JavaScript file into modules")

        elif result.file_type == FileType.PYTHON:
            if result.metrics.class_count == 0 and result.metrics.function_count > 10:
                result.recommendations.append("Consider organizing functions into classes or modules")

    def get_supported_extensions(self) -> Dict[str, str]:
        """Retourne la liste des extensions supportées avec leur description"""
        return {
            '.java': 'Java source files',
            '.js': 'JavaScript files',
            '.ts': 'TypeScript files',
            '.vue': 'Vue.js single file components',
            '.sql': 'SQL scripts and schema files',
            '.json': 'JSON data files',
            '.yaml': 'YAML configuration files',
            '.yml': 'YAML configuration files',
            '.properties': 'Java properties files',
            '.ini': 'INI configuration files',
            '.cfg': 'Configuration files',
            '.xml': 'XML files',
            '.html': 'HTML files',
            '.css': 'CSS stylesheets',
            '.py': 'Python scripts',
            '.md': 'Markdown documentation',
            '.txt': 'Plain text files',
            '.sh': 'Shell scripts',
            '.dockerfile': 'Docker build files',
            '.gitignore': 'Git ignore files'
        }

    def get_analyzer_stats(self) -> Dict[str, Any]:
        """Retourne des statistiques sur les analyseurs disponibles"""
        stats = {
            'total_analyzers': len(self.analyzers),
            'supported_extensions': list(self.analyzers.keys()),
            'analyzer_types': {}
        }

        for ext, analyzer in self.analyzers.items():
            analyzer_type = type(analyzer).__name__
            if analyzer_type not in stats['analyzer_types']:
                stats['analyzer_types'][analyzer_type] = []
            stats['analyzer_types'][analyzer_type].append(ext)

        return stats

    def analyze_file_with_fallback(self, file_path: str, primary_ext: str = None) -> AnalysisResult:
        """Analyse un fichier avec un type spécifique, avec fallback automatique"""
        try:
            # Si une extension primaire est spécifiée, l'utiliser
            if primary_ext:
                analyzer = self._get_analyzer_for_extension(primary_ext)
                result = analyzer.analyze_file(file_path)
                self._enhance_multi_language_result(result, file_path, primary_ext)
                return result

            # Sinon, utiliser la détection automatique
            return self.analyze_file(file_path)

        except Exception as e:
            logger.warning(f"Primary analyzer failed for {file_path}: {e}, trying fallback")

            # Fallback vers GenericAnalyzer
            try:
                generic_analyzer = GenericAnalyzer()
                result = generic_analyzer.analyze_file(file_path)
                self._enhance_multi_language_result(result, file_path, os.path.splitext(file_path)[1])
                return result
            except Exception as fallback_error:
                logger.error(f"Fallback analyzer also failed for {file_path}: {fallback_error}")
                return self._create_error_result(file_path, f"All analyzers failed: {str(e)}")

    def batch_analyze(self, file_paths: list[str]) -> Dict[str, AnalysisResult]:
        """Analyse plusieurs fichiers en lot"""
        results = {}

        for file_path in file_paths:
            try:
                result = self.analyze_file(file_path)
                results[file_path] = result

                # Log du statut
                if result.has_errors():
                    logger.warning(f"File {file_path} has errors: {result.errors}")
                else:
                    logger.debug(f"Successfully analyzed {file_path}")

            except Exception as e:
                logger.error(f"Failed to analyze {file_path}: {e}")
                error_result = self._create_error_result(file_path, str(e))
                results[file_path] = error_result

        return results

    def get_analysis_summary(self, results: Dict[str, AnalysisResult]) -> Dict[str, Any]:
        """Génère un résumé des analyses"""
        summary = {
            'total_files': len(results),
            'successful_analyses': 0,
            'failed_analyses': 0,
            'file_types': {},
            'total_lines': 0,
            'total_functions': 0,
            'total_classes': 0,
            'languages_found': set(),
            'frameworks_found': set()
        }

        for file_path, result in results.items():
            if result.is_valid():
                summary['successful_analyses'] += 1
            else:
                summary['failed_analyses'] += 1

            # Compter par type de fichier
            file_type = result.file_type.value
            summary['file_types'][file_type] = summary['file_types'].get(file_type, 0) + 1

            # Métriques cumulées
            summary['total_lines'] += result.metrics.total_lines
            summary['total_functions'] += result.metrics.function_count
            summary['total_classes'] += result.metrics.class_count

            # Langages et frameworks
            summary['languages_found'].add(file_type)
            for framework in result.patterns.frameworks:
                summary['frameworks_found'].add(framework.value)

        # Convertir les sets en listes
        summary['languages_found'] = list(summary['languages_found'])
        summary['frameworks_found'] = list(summary['frameworks_found'])

        return summary
