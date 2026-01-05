# generic_analyzer.py
import logging
import os
import re
import traceback
from datetime import datetime
from typing import Dict, Any, List

from parsers.analysis_result import (
    AnalysisResult, AnalysisStatus, FileType, FrameworkType,
    CodeElement, FileMetrics
)
from parsers.analyzer import Analyzer

logger = logging.getLogger(__name__)


class GenericAnalyzer(Analyzer):
    """Analyseur générique pour les fichiers sans analyseur spécifique"""

    def __init__(self):
        super().__init__()
        self.file_type = FileType.UNKNOWN

    def analyze_file(self, file_path: str) -> AnalysisResult:
        """Analyse un fichier générique et retourne un AnalysisResult"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Mesurer le temps de traitement
            start_time = datetime.now()

            # Créer le résultat de base
            result = self._create_base_result(file_path)
            result.analyzer_name = "GenericAnalyzer"

            # Analyser le contenu
            analysis = self.analyze_content(content, file_path)

            # Mettre à jour les résultats
            self._update_result_from_analysis(result, analysis)
            result.metrics = self._calculate_generic_metrics(content, analysis)

            # Déterminer le type de fichier si possible
            self._detect_file_type(result, file_path, content)

            # Calculer le temps de traitement
            result.processing_time_ms = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )

            return result

        except UnicodeDecodeError:
            # Essayer avec différents encodages
            return self._analyze_binary_file(file_path)
        except Exception as e:
            print(traceback.format_exc())
            logger.error(f"Error analyzing generic file {file_path}: {e}")
            return self._create_error_result(file_path, str(e))

    def analyze_content(self, content: str, file_path: str) -> Dict[str, Any]:
        """Analyse du contenu générique (méthode interne)"""
        return {
            'content_preview': content[:1000],
            'content_length': len(content),
            'line_count': len(content.splitlines()),
            'encoding_guess': self._guess_encoding(content),
            'contains_binary': self._contains_binary_data(content),
            'analysis': self._analyze_generic_content(content, file_path),
            'detected_patterns': self._detect_patterns(content, file_path)
        }

    def _analyze_generic_content(self, content: str, file_path: str) -> Dict[str, Any]:
        """Analyse le contenu générique"""
        lines = content.splitlines()

        return {
            'first_lines': lines[:10] if len(lines) > 10 else lines,
            'last_lines': lines[-5:] if len(lines) > 5 else lines,
            'empty_lines': len([l for l in lines if not l.strip()]),
            'comment_lines': len([l for l in lines if l.strip().startswith('#')]),
            'max_line_length': max((len(l) for l in lines), default=0),
            'avg_line_length': sum(len(l) for l in lines) / len(lines) if lines else 0,
            'character_distribution': self._analyze_character_distribution(content),
            'word_analysis': self._analyze_words(content)
        }

    def _detect_patterns(self, content: str, file_path: str) -> Dict[str, bool]:
        """Détecte les patterns dans le contenu"""
        filename = os.path.basename(file_path).lower()

        return {
            'is_markdown': self._is_markdown(content, filename),
            'is_dockerfile': self._is_dockerfile(content, filename),
            'is_shell_script': self._is_shell_script(content, filename),
            'is_makefile': self._is_makefile(content, filename),
            'is_gitignore': self._is_gitignore(content, filename),
            'is_env_file': self._is_env_file(content, filename),
            'is_license_file': self._is_license_file(content, filename),
            'is_readme': self._is_readme(content, filename),
            'is_log_file': self._is_log_file(content, filename)
        }

    def _is_markdown(self, content: str, filename: str) -> bool:
        """Détecte si c'est un fichier Markdown"""
        if filename.endswith('.md') or filename.endswith('.markdown'):
            return True

        # Vérifier les motifs Markdown
        markdown_patterns = [
            r'^# ',  # Titres
            r'\*\*.*\*\*',  # Gras
            r'\*.*\*',  # Italique
            r'\[.*\]\(.*\)',  # Liens
            r'^- ',  # Listes
            r'^\|',  # Tableaux
            r'```',  # Blocs de code
        ]

        return any(re.search(pattern, content, re.MULTILINE) for pattern in markdown_patterns)

    def _is_dockerfile(self, content: str, filename: str) -> bool:
        """Détecte si c'est un Dockerfile"""
        if filename == 'dockerfile' or filename.endswith('.dockerfile'):
            return True

        # Vérifier les commandes Docker
        docker_commands = ['FROM ', 'RUN ', 'CMD ', 'LABEL ', 'EXPOSE ', 'ENV ', 'ADD ', 'COPY ']
        return any(cmd in content for cmd in docker_commands)

    def _is_shell_script(self, content: str, filename: str) -> bool:
        """Détecte si c'est un script shell"""
        if filename.endswith('.sh') or filename.endswith('.bash'):
            return True

        # Shebang pour shell
        if content.startswith('#!/bin/bash') or content.startswith('#!/bin/sh'):
            return True

        # Commandes shell courantes
        shell_patterns = [r'^\s*echo\s+', r'^\s*export\s+', r'^\s*if\s+\[', r'^\s*for\s+\w+\s+in']
        return any(re.search(pattern, content, re.MULTILINE) for pattern in shell_patterns)

    def _is_makefile(self, content: str, filename: str) -> bool:
        """Détecte si c'est un Makefile"""
        if filename.lower() in ['makefile', 'gnumakefile']:
            return True

        # Patterns Makefile
        make_patterns = [r'^\w+:', r'^\s*@', r'^\s*\$(?:\(|{)', r'^\s*.PHONY:']
        return any(re.search(pattern, content, re.MULTILINE) for pattern in make_patterns)

    def _is_gitignore(self, content: str, filename: str) -> bool:
        """Détecte si c'est un fichier .gitignore"""
        if filename == '.gitignore':
            return True

        # Patterns .gitignore
        gitignore_patterns = [r'^\*\.\w+$', r'^/', r'^#', r'^!\w']
        lines = content.splitlines()
        gitignore_lines = sum(1 for line in lines if any(re.match(pattern, line) for pattern in gitignore_patterns))
        return gitignore_lines > len(lines) * 0.5  # > 50% des lignes ressemblent à .gitignore

    def _is_env_file(self, content: str, filename: str) -> bool:
        """Détecte si c'est un fichier .env"""
        if filename.startswith('.env'):
            return True

        # Patterns .env
        env_patterns = [r'^\w+=\w+$', r'^\w+="[^"]*"$', r"^\w+='[^']*'$"]
        lines = content.splitlines()
        env_lines = sum(1 for line in lines if any(re.match(pattern, line) for pattern in env_patterns))
        return env_lines > len(lines) * 0.3  # > 30% des lignes ressemblent à .env

    def _is_license_file(self, content: str, filename: str) -> bool:
        """Détecte si c'est un fichier de licence"""
        filename_lower = filename.lower()
        if 'license' in filename_lower or 'licence' in filename_lower:
            return True

        # Contenu typique des licences
        license_keywords = ['MIT License', 'Apache License', 'GNU General Public', 'Copyright', 'All rights reserved']
        return any(keyword in content for keyword in license_keywords)

    def _is_readme(self, content: str, filename: str) -> bool:
        """Détecte si c'est un README"""
        filename_lower = filename.lower()
        if filename_lower.startswith('readme'):
            return True

        # Contenu typique des README
        readme_patterns = [r'^#\s+\w+', r'^##\s+Installation', r'^##\s+Usage', r'^##\s+Contributing']
        return any(re.search(pattern, content, re.MULTILINE) for pattern in readme_patterns)

    def _is_log_file(self, content: str, filename: str) -> bool:
        """Détecte si c'est un fichier de log"""
        if filename.endswith('.log'):
            return True

        # Patterns de logs
        log_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # Dates
            r'\d{2}:\d{2}:\d{2}',  # Heures
            r'\[(INFO|ERROR|WARN|DEBUG)\]',  # Niveaux de log
            r'^\d',  # Lignes commençant par des chiffres (timestamp)
        ]

        lines = content.splitlines()
        log_lines = sum(1 for line in lines if any(re.search(pattern, line) for pattern in log_patterns))
        return log_lines > len(lines) * 0.3  # > 30% des lignes ressemblent à des logs

    def _guess_encoding(self, content: str) -> str:
        """Devine l'encodage du fichier"""
        # Analyse simplifiée
        try:
            content.encode('ascii')
            return 'ascii'
        except UnicodeEncodeError:
            try:
                content.encode('utf-8')
                return 'utf-8'
            except:
                return 'unknown'

    def _contains_binary_data(self, content: str) -> bool:
        """Vérifie si le contenu contient des données binaires"""
        # Si le contenu contient beaucoup de caractères non imprimables
        printable = sum(1 for c in content if 32 <= ord(c) <= 126 or c in '\n\r\t')
        return (printable / len(content)) < 0.9 if content else False

    def _analyze_character_distribution(self, content: str) -> Dict[str, float]:
        """Analyse la distribution des caractères"""
        if not content:
            return {}

        total_chars = len(content)
        char_categories = {
            'letters': sum(1 for c in content if c.isalpha()),
            'digits': sum(1 for c in content if c.isdigit()),
            'spaces': sum(1 for c in content if c.isspace()),
            'punctuation': sum(1 for c in content if c in '.,;:!?\'"()[]{}'),
            'special': total_chars - sum(1 for c in content if c.isprintable())
        }

        return {k: v / total_chars for k, v in char_categories.items()}

    def _analyze_words(self, content: str) -> Dict[str, Any]:
        """Analyse les mots dans le contenu"""
        words = re.findall(r'\b\w+\b', content)

        if not words:
            return {}

        word_lengths = [len(w) for w in words]

        return {
            'word_count': len(words),
            'unique_words': len(set(words)),
            'avg_word_length': sum(word_lengths) / len(words),
            'max_word_length': max(word_lengths) if word_lengths else 0,
            'common_words': self._find_common_words(words)
        }

    def _find_common_words(self, words: List[str], top_n: int = 10) -> List[Dict[str, Any]]:
        """Trouve les mots les plus communs"""
        from collections import Counter
        word_counts = Counter(words)
        return [{'word': word, 'count': count} for word, count in word_counts.most_common(top_n)]

    def _calculate_generic_metrics(self, content: str, analysis: Dict[str, Any]) -> FileMetrics:
        """Calcule les métriques génériques"""
        metrics = super()._calculate_metrics(content)

        # Mettre à jour avec des métriques spécifiques
        metrics.total_lines = analysis.get('line_count', 0)
        metrics.file_size_bytes = analysis.get('content_length', 0)

        # Analyser les lignes pour déterminer le type de contenu
        lines = content.splitlines()
        code_like_lines = len([l for l in lines if self._looks_like_code_line(l)])
        comment_like_lines = len([l for l in lines if l.strip().startswith('#') or l.strip().startswith('//')])

        metrics.code_lines = code_like_lines
        metrics.comment_lines = comment_like_lines
        metrics.blank_lines = len([l for l in lines if not l.strip()])

        # Complexité basée sur la structure
        sections = self._detect_sections(content[:1000])
        blocks = self._detect_content_blocks(content[:1000])

        complexity = len(sections) * 0.5 + len(blocks) * 0.3 + code_like_lines * 0.1
        metrics.complexity_score = complexity

        return metrics

    def _looks_like_code_line(self, line: str) -> bool:
        """Détermine si une ligne ressemble à du code"""
        stripped = line.strip()
        if not stripped:
            return False

        # Patterns de code
        code_patterns = [
            r'^\s*\w+\s*[=:]\s*\S+',  # Assignations
            r'^\s*(if|for|while|def|function|class)\b',  # Structures de contrôle
            r'^\s*\w+\(.*\)',  # Appels de fonction
            r'[{}()\[\]]',  # Parenthèses/accollades
            r';\s*$',  # Point-virgule à la fin
        ]

        return any(re.search(pattern, stripped) for pattern in code_patterns)

    def _detect_file_type(self, result: AnalysisResult, file_path: str, content: str) -> None:
        """Détecte et met à jour le type de fichier"""
        filename = os.path.basename(file_path).lower()

        # Basé sur l'extension et le contenu
        if filename.endswith('.md') or self._is_markdown(content, filename):
            result.file_type = FileType.MARKDOWN
        elif filename == 'dockerfile' or self._is_dockerfile(content, filename):
            result.file_type = FileType.DOCKERFILE
        elif filename.endswith('.sh') or self._is_shell_script(content, filename):
            result.file_type = FileType.SHELL
        elif filename == '.gitignore' or self._is_gitignore(content, filename):
            result.file_type = FileType.UNKNOWN  # Pas d'enum spécifique
        elif self._is_makefile(content, filename):
            result.file_type = FileType.SHELL  # Approximatif
        elif self._is_env_file(content, filename):
            result.file_type = FileType.YAML  # Approximatif
        elif self._is_license_file(content, filename):
            result.file_type = FileType.MARKDOWN  # Approximatif
        elif self._is_readme(content, filename):
            result.file_type = FileType.MARKDOWN
        elif self._is_log_file(content, filename):
            result.file_type = FileType.UNKNOWN
        else:
            # Essayer de détecter basé sur le contenu
            if self._looks_like_code(content[:1000]):
                result.notes.append("Content appears to be code-like")
            elif len(content.splitlines()) == 1 and len(content) < 100:
                result.notes.append("Single line/short file")

    def _analyze_binary_file(self, file_path: str) -> AnalysisResult:
        """Analyse un fichier binaire ou non-textuel"""
        try:
            file_size = os.path.getsize(file_path)

            # Créer un résultat de base
            result = self._create_base_result(file_path)
            result.analyzer_name = "GenericAnalyzer"
            result.status = AnalysisStatus.PARTIAL
            result.file_type = FileType.UNKNOWN

            # Essayer de lire les premiers bytes pour analyse
            with open(file_path, 'rb') as f:
                first_bytes = f.read(1024)

            # Analyse basique du binaire
            binary_analysis = self._analyze_binary_data(first_bytes, file_path)

            # Mettre à jour les résultats
            result.metrics = FileMetrics(
                file_size_bytes=file_size,
                total_lines=0,
                code_lines=0,
                comment_lines=0,
                blank_lines=0
            )

            result.language_specific = {
                'generic': {
                    'content_type': 'binary',
                    'file_size': file_size,
                    'is_binary': True,
                    'binary_analysis': binary_analysis
                }
            }

            result.notes.append("Binary file - limited analysis available")

            return result

        except Exception as e:
            logger.error(f"Error analyzing binary file {file_path}: {e}")
            return self._create_error_result(file_path, f"Binary file analysis error: {str(e)}")

    def _analyze_binary_data(self, data: bytes, file_path: str) -> Dict[str, Any]:
        """Analyse les données binaires"""
        filename = os.path.basename(file_path).lower()

        # Vérifier les signatures magiques courantes
        magic_numbers = {
            b'\x89PNG\r\n\x1a\n': 'png_image',
            b'\xff\xd8\xff': 'jpeg_image',
            b'GIF87a': 'gif_image',
            b'GIF89a': 'gif_image',
            b'%PDF': 'pdf_document',
            b'PK\x03\x04': 'zip_archive',
            b'\x1f\x8b\x08': 'gzip_archive',
            b'Rar!\x1a\x07': 'rar_archive',
            b'\x00\x00\x01\x00': 'ico_image',
            b'\x00\x00\x02\x00': 'cur_image',
            b'BM': 'bmp_image',
            b'\x49\x49\x2a\x00': 'tiff_image_little_endian',
            b'\x4d\x4d\x00\x2a': 'tiff_image_big_endian',
            b'\x7fELF': 'elf_executable'
        }

        detected_type = 'unknown_binary'
        for magic, file_type in magic_numbers.items():
            if data.startswith(magic):
                detected_type = file_type
                break

        # Analyser la distribution des bytes
        byte_counts = {}
        for byte in data:
            byte_counts[byte] = byte_counts.get(byte, 0) + 1

        return {
            'detected_type': detected_type,
            'data_length': len(data),
            'null_bytes': byte_counts.get(0, 0),
            'printable_bytes': sum(1 for b in data if 32 <= b <= 126 or b in (9, 10, 13)),
            'is_likely_text': self._is_likely_text(data),
            'file_extension_hint': os.path.splitext(filename)[1]
        }

    def _is_likely_text(self, data: bytes) -> bool:
        """Détermine si les données ressemblent à du texte"""
        if len(data) == 0:
            return False

        printable = sum(1 for b in data if 32 <= b <= 126 or b in (9, 10, 13))
        return printable / len(data) > 0.9
