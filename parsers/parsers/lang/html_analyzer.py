# html_analyzer.py
import logging
import re
from typing import Dict, List, Any
from datetime import datetime

from parsers.analyzer import Analyzer
from parsers.analysis_result import (
    AnalysisResult, AnalysisStatus, FileType, FrameworkType,
    CodeElement, FileMetrics, PatternDetection, DependencyInfo,
    SecurityAnalysis, SectionAnalysis
)

logger = logging.getLogger(__name__)


class HTMLAnalyzer(Analyzer):
    """Analyseur de fichiers HTML retournant des AnalysisResult"""

    def __init__(self):
        super().__init__()
        self.file_type = FileType.HTML

    def analyze_file(self, file_path: str) -> AnalysisResult:
        """Analyse un fichier HTML et retourne un AnalysisResult"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Mesurer le temps de traitement
            start_time = datetime.now()

            # Créer le résultat de base
            result = self._create_base_result(file_path)
            result.analyzer_name = "HTMLAnalyzer"

            # Analyser le contenu
            analysis = self.analyze_content(content, file_path)

            # Mettre à jour les résultats
            self._update_result_from_analysis(result, analysis)
            result.metrics = self._calculate_metrics(content)

            # Calculer le temps de traitement
            result.processing_time_ms = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )

            return result

        except Exception as e:
            logger.error(f"Error analyzing HTML file {file_path}: {e}")
            return self._create_error_result(file_path, str(e))

    def analyze_content(self, content: str, file_path: str) -> Dict[str, Any]:
        """Analyse du contenu HTML (méthode interne)"""
        return {
            'doctype': self._extract_doctype(content),
            'root_element': self._extract_root_element(content),
            'elements': self._extract_elements(content),
            'attributes': self._extract_attributes(content),
            'scripts': self._extract_scripts(content),
            'styles': self._extract_styles(content),
            'analysis': self._analyze_html_structure(content),
            'links': self._extract_links(content),
            'meta_tags': self._extract_meta_tags(content)
        }

    def _update_result_from_analysis(self, result: AnalysisResult, analysis: Dict[str, Any]) -> None:
        """Met à jour l'AnalysisResult avec les données de l'analyse HTML"""

        # Ajouter les éléments de code
        self._add_html_elements(result, analysis['elements'])

        # Mettre à jour les métriques spécifiques
        result.metrics.function_count = analysis['analysis'].get('script_count', 0)
        result.metrics.import_count = len(analysis.get('scripts', []))

        # Mettre à jour les dépendances
        self._update_dependencies(result, analysis)

        # Mettre à jour les patterns
        self._update_patterns(result, analysis)

        # Mettre à jour les sections (template/script/style)
        self._update_sections(result, analysis)

        # Mettre à jour les données spécifiques au langage
        self._update_language_specific(result, analysis)

        # Ajouter des diagnostics
        self._add_diagnostics(result, analysis)

    def _add_html_elements(self, result: AnalysisResult, elements: List[Dict]) -> None:
        """Convertit les éléments HTML en CodeElement"""
        for elem in elements:
            code_element = CodeElement(
                name=elem['tag'],
                element_type='html_tag',
                metadata={
                    'attributes': elem['attributes'],
                    'is_self_closing': elem.get('is_self_closing', False)
                }
            )
            result.elements.append(code_element)

    def _update_dependencies(self, result: AnalysisResult, analysis: Dict[str, Any]) -> None:
        """Met à jour les informations de dépendances"""
        deps = result.dependencies

        # Scripts externes
        for script in analysis.get('scripts', []):
            if script.get('has_src'):
                src = script['attributes'].get('src', '')
                if src.startswith('http') or src.endswith('.js'):
                    deps.external_deps.append(src)

        # Feuilles de style externes
        for link in analysis.get('links', []):
            href = link['attributes'].get('href', '')
            if href.endswith('.css'):
                deps.external_deps.append(href)

        # Dépendances internes
        deps.internal_deps = list(set(deps.internal_deps))  # Dédupliquer

    def _update_patterns(self, result: AnalysisResult, analysis: Dict[str, Any]) -> None:
        """Met à jour la détection de patterns"""
        patterns = result.patterns

        # Détecter les frameworks basés sur les classes CSS
        if self._detect_framework(analysis):
            patterns.frameworks.append(FrameworkType.VANILLA)

        # Architecture hints
        if analysis['analysis'].get('has_forms'):
            patterns.architecture_hints.append('has_forms')
        if analysis['analysis'].get('has_tables'):
            patterns.architecture_hints.append('has_tables')
        if analysis['analysis'].get('semantic_elements', {}):
            patterns.architecture_hints.append('semantic_html')

    def _update_sections(self, result: AnalysisResult, analysis: Dict[str, Any]) -> None:
        """Met à jour les sections (template, script, style)"""

        # Template section (le HTML principal)
        template_section = SectionAnalysis(
            section_type='template',
            content=self._extract_html_body(analysis),
            language='html',
            metrics=FileMetrics(
                total_lines=result.metrics.total_lines,
                code_lines=result.metrics.code_lines
            ),
            analysis={
                'element_count': analysis['analysis'].get('total_elements', 0),
                'semantic_score': self._calculate_semantic_score(analysis)
            }
        )
        result.sections['template'] = template_section

        # Script sections
        for i, script in enumerate(analysis.get('scripts', [])):
            if not script.get('has_src'):
                script_section = SectionAnalysis(
                    section_type='script',
                    language='javascript',
                    analysis={
                        'is_inline': True,
                        'content_length': script.get('content_length', 0)
                    }
                )
                result.sections[f'script_{i}'] = script_section

        # Style sections
        for i, style in enumerate(analysis.get('styles', [])):
            style_section = SectionAnalysis(
                section_type='style',
                language='css',
                analysis={
                    'is_inline': True,
                    'content_length': style.get('content_length', 0)
                }
            )
            result.sections[f'style_{i}'] = style_section

    def _update_language_specific(self, result: AnalysisResult, analysis: Dict[str, Any]) -> None:
        """Met à jour les données spécifiques au HTML"""
        result.language_specific = {
            'html': {
                'doctype': analysis.get('doctype', ''),
                'root_element': analysis.get('root_element', {}),
                'element_breakdown': analysis['analysis'].get('element_counts', {}),
                'semantic_elements': analysis['analysis'].get('semantic_elements', {}),
                'has_head': '<head>' in analysis.get('analysis', {}),
                'has_body': '<body>' in analysis.get('analysis', {}),
                'is_html5': analysis.get('doctype', '').upper() == '<!DOCTYPE HTML>'
            }
        }

    def _add_diagnostics(self, result: AnalysisResult, analysis: Dict[str, Any]) -> None:
        """Ajoute des diagnostics et recommandations"""

        # Vérifier les problèmes courants
        if not analysis.get('doctype'):
            result.warnings.append("No DOCTYPE declaration found")

        if not analysis['analysis'].get('semantic_elements'):
            result.notes.append("Consider using more semantic HTML elements")

        if len(analysis.get('scripts', [])) > 5:
            result.warnings.append("Too many script tags, consider bundling")

        if len(analysis.get('styles', [])) > 3:
            result.notes.append("Consider external CSS files for better maintainability")

    def _extract_doctype(self, content: str) -> str:
        """Extrait la déclaration DOCTYPE"""
        match = re.search(r'<!DOCTYPE[^>]*>', content, re.IGNORECASE)
        return match.group(0) if match else ""

    def _extract_root_element(self, content: str) -> Dict[str, Any]:
        """Extrait l'élément racine HTML"""
        match = re.search(r'<html([^>]*)>', content, re.IGNORECASE)
        if match:
            return {
                'tag': 'html',
                'attributes': self._extract_element_attributes(match.group(1)),
                'has_lang': 'lang=' in match.group(1)
            }
        return {}

    def _extract_elements(self, content: str) -> List[Dict]:
        """Extrait tous les éléments HTML"""
        elements = []
        tag_pattern = r'<([a-zA-Z][a-zA-Z0-9]*)([^>]*)>'

        for match in re.finditer(tag_pattern, content):
            tag_name = match.group(1).lower()
            if tag_name not in ['!doctype']:  # Exclure doctype
                elements.append({
                    'tag': tag_name,
                    'attributes': self._extract_element_attributes(match.group(2)),
                    'is_self_closing': match.group(0).endswith('/>'),
                    'is_closing': match.group(0).startswith('</')
                })

        return elements

    def _extract_element_attributes(self, attributes_str: str) -> Dict[str, str]:
        """Extrait les attributs d'un élément"""
        attributes = {}
        attr_pattern = r'([a-zA-Z_][a-zA-Z0-9_-]*)\s*=\s*["\']([^"\']*)["\']'

        for match in re.finditer(attr_pattern, attributes_str):
            attributes[match.group(1).lower()] = match.group(2)

        # Ajouter les attributs sans valeur
        bool_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_-]*)\b(?=\s|>|/)'
        for match in re.finditer(bool_pattern, attributes_str):
            attr = match.group(1).lower()
            if attr not in attributes:
                attributes[attr] = 'true'

        return attributes

    def _extract_scripts(self, content: str) -> List[Dict]:
        """Extrait les scripts"""
        scripts = []
        script_pattern = r'<script([^>]*)>([\s\S]*?)</script>'

        for match in re.finditer(script_pattern, content, re.IGNORECASE):
            scripts.append({
                'attributes': self._extract_element_attributes(match.group(1)),
                'content_length': len(match.group(2)),
                'has_src': 'src=' in match.group(1),
                'is_module': 'type="module"' in match.group(1) or 'type=module' in match.group(1)
            })

        return scripts

    def _extract_styles(self, content: str) -> List[Dict]:
        """Extrait les styles"""
        styles = []
        style_pattern = r'<style([^>]*)>([\s\S]*?)</style>'

        for match in re.finditer(style_pattern, content, re.IGNORECASE):
            styles.append({
                'attributes': self._extract_element_attributes(match.group(1)),
                'content_length': len(match.group(2)),
                'is_scoped': 'scoped' in match.group(1)
            })

        return styles

    def _extract_links(self, content: str) -> List[Dict]:
        """Extrait les liens"""
        links = []
        link_pattern = r'<link([^>]*)/?>'

        for match in re.finditer(link_pattern, content, re.IGNORECASE):
            links.append({
                'attributes': self._extract_element_attributes(match.group(1))
            })

        return links

    def _extract_meta_tags(self, content: str) -> List[Dict]:
        """Extrait les balises meta"""
        metas = []
        meta_pattern = r'<meta([^>]*)/?>'

        for match in re.finditer(meta_pattern, content, re.IGNORECASE):
            metas.append({
                'attributes': self._extract_element_attributes(match.group(1))
            })

        return metas

    def _extract_attributes(self, content: str) -> List[Dict]:
        """Extrait tous les attributs"""
        attributes = []
        attr_pattern = r'([a-zA-Z_][a-zA-Z0-9_-]*)\s*=\s*["\']([^"\']*)["\']'

        for match in re.finditer(attr_pattern, content):
            attributes.append({
                'name': match.group(1),
                'value': match.group(2)
            })

        return attributes

    def _analyze_html_structure(self, content: str) -> Dict[str, Any]:
        """Analyse la structure HTML"""
        elements = self._extract_elements(content)
        element_counts = {}
        semantic_elements = self._count_semantic_elements(elements)

        for element in elements:
            if not element.get('is_closing'):
                tag = element['tag']
                element_counts[tag] = element_counts.get(tag, 0) + 1

        return {
            'total_elements': len([e for e in elements if not e.get('is_closing')]),
            'element_counts': element_counts,
            'has_forms': any(element['tag'] == 'form' for element in elements),
            'has_tables': any(element['tag'] == 'table' for element in elements),
            'has_images': any(element['tag'] == 'img' for element in elements),
            'script_count': len(self._extract_scripts(content)),
            'style_count': len(self._extract_styles(content)),
            'link_count': len(self._extract_links(content)),
            'meta_count': len(self._extract_meta_tags(content)),
            'semantic_elements': semantic_elements,
            'has_head': bool(re.search(r'<head[^>]*>', content, re.IGNORECASE)),
            'has_body': bool(re.search(r'<body[^>]*>', content, re.IGNORECASE))
        }

    def _count_semantic_elements(self, elements: List[Dict]) -> Dict[str, int]:
        """Compte les éléments sémantiques"""
        semantic_tags = [
            'header', 'footer', 'nav', 'main', 'article',
            'section', 'aside', 'figure', 'figcaption',
            'time', 'mark', 'summary', 'details'
        ]
        counts = {}

        for tag in semantic_tags:
            count = sum(1 for element in elements
                        if element['tag'] == tag and not element.get('is_closing'))
            if count > 0:
                counts[tag] = count

        return counts

    def _detect_framework(self, analysis: Dict[str, Any]) -> bool:
        """Détecte les frameworks basés sur les attributs"""
        # Vérifier les attributs communs des frameworks
        for element in analysis['elements']:
            attrs = element['attributes']
            if 'v-' in str(attrs) or ':' in str(attrs):
                return False  # Probablement Vue.js
            if '*ng' in str(attrs):
                return False  # Probablement Angular
            if 'data-react' in str(attrs):
                return False  # Probablement React

        return True  # Vanilla HTML

    def _extract_html_body(self, analysis: Dict[str, Any]) -> str:
        """Extrait le corps HTML pour la section template"""
        # Méthode simplifiée - dans une vraie implémentation, extraire le body
        return "<body>...</body>"

    def _calculate_semantic_score(self, analysis: Dict[str, Any]) -> float:
        """Calcule un score sémantique pour le HTML"""
        semantic_elements = analysis['analysis'].get('semantic_elements', {})
        total_elements = analysis['analysis'].get('total_elements', 1)

        if total_elements == 0:
            return 0.0

        semantic_count = sum(semantic_elements.values())
        return (semantic_count / total_elements) * 100

    def _calculate_metrics(self, content: str) -> FileMetrics:
        """Calcule les métriques spécifiques au HTML"""
        metrics = super()._calculate_metrics(content)

        # Ajouter des métriques spécifiques au HTML
        lines = content.split('\n')

        # Compter les lignes HTML
        html_lines = len([l for l in lines if re.search(r'<[^>]+>', l)])

        # Compter les commentaires HTML
        html_comments = len([l for l in lines if re.search(r'<!--.*?-->', l)])

        # Mettre à jour les métriques
        metrics.comment_lines = html_comments
        metrics.code_lines = html_lines

        # Calculer une complexité simple basée sur le nesting
        complexity = self._calculate_html_complexity(content)
        metrics.complexity_score = complexity

        return metrics

    def _calculate_html_complexity(self, content: str) -> float:
        """Calcule une métrique de complexité pour le HTML"""
        # Simplifié : compte la profondeur d'imbrication
        max_depth = 0
        current_depth = 0

        for line in content.split('\n'):
            open_tags = len(re.findall(r'<([a-zA-Z][a-zA-Z0-9]*)[^>]*>', line))
            close_tags = len(re.findall(r'</([a-zA-Z][a-zA-Z0-9]*)>', line))

            current_depth += open_tags - close_tags
            max_depth = max(max_depth, current_depth)

        return float(max_depth)