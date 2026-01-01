# html_analyzer.py
import logging
import re
from typing import Dict, List, Any

from parsers.analyzer import Analyzer

logger = logging.getLogger(__name__)


class HTMLAnalyzer(Analyzer):
    def analyze(self, file_path: str) -> Dict[str, Any]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return self.analyze_content(content, file_path)
        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {e}")
            return {
                'file_type': 'html',
                'file_path': file_path,
                'error': str(e)
            }

    def analyze_content(self, content: str, file_path: str) -> Dict[str, Any]:
        return {
            'file_type': 'html',
            'file_path': file_path,
            'doctype': self._extract_doctype(content),
            'root_element': self._extract_root_element(content),
            'elements': self._extract_elements(content),
            'attributes': self._extract_attributes(content),
            'scripts': self._extract_scripts(content),
            'styles': self._extract_styles(content),
            'analysis': self._analyze_html_structure(content)
        }

    def _extract_doctype(self, content: str) -> str:
        match = re.search(r'<!DOCTYPE[^>]*>', content)
        return match.group(0) if match else ""

    def _extract_root_element(self, content: str) -> Dict[str, str]:
        match = re.search(r'<html[^>]*>', content)
        if match:
            return {
                'tag': 'html',
                'attributes': self._extract_element_attributes(match.group(0))
            }
        return {}

    def _extract_elements(self, content: str) -> List[Dict]:
        elements = []
        # Pattern pour capturer les balises HTML
        tag_pattern = r'<(\w+)([^>]*)>'

        for match in re.finditer(tag_pattern, content):
            tag_name = match.group(1).lower()
            if tag_name not in ['!doctype', 'meta', 'link']:  # Exclure certains tags
                elements.append({
                    'tag': tag_name,
                    'attributes': self._extract_element_attributes(match.group(2)),
                    'is_self_closing': match.group(0).endswith('/>')
                })

        return elements

    def _extract_element_attributes(self, attributes_str: str) -> Dict[str, str]:
        attributes = {}
        attr_pattern = r'(\w+)\s*=\s*["\']([^"\']*)["\']'

        for match in re.finditer(attr_pattern, attributes_str):
            attributes[match.group(1)] = match.group(2)

        return attributes

    def _extract_attributes(self, content: str) -> List[Dict]:
        attributes = []
        attr_pattern = r'(\w+)\s*=\s*["\']([^"\']*)["\']'

        for match in re.finditer(attr_pattern, content):
            attributes.append({
                'name': match.group(1),
                'value': match.group(2)
            })

        return attributes

    def _extract_scripts(self, content: str) -> List[Dict]:
        scripts = []
        script_pattern = r'<script([^>]*)>([\s\S]*?)</script>'

        for match in re.finditer(script_pattern, content):
            scripts.append({
                'attributes': self._extract_element_attributes(match.group(1)),
                'content_length': len(match.group(2)),
                'has_src': 'src=' in match.group(1)
            })

        return scripts

    def _extract_styles(self, content: str) -> List[Dict]:
        styles = []
        style_pattern = r'<style([^>]*)>([\s\S]*?)</style>'

        for match in re.finditer(style_pattern, content):
            styles.append({
                'attributes': self._extract_element_attributes(match.group(1)),
                'content_length': len(match.group(2))
            })

        return styles

    def _analyze_html_structure(self, content: str) -> Dict[str, Any]:
        elements = self._extract_elements(content)
        element_counts = {}

        for element in elements:
            tag = element['tag']
            element_counts[tag] = element_counts.get(tag, 0) + 1

        return {
            'total_elements': len(elements),
            'element_counts': element_counts,
            'has_forms': any(element['tag'] == 'form' for element in elements),
            'has_tables': any(element['tag'] == 'table' for element in elements),
            'has_images': any(element['tag'] == 'img' for element in elements),
            'script_count': len(self._extract_scripts(content)),
            'style_count': len(self._extract_styles(content)),
            'semantic_elements': self._count_semantic_elements(elements)
        }

    def _count_semantic_elements(self, elements: List[Dict]) -> Dict[str, int]:
        semantic_tags = ['header', 'footer', 'nav', 'main', 'article', 'section', 'aside']
        counts = {}

        for tag in semantic_tags:
            counts[tag] = sum(1 for element in elements if element['tag'] == tag)

        return counts
