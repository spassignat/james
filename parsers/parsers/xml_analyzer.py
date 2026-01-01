# xml_analyzer.py
import logging
import re
from typing import Dict, List, Any

from parsers.analyzer import Analyzer

logger = logging.getLogger(__name__)


class XMLAnalyzer(Analyzer):
    def analyze(self, file_path: str) -> Dict[str, Any]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return self.analyze_content(content, file_path)
        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {e}")
            return {
                'file_type': 'xml',
                'file_path': file_path,
                'is_valid': False,
                'error': str(e)
            }

    def analyze_content(self, content: str, file_path: str) -> Dict[str, Any]:
        return {
            'file_type': 'xml',
            'root_element': self._extract_root_element(content),
            'elements': self._extract_elements(content),
            'attributes': self._extract_attributes(content),
            'namespaces': self._extract_namespaces(content),
            'analysis': self._analyze_xml_structure(content)
        }

    def _extract_root_element(self, content: str) -> Dict[str, str]:
        match = re.search(r'<(\w+)[^>]*>', content)
        if match:
            return {
                'name': match.group(1),
                'attributes': self._extract_element_attributes(match.group(0))
            }
        return {}

    def _extract_elements(self, content: str) -> List[Dict]:
        elements = []
        # Pattern pour capturer les éléments avec leurs attributs
        element_pattern = r'<(\w+)([^>]*)>'

        for match in re.finditer(element_pattern, content):
            elements.append({
                'name': match.group(1),
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

    def _extract_namespaces(self, content: str) -> List[Dict]:
        namespaces = []
        ns_pattern = r'xmlns:?(\w*)\s*=\s*["\']([^"\']*)["\']'

        for match in re.finditer(ns_pattern, content):
            namespaces.append({
                'prefix': match.group(1) or 'default',
                'uri': match.group(2)
            })

        return namespaces

    def _analyze_xml_structure(self, content: str) -> Dict[str, Any]:
        return {
            'element_count': len(self._extract_elements(content)),
            'attribute_count': len(self._extract_attributes(content)),
            'max_depth': self._calculate_max_depth(content),
            'has_cdata': '<![CDATA[' in content,
            'has_comments': '<!--' in content,
            'schema_type': self._detect_schema_type(content)
        }

    def _calculate_max_depth(self, content: str) -> int:
        depth = 0
        max_depth = 0
        stack = []

        # Pattern pour les balises ouvrantes et fermantes
        tag_pattern = r'</?(\w+)[^>]*>'

        for match in re.finditer(tag_pattern, content):
            tag = match.group(0)
            if tag.startswith('</'):
                # Balise fermante
                if stack:
                    stack.pop()
                    depth = len(stack)
            elif not tag.endswith('/>'):
                # Balise ouvrante (pas auto-fermante)
                stack.append(match.group(1))
                depth = len(stack)
                max_depth = max(max_depth, depth)

        return max_depth

    def _detect_schema_type(self, content: str) -> str:
        if any(schema in content for schema in ['web-app', 'servlet', 'filter']):
            return 'web_xml'
        elif any(spring in content for spring in ['beans', 'context:', 'aop:']):
            return 'spring_config'
        elif any(maven in content for maven in ['project', 'groupId', 'artifactId']):
            return 'maven_pom'
        elif any(config in content for config in ['configuration', 'appSettings']):
            return 'app_config'
        return 'generic'
