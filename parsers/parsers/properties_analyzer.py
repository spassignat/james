# properties_analyzer.py
import logging
import re
from typing import Dict, List, Any

from parsers.analyzer import Analyzer

logger = logging.getLogger(__name__)


class PropertiesAnalyzer(Analyzer):
    def analyze(self, file_path: str) -> Dict[str, Any]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return self.analyze_content(content, file_path)
        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {e}")
            return {
                'file_type': 'properties',
                'file_path': file_path,
                'is_valid': False,
                'error': str(e)
            }

    def analyze_content(self, content: str, file_path: str) -> Dict[str, Any]:
        properties = []
        lines = content.split('\n')

        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, _, value = line.partition('=')
                properties.append({
                    'key': key.strip(),
                    'value': value.strip(),
                    'line': line_num,
                    'has_expression': self._has_expression(value)
                })

        return properties

    def _extract_sections(self, content: str) -> List[Dict]:
        sections = []
        current_section = None

        lines = content.split('\n')
        for line_num, line in enumerate(lines, 1):
            line = line.strip()

            # Section header
            if line.startswith('[') and line.endswith(']'):
                if current_section:
                    sections.append(current_section)

                current_section = {
                    'name': line[1:-1],
                    'start_line': line_num,
                    'properties': []
                }
            elif current_section and '=' in line and not line.startswith('#'):
                key, _, value = line.partition('=')
                current_section['properties'].append({
                    'key': key.strip(),
                    'value': value.strip()
                })

        if current_section:
            sections.append(current_section)

        return sections

    def _has_expression(self, value: str) -> bool:
        # Détecte les expressions ${...}, #{...}, etc.
        return bool(re.search(r'\$\{.*?\}|#\{.*?\}|%\{.*?\}', value))

    def _analyze_properties_patterns(self, content: str) -> Dict[str, Any]:
        properties = self._extract_properties(content)

        return {
            'total_properties': len(properties),
            'commented_lines': len([l for l in content.split('\n') if l.strip().startswith('#')]),
            'has_expressions': any(p['has_expression'] for p in properties),
            'common_patterns': self._detect_common_patterns(properties),
            'property_categories': self._categorize_properties(properties)
        }

    def _detect_common_patterns(self, properties: List[Dict]) -> List[str]:
        patterns = []
        keys = [p['key'] for p in properties]
        values = [p['value'] for p in properties]

        # Détection des patterns communs
        if any('url' in key.lower() for key in keys):
            patterns.append('database_urls')
        if any('password' in key.lower() or 'secret' in key.lower() for key in keys):
            patterns.append('sensitive_data')
        if any('port' in key.lower() for key in keys):
            patterns.append('network_ports')
        if any('timeout' in key.lower() for key in keys):
            patterns.append('timeout_configurations')

        return patterns

    def _categorize_properties(self, properties: List[Dict]) -> Dict[str, int]:
        categories = {
            'database': 0,
            'server': 0,
            'security': 0,
            'logging': 0,
            'external_services': 0,
            'other': 0
        }

        for prop in properties:
            key = prop['key'].lower()

            if any(db_word in key for db_word in ['db', 'database', 'jdbc', 'sql']):
                categories['database'] += 1
            elif any(server_word in key for server_word in ['server', 'port', 'host', 'url']):
                categories['server'] += 1
            elif any(sec_word in key for sec_word in ['password', 'secret', 'key', 'auth', 'ssl']):
                categories['security'] += 1
            elif any(log_word in key for log_word in ['log', 'debug', 'verbose']):
                categories['logging'] += 1
            elif any(ext_word in key for ext_word in ['api', 'service', 'endpoint', 'client']):
                categories['external_services'] += 1
            else:
                categories['other'] += 1

        return categories
