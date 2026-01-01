# json_analyzer.py
import json
import logging
from typing import Dict, Any

from james_parsers.analyzer import Analyzer

logger = logging.getLogger(__name__)


class JSONAnalyzer(Analyzer):
    def analyze(self, file_path: str) -> Dict[str, Any]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return self.analyze_content(content, file_path)
        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {e}")
            return {
                'file_type': 'json',
                'file_path': file_path,
                'is_valid': False,
                'error': str(e)
            }

    def analyze_content(self, content: str, file_path: str) -> Dict[str, Any]:
        try:
            data = json.loads(content)
            return {
                'file_type': 'json',
                'file_path': file_path,
                'is_valid': True,
                'structure': self._analyze_structure(data),
                'schema_type': self._detect_schema_type(data),
                'size_metrics': self._calculate_metrics(data)
            }
        except json.JSONDecodeError as e:
            return {
                'file_type': 'json',
                'is_valid': False,
                'error': str(e)
            }

    def _analyze_structure(self, data: Any) -> Dict[str, Any]:
        if isinstance(data, dict):
            return {
                'type': 'object',
                'keys': list(data.keys()),
                'key_count': len(data),
                'nested_objects': self._count_nested_objects(data),
                'value_types': self._analyze_value_types(data)
            }
        elif isinstance(data, list):
            return {
                'type': 'array',
                'length': len(data),
                'item_types': self._analyze_array_types(data)
            }
        else:
            return {'type': 'primitive', 'value_type': type(data).__name__}

    def _count_nested_objects(self, data: dict) -> int:
        count = 0
        for value in data.values():
            if isinstance(value, dict):
                count += 1 + self._count_nested_objects(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        count += 1 + self._count_nested_objects(item)
        return count

    def _analyze_value_types(self, data: dict) -> Dict[str, int]:
        type_count = {}
        for value in data.values():
            value_type = type(value).__name__
            type_count[value_type] = type_count.get(value_type, 0) + 1
        return type_count

    def _analyze_array_types(self, data: list) -> Dict[str, int]:
        if not data:
            return {'empty': 1}

        type_count = {}
        for item in data:
            item_type = type(item).__name__
            type_count[item_type] = type_count.get(item_type, 0) + 1
        return type_count

    def _detect_schema_type(self, data: Any) -> str:
        if isinstance(data, dict):
            if 'schema' in data or '$schema' in data:
                return 'json_schema'
            elif 'openapi' in str(data).lower() or 'swagger' in str(data).lower():
                return 'openapi'
            elif 'paths' in data and 'components' in data:
                return 'openapi'
            elif all(key in data for key in ['name', 'version', 'dependencies']):
                return 'package_json'
            elif any(key in data for key in ['scripts', 'devDependencies']):
                return 'package_json'
        return 'generic'

    def _calculate_metrics(self, data: Any) -> Dict[str, int]:
        def count_elements(obj):
            if isinstance(obj, dict):
                return 1 + sum(count_elements(v) for v in obj.values())
            elif isinstance(obj, list):
                return 1 + sum(count_elements(item) for item in obj)
            else:
                return 1

        return {
            'total_elements': count_elements(data),
            'depth': self._calculate_depth(data)
        }

    def _calculate_depth(self, obj, current_depth=0):
        if isinstance(obj, dict):
            if obj:
                return max(self._calculate_depth(v, current_depth + 1) for v in obj.values())
            else:
                return current_depth + 1
        elif isinstance(obj, list):
            if obj:
                return max(self._calculate_depth(item, current_depth + 1) for item in obj)
            else:
                return current_depth + 1
        else:
            return current_depth
