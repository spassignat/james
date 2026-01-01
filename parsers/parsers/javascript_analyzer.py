# javascript_analyzer.py
import logging
import re
from typing import Dict, List, Any

from parsers.analyzer import Analyzer

logger = logging.getLogger(__name__)


class JavaScriptAnalyzer(Analyzer):
    def analyze(self, file_path: str) -> Dict[str, Any]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return self.analyze_content(content, file_path)
        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {e}")
            return {
                'file_type': 'javascript',
                'file_path': file_path,
                'is_valid': False,
                'error': str(e)
            }

    def analyze_content(self, content: str, file_path: str) -> Dict[str, Any]:
        return {
            'file_type': 'javascript',
            'file_path': file_path,
            'imports': self._extract_imports(content),
            'exports': self._extract_exports(content),
            'functions': self._extract_functions(content),
            'classes': self._extract_classes(content),
            'variables': self._extract_variables(content),
            'analysis': self._analyze_js_patterns(content)
        }

    def _extract_imports(self, content: str) -> List[str]:
        imports = []
        # ES6 imports
        imports.extend(re.findall(r'import\s+.*?from\s+[\'"]([^\'"]+)[\'"]', content))
        # CommonJS requires
        imports.extend(re.findall(r'require\s*\(\s*[\'"]([^\'"]+)[\'"]\s*\)', content))
        return imports

    def _extract_exports(self, content: str) -> List[str]:
        exports = []
        # Named exports
        exports.extend(re.findall(r'export\s+(?:const|let|var|function|class)\s+(\w+)', content))
        # Default exports
        if 'export default' in content:
            exports.append('default')
        return exports

    def _extract_functions(self, content: str) -> List[Dict]:
        functions = []
        patterns = [
            r'function\s+(\w+)\s*\([^)]*\)',
            r'const\s+(\w+)\s*=\s*\([^)]*\)\s*=>',
            r'let\s+(\w+)\s*=\s*\([^)]*\)\s*=>',
            r'var\s+(\w+)\s*=\s*\([^)]*\)\s*=>',
            r'(\w+)\s*\([^)]*\)\s*\{'
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, content):
                functions.append({
                    'name': match.group(1),
                    'type': self._get_function_type(pattern)
                })

        return functions

    def _extract_classes(self, content: str) -> List[Dict]:
        classes = []
        pattern = r'class\s+(\w+)'

        for match in re.finditer(pattern, content):
            classes.append({
                'name': match.group(1),
                'methods': self._extract_class_methods(content, match.group(1))
            })

        return classes

    def _extract_class_methods(self, content: str, class_name: str) -> List[str]:
        # Pattern simplifié pour les méthodes de classe
        methods = re.findall(r'(\w+)\s*\([^)]*\)\s*\{', content)
        return [m for m in methods if m not in ['constructor', 'super']]

    def _extract_variables(self, content: str) -> List[Dict]:
        variables = []
        patterns = [
            r'const\s+(\w+)\s*=',
            r'let\s+(\w+)\s*=',
            r'var\s+(\w+)\s*='
        ]

        for pattern in patterns:
            variables.extend(re.findall(pattern, content))

        return [{'name': var, 'type': 'variable'} for var in variables]

    def _analyze_js_patterns(self, content: str) -> Dict[str, Any]:
        return {
            'uses_promises': 'Promise' in content or 'async' in content,
            'uses_async_await': 'async' in content and 'await' in content,
            'uses_dom': 'document.' in content or 'window.' in content,
            'uses_jquery': '$(' in content or 'jQuery' in content,
            'module_type': self._detect_module_type(content)
        }

    def _get_function_type(self, pattern: str) -> str:
        if 'function' in pattern:
            return 'function_declaration'
        elif '=>' in pattern:
            return 'arrow_function'
        else:
            return 'method'

    def _detect_module_type(self, content: str) -> str:
        if 'import' in content or 'export' in content:
            return 'ES6'
        elif 'require' in content or 'module.exports' in content:
            return 'CommonJS'
        else:
            return 'Script'
