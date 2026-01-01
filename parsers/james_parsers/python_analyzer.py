# python_analyzer.py
import ast
import logging
import re
from typing import Dict, List, Any

from james_parsers.analyzer import Analyzer

logger = logging.getLogger(__name__)


class PythonAnalyzer(Analyzer):
    def analyze(self, file_path: str) -> Dict[str, Any]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return self.analyze_content(content, file_path)
        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {e}")
            return {
                'file_type': 'python',
                'file_path': file_path,
                'is_valid': False,
                'error': str(e)
            }

    def analyze_content(self, content: str, file_path: str) -> Dict[str, Any]:
        try:
            tree = ast.parse(content)
            return {
                'file_type': 'python',
                'imports': self._extract_imports(tree),
                'functions': self._extract_functions(tree),
                'classes': self._extract_classes(tree),
                'analysis': self._analyze_python_patterns(tree, content)
            }
        except SyntaxError as e:
            return {
                'file_type': 'python',
                'error': f'Syntax error: {e}',
                'basic_analysis': self._basic_analysis(content)
            }

    def _extract_imports(self, tree: ast.AST) -> List[Dict]:
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append({
                        'type': 'import',
                        'module': alias.name,
                        'alias': alias.asname
                    })
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    imports.append({
                        'type': 'from_import',
                        'module': module,
                        'name': alias.name,
                        'alias': alias.asname
                    })

        return imports

    def _extract_functions(self, tree: ast.AST) -> List[Dict]:
        functions = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append({
                    'name': node.name,
                    'args': [arg.arg for arg in node.args.args],
                    'decorators': [d.id for d in node.decorator_list if isinstance(d, ast.Name)],
                    'lineno': node.lineno
                })

        return functions

    def _extract_classes(self, tree: ast.AST) -> List[Dict]:
        classes = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append({
                    'name': node.name,
                    'bases': [base.id for base in node.bases if isinstance(base, ast.Name)],
                    'methods': [n.name for n in node.body if isinstance(n, ast.FunctionDef)],
                    'decorators': [d.id for d in node.decorator_list if isinstance(d, ast.Name)],
                    'lineno': node.lineno
                })

        return classes

    def _analyze_python_patterns(self, tree: ast.AST, content: str) -> Dict[str, Any]:
        return {
            'uses_type_hints': ':' in content and '->' in content,
            'uses_async': 'async' in content,
            'uses_decorators': '@' in content,
            'code_style': self._detect_code_style(content),
            'framework_indicators': self._detect_framework(content)
        }

    def _detect_code_style(self, content: str) -> str:
        if 'snake_case' in content or '_' in content:
            return 'snake_case'
        return 'unknown'

    def _detect_framework(self, content: str) -> List[str]:
        frameworks = []
        if 'import flask' in content or 'from flask' in content:
            frameworks.append('flask')
        if 'import django' in content or 'from django' in content:
            frameworks.append('django')
        if 'import fastapi' in content or 'from fastapi' in content:
            frameworks.append('fastapi')
        return frameworks

    def _basic_analysis(self, content: str) -> Dict[str, Any]:
        return {
            'line_count': len(content.splitlines()),
            'has_functions': 'def ' in content,
            'has_classes': 'class ' in content,
            'import_count': len(re.findall(r'import\s+\w+|from\s+\w+\s+import', content))
        }
