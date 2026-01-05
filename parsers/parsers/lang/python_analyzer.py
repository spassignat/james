# python_analyzer.py
import ast
import logging
import re
from typing import Dict, List, Any, Optional
from datetime import datetime

from parsers.analyzer import Analyzer
from parsers.analysis_result import (
    AnalysisResult, AnalysisStatus, FileType, FrameworkType,
    CodeElement, FileMetrics, PatternDetection, DependencyInfo,
    SecurityAnalysis, SectionAnalysis
)

logger = logging.getLogger(__name__)


class PythonAnalyzer(Analyzer):
    """Analyseur de fichiers Python retournant des AnalysisResult"""

    def __init__(self):
        super().__init__()
        self.file_type = FileType.PYTHON

    def analyze_file(self, file_path: str) -> AnalysisResult:
        """Analyse un fichier Python et retourne un AnalysisResult"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Mesurer le temps de traitement
            start_time = datetime.now()

            # Créer le résultat de base
            result = self._create_base_result(file_path)
            result.analyzer_name = "PythonAnalyzer"

            # Analyser le contenu
            analysis = self.analyze_content(content, file_path)

            # Mettre à jour les résultats
            self._update_result_from_analysis(result, analysis)
            result.metrics = self._calculate_python_metrics(content, analysis)

            # Calculer le temps de traitement
            result.processing_time_ms = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )

            return result

        except Exception as e:
            logger.error(f"Error analyzing Python file {file_path}: {e}")
            return self._create_error_result(file_path, str(e))

    def analyze_content(self, content: str, file_path: str) -> Dict[str, Any]:
        """Analyse du contenu Python (méthode interne)"""
        try:
            tree = ast.parse(content)
            return {
                'is_valid': True,
                'tree': tree,
                'imports': self._extract_imports(tree),
                'functions': self._extract_functions(tree),
                'classes': self._extract_classes(tree),
                'variables': self._extract_variables(tree),
                'decorators': self._extract_decorators(tree),
                'exceptions': self._extract_exceptions(tree),
                'type_hints': self._extract_type_hints(tree),
                'async_elements': self._extract_async_elements(tree),
                'analysis': self._analyze_python_patterns(tree, content),
                'raw_content': content
            }
        except SyntaxError as e:
            return {
                'is_valid': False,
                'error': f'Syntax error: {e}',
                'basic_analysis': self._basic_analysis(content),
                'raw_content': content
            }
        except Exception as e:
            return {
                'is_valid': False,
                'error': f'AST parsing error: {e}',
                'basic_analysis': self._basic_analysis(content),
                'raw_content': content
            }

    def _update_result_from_analysis(self, result: AnalysisResult, analysis: Dict[str, Any]) -> None:
        """Met à jour l'AnalysisResult avec les données de l'analyse Python"""

        if not analysis.get('is_valid', False):
            result.status = AnalysisStatus.ERROR
            result.errors.append(f"Python syntax error: {analysis.get('error', 'Unknown error')}")
            # Ajouter une analyse basique quand même
            self._add_basic_elements(result, analysis.get('basic_analysis', {}))
            return

        # Ajouter les éléments de code (classes, fonctions, imports, etc.)
        self._add_python_elements(result, analysis)

        # Mettre à jour les dépendances
        self._update_dependencies(result, analysis)

        # Mettre à jour les patterns
        self._update_patterns(result, analysis)

        # Mettre à jour la sécurité
        self._update_security(result, analysis)

        # Mettre à jour les données spécifiques au langage
        self._update_language_specific(result, analysis)

        # Ajouter des diagnostics
        self._add_diagnostics(result, analysis)

    def _add_python_elements(self, result: AnalysisResult, analysis: Dict[str, Any]) -> None:
        """Convertit les éléments Python en CodeElement"""

        # Classes
        for cls in analysis['classes']:
            class_element = CodeElement(
                name=cls['name'],
                element_type='class',
                line_start=cls.get('lineno'),
                line_end=cls.get('end_lineno'),
                modifiers=cls.get('modifiers', []),
                metadata={
                    'bases': cls.get('bases', []),
                    'method_count': len(cls.get('methods', [])),
                    'decorators': cls.get('decorators', []),
                    'docstring': cls.get('docstring'),
                    'type': 'python_class',
                    'is_dataclass': 'dataclass' in cls.get('decorators', []),
                    'is_abstract': 'abstract' in cls.get('modifiers', []) or 'ABC' in str(cls.get('bases', []))
                }
            )
            result.elements.append(class_element)
            result.metrics.class_count += 1

        # Fonctions
        for func in analysis['functions']:
            function_element = CodeElement(
                name=func['name'],
                element_type='function',
                line_start=func.get('lineno'),
                line_end=func.get('end_lineno'),
                parameters=func.get('args', []),
                return_type=func.get('return_type'),
                modifiers=func.get('modifiers', []),
                metadata={
                    'decorators': func.get('decorators', []),
                    'docstring': func.get('docstring'),
                    'type': 'python_function',
                    'is_async': func.get('is_async', False),
                    'is_generator': func.get('is_generator', False),
                    'is_method': func.get('is_method', False),
                    'is_staticmethod': 'staticmethod' in func.get('decorators', []),
                    'is_classmethod': 'classmethod' in func.get('decorators', []),
                    'has_type_hints': func.get('has_type_hints', False)
                }
            )
            result.elements.append(function_element)
            result.metrics.function_count += 1

        # Variables globales
        for var in analysis['variables']:
            var_element = CodeElement(
                name=var['name'],
                element_type='variable',
                line_start=var.get('lineno'),
                metadata={
                    'value_type': var.get('value_type', 'unknown'),
                    'is_constant': var.get('is_constant', False),
                    'is_exported': var.get('is_exported', False),
                    'type': 'python_variable'
                }
            )
            result.elements.append(var_element)

        # Imports (comme éléments de dépendance)
        for imp in analysis['imports']:
            import_element = CodeElement(
                name=imp.get('module', '') + ('.' + imp.get('name', '') if imp.get('name') else ''),
                element_type='import',
                metadata={
                    'import_type': imp.get('type'),
                    'alias': imp.get('alias'),
                    'module': imp.get('module'),
                    'imported_name': imp.get('name'),
                    'level': imp.get('level', 0),
                    'type': 'python_import'
                }
            )
            result.elements.append(import_element)
            result.metrics.import_count += 1

    def _add_basic_elements(self, result: AnalysisResult, basic_analysis: Dict[str, Any]) -> None:
        """Ajoute des éléments basiques quand l'analyse AST échoue"""
        content = basic_analysis.get('raw_content', '')

        # Détecter les fonctions basiques via regex
        function_matches = re.finditer(r'def\s+(\w+)\s*\(', content)
        for match in function_matches:
            element = CodeElement(
                name=match.group(1),
                element_type='function',
                metadata={'detected_by_regex': True}
            )
            result.elements.append(element)
            result.metrics.function_count += 1

        # Détecter les classes basiques via regex
        class_matches = re.finditer(r'class\s+(\w+)\s*(?:\(|:)', content)
        for match in class_matches:
            element = CodeElement(
                name=match.group(1),
                element_type='class',
                metadata={'detected_by_regex': True}
            )
            result.elements.append(element)
            result.metrics.class_count += 1

    def _update_dependencies(self, result: AnalysisResult, analysis: Dict[str, Any]) -> None:
        """Met à jour les informations de dépendances"""
        deps = result.dependencies

        # Extraire les imports
        imports_list = []
        for imp in analysis.get('imports', []):
            import_info = {
                'module': imp.get('module', ''),
                'name': imp.get('name'),
                'alias': imp.get('alias'),
                'type': imp.get('type')
            }

            # Détecter les dépendances externes vs internes
            module_name = imp.get('module', '')
            if module_name:
                # Détecter les dépendances standards vs tierces
                if module_name.startswith(('builtins', '__future__')):
                    import_info['category'] = 'builtin'
                elif module_name in ['os', 'sys', 'json', 're', 'datetime', 'typing', 'collections', 'itertools']:
                    import_info['category'] = 'standard_library'
                    deps.external_deps.append(module_name.split('.')[0])
                elif any(fw in module_name for fw in ['django', 'flask', 'fastapi', 'numpy', 'pandas', 'requests']):
                    import_info['category'] = 'third_party'
                    deps.external_deps.append(module_name.split('.')[0])
                else:
                    # Probablement interne
                    import_info['category'] = 'internal'
                    deps.internal_deps.append(module_name)

            imports_list.append(import_info)

        deps.imports = imports_list

        # Déterminer le package manager
        if any('requirements' in dep for dep in deps.external_deps):
            deps.package_manager = 'pip'
        elif any('poetry' in dep for dep in deps.external_deps):
            deps.package_manager = 'poetry'
        elif any('pipenv' in dep for dep in deps.external_deps):
            deps.package_manager = 'pipenv'

    def _update_patterns(self, result: AnalysisResult, analysis: Dict[str, Any]) -> None:
        """Met à jour la détection de patterns"""
        patterns = result.patterns
        pattern_analysis = analysis.get('analysis', {})

        # Détecter les frameworks
        framework_indicators = pattern_analysis.get('framework_indicators', [])
        if 'django' in framework_indicators:
            patterns.frameworks.append(FrameworkType.DJANGO)
        if 'flask' in framework_indicators:
            patterns.frameworks.append(FrameworkType.FLASK)
        if 'fastapi' in framework_indicators:
            patterns.frameworks.append(FrameworkType.VANILLA)  # FastAPI n'a pas d'enum spécifique

        # Détecter les patterns de code
        if pattern_analysis.get('uses_type_hints'):
            patterns.patterns.append('type_hints')
            patterns.libraries.append('typing')

        if pattern_analysis.get('uses_async'):
            patterns.patterns.append('async_await')

        if pattern_analysis.get('uses_decorators'):
            patterns.patterns.append('decorators')

        # Architecture hints
        if len(analysis.get('classes', [])) > 0:
            patterns.architecture_hints.append('oop_design')

        if any('test' in func['name'].lower() for func in analysis.get('functions', [])):
            patterns.architecture_hints.append('test_file')

        if any('api' in func['name'].lower() or 'route' in func['name'].lower() for func in analysis.get('functions', [])):
            patterns.architecture_hints.append('web_api')

        # Détecter les bibliothèques via les imports
        imports = analysis.get('imports', [])
        for imp in imports:
            module = imp.get('module', '').lower()
            if 'numpy' in module:
                patterns.libraries.append('numpy')
            elif 'pandas' in module:
                patterns.libraries.append('pandas')
            elif 'requests' in module:
                patterns.libraries.append('requests')
            elif 'sqlalchemy' in module:
                patterns.libraries.append('sqlalchemy')
            elif 'pytest' in module:
                patterns.libraries.append('pytest')
            elif 'unittest' in module:
                patterns.libraries.append('unittest')

    def _update_security(self, result: AnalysisResult, analysis: Dict[str, Any]) -> None:
        """Met à jour l'analyse de sécurité"""
        security = result.security
        content = analysis.get('raw_content', '')

        # Vérifier les vulnérabilités courantes
        security_issues = self._detect_security_issues(content)

        # Évaluer les commandes système
        if security_issues.get('has_system_commands'):
            security.warnings.append("Use of system/os commands detected")
            security.vulnerabilities.append({
                'type': 'command_injection',
                'severity': 'high',
                'description': 'Use of os.system, subprocess, or similar may lead to command injection',
                'recommendation': 'Use safer alternatives like shlex.quote() or built-in libraries'
            })

        # Évaluer les eval/exec
        if security_issues.get('has_dynamic_code'):
            security.warnings.append("Use of eval/exec detected")
            security.vulnerabilities.append({
                'type': 'code_injection',
                'severity': 'critical',
                'description': 'Use of eval() or exec() may lead to code injection',
                'recommendation': 'Avoid eval() and exec() with user input'
            })

        # Évaluer la désérialisation pickle
        if security_issues.get('has_pickle'):
            security.warnings.append("Use of pickle detected")
            security.vulnerabilities.append({
                'type': 'deserialization',
                'severity': 'high',
                'description': 'Pickle deserialization may lead to remote code execution',
                'recommendation': 'Use safer serialization like json'
            })

        # Recommandations générales
        if 'import hashlib' in content or 'import bcrypt' in content:
            security.recommendations.append("Good: Using proper hashing libraries")
        else:
            if 'password' in content.lower() or 'secret' in content.lower():
                security.recommendations.append("Consider using hashlib or bcrypt for password hashing")

        # Calculer le score de sécurité
        security.security_score = self._calculate_security_score(analysis)

    def _update_language_specific(self, result: AnalysisResult, analysis: Dict[str, Any]) -> None:
        """Met à jour les données spécifiques au Python"""
        result.language_specific = {
            'python': {
                'is_valid': analysis.get('is_valid', False),
                'python_version': self._detect_python_version(analysis),
                'has_type_hints': analysis.get('analysis', {}).get('uses_type_hints', False),
                'has_async_code': analysis.get('analysis', {}).get('uses_async', False),
                'code_style': analysis.get('analysis', {}).get('code_style', 'unknown'),
                'import_count': len(analysis.get('imports', [])),
                'function_count': len(analysis.get('functions', [])),
                'class_count': len(analysis.get('classes', [])),
                'decorator_count': len(analysis.get('decorators', [])),
                'type_hint_count': len(analysis.get('type_hints', [])),
                'async_element_count': len(analysis.get('async_elements', [])),
                'is_script': self._is_script_file(analysis),
                'is_module': self._is_module_file(analysis),
                'is_test_file': self._is_test_file(analysis)
            }
        }

    def _add_diagnostics(self, result: AnalysisResult, analysis: Dict[str, Any]) -> None:
        """Ajoute des diagnostics et recommandations"""

        if not analysis.get('is_valid', False):
            result.notes.append("File contains syntax errors - analysis limited")
            return

        # Vérifier les imports wildcard
        imports = analysis.get('imports', [])
        wildcard_imports = [imp for imp in imports if imp.get('name') == '*']
        if wildcard_imports:
            result.warnings.append(f"Wildcard imports found: {len(wildcard_imports)}")

        # Vérifier les fonctions longues
        functions = analysis.get('functions', [])
        long_functions = [f for f in functions if f.get('line_count', 0) > 50]
        if long_functions:
            result.notes.append(f"Long functions detected: {len(long_functions)}")

        # Vérifier les classes complexes
        classes = analysis.get('classes', [])
        complex_classes = [c for c in classes if len(c.get('methods', [])) > 10]
        if complex_classes:
            result.notes.append(f"Complex classes detected: {len(complex_classes)}")

        # Recommandations de style
        if not analysis.get('analysis', {}).get('uses_type_hints'):
            result.recommendations.append("Consider adding type hints for better code clarity")

        # Vérifier les noms de variables
        variables = analysis.get('variables', [])
        bad_names = [v for v in variables if len(v.get('name', '')) == 1 or v.get('name', '').startswith('_')]
        if len(bad_names) > 5:
            result.notes.append("Many single-letter or underscore-prefixed variable names")

    def _extract_imports(self, tree: ast.AST) -> List[Dict]:
        """Extrait les imports avec plus de détails"""
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append({
                        'type': 'import',
                        'module': alias.name,
                        'alias': alias.asname,
                        'lineno': node.lineno,
                        'end_lineno': getattr(node, 'end_lineno', None),
                        'col_offset': node.col_offset
                    })
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                level = node.level
                for alias in node.names:
                    imports.append({
                        'type': 'from_import',
                        'module': module,
                        'name': alias.name,
                        'alias': alias.asname,
                        'level': level,
                        'lineno': node.lineno,
                        'end_lineno': getattr(node, 'end_lineno', None),
                        'col_offset': node.col_offset,
                        'is_relative': level > 0
                    })

        return imports

    def _extract_functions(self, tree: ast.AST) -> List[Dict]:
        """Extrait les fonctions avec plus de détails"""
        functions = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Arguments
                args = []
                if node.args.args:
                    args = [arg.arg for arg in node.args.args]

                # Décorateurs
                decorators = []
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Name):
                        decorators.append(decorator.id)
                    elif isinstance(decorator, ast.Attribute):
                        decorators.append(self._get_attribute_name(decorator))
                    elif isinstance(decorator, ast.Call):
                        if isinstance(decorator.func, ast.Name):
                            decorators.append(decorator.func.id)

                # Docstring
                docstring = ast.get_docstring(node)

                # Type hints
                return_annotation = None
                if node.returns:
                    return_annotation = ast.unparse(node.returns) if hasattr(ast, 'unparse') else str(node.returns)

                # Arguments avec annotations
                args_with_types = []
                for arg in node.args.args:
                    arg_info = {'name': arg.arg}
                    if arg.annotation:
                        arg_info['type'] = ast.unparse(arg.annotation) if hasattr(ast, 'unparse') else str(arg.annotation)
                    args_with_types.append(arg_info)

                functions.append({
                    'name': node.name,
                    'args': args,
                    'args_with_types': args_with_types,
                    'decorators': decorators,
                    'lineno': node.lineno,
                    'end_lineno': getattr(node, 'end_lineno', None),
                    'docstring': docstring,
                    'return_type': return_annotation,
                    'is_async': isinstance(node, ast.AsyncFunctionDef),
                    'is_generator': any(isinstance(n, ast.Yield) or isinstance(n, ast.YieldFrom) for n in ast.walk(node)),
                    'has_type_hints': bool(return_annotation) or any('type' in arg for arg in args_with_types),
                    'line_count': self._count_function_lines(node)
                })

        return functions

    def _extract_classes(self, tree: ast.AST) -> List[Dict]:
        """Extrait les classes avec plus de détails"""
        classes = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Bases
                bases = []
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        bases.append(base.id)
                    elif hasattr(ast, 'unparse'):
                        bases.append(ast.unparse(base))

                # Méthodes
                methods = []
                for n in node.body:
                    if isinstance(n, ast.FunctionDef) or isinstance(n, ast.AsyncFunctionDef):
                        methods.append(n.name)

                # Décorateurs
                decorators = []
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Name):
                        decorators.append(decorator.id)
                    elif isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name):
                        decorators.append(decorator.func.id)

                # Docstring
                docstring = ast.get_docstring(node)

                # Modificateurs
                modifiers = []
                if any('abstract' in d for d in decorators):
                    modifiers.append('abstract')
                if 'dataclass' in decorators:
                    modifiers.append('dataclass')

                classes.append({
                    'name': node.name,
                    'bases': bases,
                    'methods': methods,
                    'method_count': len(methods),
                    'decorators': decorators,
                    'lineno': node.lineno,
                    'end_lineno': getattr(node, 'end_lineno', None),
                    'docstring': docstring,
                    'modifiers': modifiers
                })

        return classes

    def _extract_variables(self, tree: ast.AST) -> List[Dict]:
        """Extrait les variables globales"""
        variables = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        # Essayer de déterminer le type
                        value_type = 'unknown'
                        if isinstance(node.value, ast.Constant):
                            value_type = type(node.value.value).__name__
                        elif isinstance(node.value, ast.List):
                            value_type = 'list'
                        elif isinstance(node.value, ast.Dict):
                            value_type = 'dict'
                        elif isinstance(node.value, ast.Set):
                            value_type = 'set'
                        elif isinstance(node.value, ast.Tuple):
                            value_type = 'tuple'

                        variables.append({
                            'name': target.id,
                            'lineno': node.lineno,
                            'value_type': value_type,
                            'is_constant': target.id.isupper(),  # Convention Python
                            'is_exported': not target.id.startswith('_')  # Pas privé
                        })

        return variables

    def _extract_decorators(self, tree: ast.AST) -> List[Dict]:
        """Extrait tous les décorateurs"""
        decorators = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) or isinstance(node, ast.ClassDef):
                for decorator in node.decorator_list:
                    decorator_info = {'lineno': node.lineno}

                    if isinstance(decorator, ast.Name):
                        decorator_info['name'] = decorator.id
                        decorator_info['type'] = 'simple'
                    elif isinstance(decorator, ast.Call):
                        if isinstance(decorator.func, ast.Name):
                            decorator_info['name'] = decorator.func.id
                            decorator_info['type'] = 'call'
                            decorator_info['has_args'] = len(decorator.args) > 0 or len(decorator.keywords) > 0

                    if decorator_info.get('name'):
                        decorators.append(decorator_info)

        return decorators

    def _extract_exceptions(self, tree: ast.AST) -> List[Dict]:
        """Extrait les blocs d'exception"""
        exceptions = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                for handler in node.handlers:
                    if handler.type:
                        type_name = ast.unparse(handler.type) if hasattr(ast, 'unparse') else str(handler.type)
                        exceptions.append({
                            'type': type_name,
                            'lineno': handler.lineno,
                            'has_else': bool(node.orelse),
                            'has_finally': bool(node.finalbody)
                        })

        return exceptions

    def _extract_type_hints(self, tree: ast.AST) -> List[Dict]:
        """Extrait les annotations de type"""
        type_hints = []

        for node in ast.walk(tree):
            # Annotations de fonction
            if isinstance(node, ast.FunctionDef):
                if node.returns:
                    type_hints.append({
                        'context': 'function_return',
                        'function': node.name,
                        'type': ast.unparse(node.returns) if hasattr(ast, 'unparse') else str(node.returns),
                        'lineno': node.lineno
                    })

                for arg in node.args.args:
                    if arg.annotation:
                        type_hints.append({
                            'context': 'function_arg',
                            'function': node.name,
                            'arg': arg.arg,
                            'type': ast.unparse(arg.annotation) if hasattr(ast, 'unparse') else str(arg.annotation),
                            'lineno': arg.lineno if hasattr(arg, 'lineno') else node.lineno
                        })

            # Annotations de variable
            elif isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name) and node.annotation:
                    type_hints.append({
                        'context': 'variable',
                        'variable': node.target.id,
                        'type': ast.unparse(node.annotation) if hasattr(ast, 'unparse') else str(node.annotation),
                        'lineno': node.lineno
                    })

        return type_hints

    def _extract_async_elements(self, tree: ast.AST) -> List[Dict]:
        """Extrait les éléments async"""
        async_elements = []

        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef):
                async_elements.append({
                    'type': 'async_function',
                    'name': node.name,
                    'lineno': node.lineno
                })
            elif isinstance(node, ast.AsyncFor):
                async_elements.append({
                    'type': 'async_for',
                    'lineno': node.lineno
                })
            elif isinstance(node, ast.AsyncWith):
                async_elements.append({
                    'type': 'async_with',
                    'lineno': node.lineno
                })

        return async_elements

    def _analyze_python_patterns(self, tree: ast.AST, content: str) -> Dict[str, Any]:
        """Analyse les patterns Python"""
        return {
            'uses_type_hints': self._has_type_hints(tree),
            'uses_async': self._has_async_elements(tree),
            'uses_decorators': self._has_decorators(tree),
            'uses_comprehensions': self._has_comprehensions(tree),
            'uses_context_managers': self._has_context_managers(tree),
            'code_style': self._detect_code_style(content),
            'framework_indicators': self._detect_framework(content),
            'complexity_metrics': self._calculate_ast_complexity(tree)
        }

    def _has_type_hints(self, tree: ast.AST) -> bool:
        """Vérifie si le code utilise des annotations de type"""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.returns:
                return True
            if isinstance(node, ast.FunctionDef):
                for arg in node.args.args:
                    if arg.annotation:
                        return True
            if isinstance(node, ast.AnnAssign):
                return True
        return False

    def _has_async_elements(self, tree: ast.AST) -> bool:
        """Vérifie si le code utilise async/await"""
        for node in ast.walk(tree):
            if isinstance(node, (ast.AsyncFunctionDef, ast.AsyncFor, ast.AsyncWith, ast.Await)):
                return True
        return False

    def _has_decorators(self, tree: ast.AST) -> bool:
        """Vérifie si le code utilise des décorateurs"""
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.decorator_list:
                return True
        return False

    def _has_comprehensions(self, tree: ast.AST) -> bool:
        """Vérifie si le code utilise des compréhensions"""
        for node in ast.walk(tree):
            if isinstance(node, (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)):
                return True
        return False

    def _has_context_managers(self, tree: ast.AST) -> bool:
        """Vérifie si le code utilise des gestionnaires de contexte"""
        for node in ast.walk(tree):
            if isinstance(node, ast.With):
                return True
        return False

    def _detect_code_style(self, content: str) -> str:
        """Détecte le style de code"""
        lines = content.split('\n')

        # Vérifier le style de nommage
        variable_pattern = re.compile(r'[a-z_][a-z0-9_]*\s*=')
        class_pattern = re.compile(r'class\s+([A-Z][a-zA-Z0-9]*)')
        constant_pattern = re.compile(r'([A-Z_][A-Z0-9_]*)\s*=')

        variable_matches = len(variable_pattern.findall(content))
        class_matches = len(class_pattern.findall(content))
        constant_matches = len(constant_pattern.findall(content))

        if variable_matches > 0 and class_matches > 0:
            return 'mixed'
        elif variable_matches > constant_matches:
            return 'snake_case'
        else:
            return 'unknown'

    def _detect_framework(self, content: str) -> List[str]:
        """Détecte les frameworks utilisés"""
        frameworks = []
        content_lower = content.lower()

        framework_patterns = {
            'django': ['django', 'models.Model', 'from django'],
            'flask': ['flask', 'Flask(', '@app.route'],
            'fastapi': ['fastapi', 'FastAPI(', '@app.get', '@app.post'],
            'numpy': ['numpy', 'np.', 'import numpy'],
            'pandas': ['pandas', 'pd.', 'import pandas'],
            'pytest': ['pytest', '@pytest', 'import pytest'],
            'sqlalchemy': ['sqlalchemy', 'Column(', 'declarative_base'],
            'celery': ['celery', '@celery.task', 'Celery('],
            'asyncio': ['asyncio', 'async def', 'await ']
        }

        for framework, patterns in framework_patterns.items():
            if any(pattern.lower() in content_lower for pattern in patterns):
                frameworks.append(framework)

        return frameworks

    def _calculate_ast_complexity(self, tree: ast.AST) -> Dict[str, int]:
        """Calcule des métriques de complexité AST"""
        metrics = {
            'function_count': 0,
            'class_count': 0,
            'import_count': 0,
            'if_count': 0,
            'for_count': 0,
            'while_count': 0,
            'try_count': 0,
            'with_count': 0
        }

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                metrics['function_count'] += 1
            elif isinstance(node, ast.ClassDef):
                metrics['class_count'] += 1
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                metrics['import_count'] += 1
            elif isinstance(node, ast.If):
                metrics['if_count'] += 1
            elif isinstance(node, ast.For):
                metrics['for_count'] += 1
            elif isinstance(node, ast.While):
                metrics['while_count'] += 1
            elif isinstance(node, ast.Try):
                metrics['try_count'] += 1
            elif isinstance(node, ast.With):
                metrics['with_count'] += 1

        return metrics

    def _detect_security_issues(self, content: str) -> Dict[str, bool]:
        """Détecte les problèmes de sécurité"""
        return {
            'has_system_commands': bool(re.search(r'\b(os\.system|subprocess\.(call|run|Popen)|exec\(|eval\()', content)),
            'has_dynamic_code': bool(re.search(r'\b(eval\(|exec\(|compile\()', content)),
            'has_pickle': bool(re.search(r'\b(pickle\.(load|loads)|cPickle)', content)),
            'has_weak_crypto': bool(re.search(r'\b(md5\(|sha1\(|base64\.b64decode)', content)),
            'has_hardcoded_secrets': bool(re.search(r'(password|secret|key)\s*=\s*[\'"][^\'"]+[\'"]', content, re.IGNORECASE))
        }

    def _basic_analysis(self, content: str) -> Dict[str, Any]:
        """Analyse basique quand AST échoue"""
        return {
            'line_count': len(content.splitlines()),
            'has_functions': bool(re.search(r'def\s+\w+\s*\(', content)),
            'has_classes': bool(re.search(r'class\s+\w+\s*[\(:]', content)),
            'import_count': len(re.findall(r'import\s+\w+|from\s+\w+\s+import', content)),
            'has_comments': '#' in content,
            'has_docstrings': '"""' in content or "'''" in content,
            'raw_content': content
        }

    def _count_function_lines(self, node: ast.FunctionDef) -> int:
        """Compte le nombre de lignes d'une fonction"""
        start_line = node.lineno
        end_line = getattr(node, 'end_lineno', start_line)
        return end_line - start_line + 1

    def _get_attribute_name(self, node: ast.Attribute) -> str:
        """Récupère le nom complet d'un attribut"""
        parts = []
        current = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        return '.'.join(reversed(parts))

    def _calculate_python_metrics(self, content: str, analysis: Dict[str, Any]) -> FileMetrics:
        """Calcule les métriques spécifiques au Python"""
        metrics = super()._calculate_metrics(content)

        if not analysis.get('is_valid', False):
            return metrics

        # Mettre à jour avec les métriques spécifiques au Python
        metrics.function_count = len(analysis.get('functions', []))
        metrics.class_count = len(analysis.get('classes', []))
        metrics.import_count = len(analysis.get('imports', []))

        # Compter les exports (fonctions/classes publiques)
        functions = analysis.get('functions', [])
        classes = analysis.get('classes', [])
        public_functions = [f for f in functions if not f['name'].startswith('_')]
        public_classes = [c for c in classes if not c['name'].startswith('_')]
        metrics.export_count = len(public_functions) + len(public_classes)

        # Calculer la complexité cyclomatique simplifiée
        ast_complexity = analysis.get('analysis', {}).get('complexity_metrics', {})
        complexity_score = (
                ast_complexity.get('if_count', 0) +
                ast_complexity.get('for_count', 0) +
                ast_complexity.get('while_count', 0) +
                ast_complexity.get('try_count', 0)
        )
        metrics.complexity_score = complexity_score / max(1, metrics.function_count)

        # Compter les lignes de code (exclure les commentaires et docstrings)
        lines = content.split('\n')
        code_lines = 0
        in_multiline_string = False

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith('#'):
                continue
            if stripped.startswith('"""') or stripped.startswith("'''"):
                in_multiline_string = not in_multiline_string
                continue
            if in_multiline_string:
                continue
            code_lines += 1

        metrics.code_lines = code_lines

        return metrics

    def _calculate_security_score(self, analysis: Dict[str, Any]) -> float:
        """Calcule un score de sécurité pour Python"""
        score = 100.0
        content = analysis.get('raw_content', '')

        # Détecter les problèmes de sécurité
        security_issues = self._detect_security_issues(content)

        # Pénalités
        if security_issues.get('has_dynamic_code'):
            score -= 40  # eval/exec très dangereux

        if security_issues.get('has_system_commands'):
            score -= 30  # Command injection

        if security_issues.get('has_pickle'):
            score -= 25  # Deserialization RCE

        if security_issues.get('has_weak_crypto'):
            score -= 20  # Crypto faible

        if security_issues.get('has_hardcoded_secrets'):
            score -= 15  # Secrets en dur

        # Bonus pour les bonnes pratiques
        if 'import hashlib' in content:
            score += 10

        if 'from typing import' in content:
            score += 5

        if 'import logging' in content:
            score += 5

        return max(0, min(100, score))

    def _detect_python_version(self, analysis: Dict[str, Any]) -> str:
        """Détecte la version de Python ciblée"""
        content = analysis.get('raw_content', '')

        # Vérifier les imports __future__
        if 'from __future__ import' in content:
            if 'annotations' in content:
                return '3.7+'
            elif 'generator_stop' in content:
                return '3.7+'

        # Vérifier les fonctionnalités spécifiques
        if 'async def' in content:
            return '3.5+'

        if 'f"' in content or 'f\'' in content:
            return '3.6+'

        if 'match ' in content and 'case ' in content:
            return '3.10+'

        return 'unknown'

    def _is_script_file(self, analysis: Dict[str, Any]) -> bool:
        """Détermine si c'est un script exécutable"""
        content = analysis.get('raw_content', '')
        return 'if __name__ == "__main__"' in content

    def _is_module_file(self, analysis: Dict[str, Any]) -> bool:
        """Détermine si c'est un module importable"""
        functions = analysis.get('functions', [])
        classes = analysis.get('classes', [])
        return len(functions) > 0 or len(classes) > 0

    def _is_test_file(self, analysis: Dict[str, Any]) -> bool:
        """Détermine si c'est un fichier de test"""
        file_path = analysis.get('file_path', '').lower()
        functions = analysis.get('functions', [])

        # Vérifier le nom du fichier
        if any(test_pattern in file_path for test_pattern in ['test_', '_test.py', 'tests/']):
            return True

        # Vérifier le nom des fonctions
        if any(func['name'].startswith('test_') for func in functions):
            return True

        # Vérifier les imports de test
        imports = analysis.get('imports', [])
        test_imports = ['pytest', 'unittest', 'nose']
        if any(any(test in imp.get('module', '') for test in test_imports) for imp in imports):
            return True

        return False