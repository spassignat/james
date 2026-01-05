# parsers/javascript_analyzer.py
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional

import time

from parsers.analysis_result import AnalysisResult, AnalysisStatus, FileType, DependencyInfo, CodeElement, \
    FrameworkType, \
    PatternDetection, SecurityAnalysis, CodeMetrics, ComplexityLevel
from parsers.analyzer import Analyzer

logger = logging.getLogger(__name__)


class JavaScriptAnalyzer(Analyzer):
    def __init__(self):
        super().__init__()
        self.file_type = FileType.JAVASCRIPT
        self.supported_extensions = ['.js', '.jsx', '.ts', '.tsx', '.mjs', '.cjs']

        # Patterns regex pour l'analyse
        self.function_pattern = re.compile(
            r'(async\s+)?(function\s+(\w+)\s*\(([^)]*)\)|'
            r'const\s+(\w+)\s*=\s*(async\s+)?\(([^)]*)\)\s*=>|'
            r'let\s+(\w+)\s*=\s*(async\s+)?\(([^)]*)\)\s*=>|'
            r'var\s+(\w+)\s*=\s*(async\s+)?\(([^)]*)\)\s*=>|'
            r'(\w+)\s*\(([^)]*)\)\s*{)',
            re.MULTILINE
        )

        self.class_pattern = re.compile(
            r'class\s+(\w+)(?:\s+extends\s+(\w+))?\s*{',
            re.MULTILINE
        )

        self.import_pattern = re.compile(
            r'import\s+(?:{[^}]*}|\*\s+as\s+\w+|\w+)\s+from\s+[\'"]([^\'"]+)[\'"]|'
            r'require\s*\([\'"]([^\'"]+)[\'"]\)',
            re.MULTILINE
        )

        self.export_pattern = re.compile(
            r'export\s+(?:default\s+)?(?:class\s+(\w+)|function\s+(\w+)|const\s+(\w+)|let\s+(\w+)|var\s+(\w+)|{[^}]*})',
            re.MULTILINE
        )

        self.jsx_pattern = re.compile(r'return\s*\(.*?<[A-Z]', re.DOTALL)
        self.react_hook_pattern = re.compile(r'use[A-Z][a-zA-Z]*\s*\(')
        self.vue_pattern = re.compile(r'Vue\.|new\s+Vue\(')
        self.eval_pattern = re.compile(r'eval\s*\(')
        self.inner_html_pattern = re.compile(r'\.innerHTML\s*=')

    def analyze_file(self, file_path: str) -> AnalysisResult:
        """Analyse un fichier JavaScript/TypeScript"""
        start_time = time.time() * 1000

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            result = self._analyze_file_content(content, file_path)
            result.processing_time_ms = int(time.time() * 1000 - start_time)
            result.status = AnalysisStatus.SUCCESS
            return result

        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    content = f.read()
                result = self._analyze_file_content(content, file_path)
                result.processing_time_ms = int(time.time() * 1000 - start_time)
                result.status = AnalysisStatus.SUCCESS
                result.warnings.append("Fichier encodé en latin-1, conversion UTF-8 effectuée")
                return result
            except Exception as e:
                logger.error(f"Erreur d'encodage {file_path}: {e}")
                return self._create_error_result(file_path, f"Erreur d'encodage: {str(e)}")

        except Exception as e:
            logger.error(f"Erreur analyse {file_path}: {e}")
            result = self._create_error_result(file_path, str(e))
            result.processing_time_ms = int(time.time() * 1000 - start_time)
            return result

    def _analyze_file_content(self, content: str, file_path: str) -> AnalysisResult:
        """Analyse le contenu d'un fichier"""
        result = self._create_base_result(file_path)

        # Analyse du contenu
        raw_analysis = self.analyze_content(content, file_path)

        # Remplir les champs du résultat
        result.metrics = self._calculate_metrics(content)
        result.elements = self._convert_to_code_elements(raw_analysis)
        result.dependencies = self._extract_dependencies(raw_analysis)
        result.patterns = self._detect_patterns(raw_analysis)
        result.security = self._analyze_security(content)
        result.language_specific = raw_analysis.get('analysis', {})

        # Mettre à jour les métriques spécifiques
        result.metrics.function_count = len([e for e in result.elements if e.element_type == 'function'])
        result.metrics.class_count = len([e for e in result.elements if e.element_type == 'class'])
        result.metrics.import_count = len(result.dependencies.imports)
        result.metrics.export_count = len(result.dependencies.exports)
        result.metrics.complexity_score = raw_analysis.get('metrics', {}).get('complexity_score', 0)

        # Déterminer le niveau de complexité
        if result.metrics.complexity_score > 30:
            result.metrics.complexity_level = ComplexityLevel.HIGH
        elif result.metrics.complexity_score > 15:
            result.metrics.complexity_level = ComplexityLevel.MEDIUM
        else:
            result.metrics.complexity_level = ComplexityLevel.LOW

        return result

    def _create_base_result(self, file_path: str) -> AnalysisResult:
        """Crée un résultat d'analyse de base"""
        return AnalysisResult(
            file_path=file_path,
            filename=file_path.split('/')[-1],
            file_type=self.file_type,
            file_size=os.path.getsize(file_path),
            status=AnalysisStatus.PENDING,
            language_specific={},
            processing_time_ms=0,
            naming_analysis=self._get_naming_analysis(file_path),
            last_modified=os.path.getmtime(file_path)
        )

    def _create_error_result(self, file_path: str, error_message: str) -> AnalysisResult:
        """Crée un résultat d'analyse en erreur"""
        result = self._create_base_result(file_path)
        result.status = AnalysisStatus.ERROR
        result.errors.append(error_message)
        return result

    def analyze_content(self, content: str, file_path: str) -> Dict[str, Any]:
        """Analyse le contenu JavaScript/TypeScript"""
        try:
            lines = content.split('\n')

            analysis = {
                'functions': self._extract_functions(content, lines),
                'classes': self._extract_classes(content, lines),
                'imports': self._extract_imports(content, file_path),
                'exports': self._extract_exports(content),
                'analysis': self._analyze_js_patterns(content),
                'metrics': self._calculate_detailed_metrics(content, lines)
            }

            return analysis

        except Exception as e:
            logger.error(f"Erreur analyse contenu {file_path}: {e}")
            return {
                'functions': [],
                'classes': [],
                'imports': [],
                'exports': [],
                'analysis': {'error': str(e)},
                'metrics': {}
            }

    def _extract_functions(self, content: str, lines: List[str]) -> List[Dict[str, Any]]:
        """Extrait les fonctions du code"""
        functions = []

        # Pattern simplifié pour les fonctions
        function_patterns = [
            # Fonction déclaration
            (r'function\s+(\w+)\s*\(([^)]*)\)', 'function'),
            # Fonction async
            (r'async\s+function\s+(\w+)\s*\(([^)]*)\)', 'async_function'),
            # Fonction fléchée avec const/let/var
            (r'(?:const|let|var)\s+(\w+)\s*=\s*(async\s+)?\(([^)]*)\)\s*=>', 'arrow_function'),
            # Méthode de classe
            (r'(\w+)\s*\(([^)]*)\)\s*{', 'method'),
            # Méthode async de classe
            (r'async\s+(\w+)\s*\(([^)]*)\)\s*{', 'async_method'),
        ]

        for pattern, func_type in function_patterns:
            for match in re.finditer(pattern, content, re.MULTILINE):
                line_number = content[:match.start()].count('\n') + 1
                groups = match.groups()

                if len(groups) >= 1:
                    func_name = groups[0]
                    parameters = []

                    if len(groups) >= 2 and groups[1]:
                        # Le deuxième groupe peut être 'async' ou les paramètres
                        if 'async' in groups[1]:
                            is_async = True
                            if len(groups) >= 3 and groups[2]:
                                parameters = [p.strip() for p in groups[2].split(',')]
                        else:
                            is_async = False
                            parameters = [p.strip() for p in groups[1].split(',')] if groups[1] else []
                    else:
                        is_async = func_type.startswith('async')

                    functions.append({
                        'name': func_name,
                        'parameters': parameters,
                        'line_number': line_number,
                        'is_async': is_async,
                        'type': func_type
                    })

        return functions

    def _parse_function_match(self, match: re.Match, lines: List[str]) -> Optional[Dict[str, Any]]:
        """Parse un match de fonction"""
        groups = match.groups()

        if not any(groups):
            return None

        # Déterminer le nom de la fonction
        func_name = None
        parameters = []
        is_async = bool(groups[0])  # Le premier groupe est 'async'

        # Déterminer quel groupe contient le nom
        for i, group in enumerate(groups[1:], 1):
            if group and i in [2, 4, 7, 10, 13]:  # Indices des noms de fonction
                func_name = group
            elif group and i in [3, 6, 9, 12, 14]:  # Indices des paramètres
                parameters = [p.strip() for p in group.split(',')] if group else []

        if not func_name:
            return None

        line_number = len(re.findall(r'\n', content[:match.start()])) + 1 if 'content' in locals() else 1

        return {
            'name': func_name,
            'parameters': parameters,
            'line_number': line_number,
            'is_async': is_async,
            'type': 'function'
        }

    def _extract_classes(self, content: str, lines: List[str]) -> List[Dict[str, Any]]:
        """Extrait les classes du code"""
        classes = []

        for match in self.class_pattern.finditer(content):
            class_name = match.group(1)
            parent_class = match.group(2)
            line_number = content[:match.start()].count('\n') + 1

            # Extraire les méthodes de la classe
            class_content = self._extract_class_content(content[match.start():])
            methods = self._extract_class_methods(class_content)

            classes.append({
                'name': class_name,
                'parent': parent_class,
                'line_number': line_number,
                'methods': methods
            })

        return classes

    def _extract_class_content(self, content: str) -> str:
        """Extrait le contenu d'une classe"""
        brace_count = 0
        end_pos = 0

        for i, char in enumerate(content):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_pos = i
                    break

        return content[:end_pos + 1] if end_pos > 0 else content

    def _extract_class_methods(self, class_content: str) -> List[Dict[str, Any]]:
        """Extrait les méthodes d'une classe"""
        methods = []

        # Patterns pour les méthodes de classe
        method_patterns = [
            r'(\w+)\s*\(([^)]*)\)\s*{',  # Méthode normale
            r'get\s+(\w+)\s*\(\)\s*{',  # Getter
            r'set\s+(\w+)\s*\(([^)]*)\)\s*{',  # Setter
            r'static\s+(\w+)\s*\(([^)]*)\)\s*{',  # Méthode statique
            r'async\s+(\w+)\s*\(([^)]*)\)\s*{',  # Méthode async
        ]

        for pattern in method_patterns:
            for match in re.finditer(pattern, class_content, re.MULTILINE):
                method_name = match.group(1)
                parameters = match.group(2) if match.lastindex > 1 else ''

                methods.append({
                    'name': method_name,
                    'parameters': [p.strip() for p in parameters.split(',')] if parameters else [],
                    'type': 'method'
                })

        return methods

    def _extract_imports(self, content: str, file_path: str) -> List[Dict[str, Any]]:
        """Extrait les imports du code"""
        imports = []

        for match in self.import_pattern.finditer(content):
            module_path = match.group(1) or match.group(2)
            if not module_path:
                continue

            line_number = content[:match.start()].count('\n') + 1

            # Déterminer si c'est une dépendance relative ou externe
            is_relative = module_path.startswith('.') or module_path.startswith('/')

            imports.append({
                'module': module_path,
                'line_number': line_number,
                'is_relative': is_relative,
                'type': 'import' if match.group(1) else 'require'
            })

        return imports

    def _extract_exports(self, content: str) -> List[Dict[str, Any]]:
        """Extrait les exports du code"""
        exports = []

        for match in self.export_pattern.finditer(content):
            # Déterminer ce qui est exporté
            for i in range(1, 6):
                if match.group(i):
                    line_number = content[:match.start()].count('\n') + 1
                    exports.append({
                        'name': match.group(i),
                        'line_number': line_number,
                        'type': 'export'
                    })
                    break

        return exports

    def _analyze_js_patterns(self, content: str) -> Dict[str, Any]:
        """Analyse les patterns JavaScript spécifiques"""
        patterns = {
            'uses_react': bool(self.jsx_pattern.search(content)) or 'React' in content,
            'uses_vue': bool(self.vue_pattern.search(content)),
            'uses_node': 'require(' in content or 'process.' in content or '__dirname' in content,
            'has_hooks': bool(self.react_hook_pattern.search(content)),
            'is_module': 'export' in content or 'import' in content,
            'is_typescript': any(
                pattern in content for pattern in ['interface', 'type ', 'declare ', ': string', ': number']),
            'has_promises': 'Promise' in content or '.then(' in content or 'async ' in content,
            'uses_axios': 'axios.' in content or 'from \'axios\'' in content,
            'uses_lodash': 'lodash' in content or '_.' in content,
            'has_error_handling': 'try {' in content or '.catch(' in content,
        }

        return patterns

    def _calculate_metrics(self, content: str) -> CodeMetrics:
        """Calcule les métriques de base"""
        lines = content.split('\n')

        return CodeMetrics(
            lines_of_code=len(lines),
            comment_lines=self._count_comment_lines(content),
            empty_lines=self._count_empty_lines(lines),
            average_line_length=self._calculate_average_line_length(lines)
        )

    def _calculate_detailed_metrics(self, content: str, lines: List[str]) -> Dict[str, Any]:
        """Calcule des métriques détaillées"""
        # Compter les structures de contrôle
        control_structures = self._count_control_structures(content)

        # Calculer la complexité cyclomatique (simplifiée)
        complexity_score = self._calculate_complexity_score(content)

        # Calculer le ratio commentaires/code
        total_lines = len(lines)
        comment_lines = self._count_comment_lines(content)
        comment_ratio = comment_lines / total_lines if total_lines > 0 else 0

        return {
            'control_structures': control_structures,
            'complexity_score': complexity_score,
            'comment_ratio': comment_ratio,
            'function_count': len(self._extract_functions(content, lines)),
            'class_count': len(self._extract_classes(content, lines)),
            'nested_depth': self._calculate_max_nesting_depth(content)
        }

    def _count_comment_lines(self, content: str) -> int:
        """Compte les lignes de commentaires"""
        # Commentaires sur une ligne
        single_line_comments = len(re.findall(r'//.*', content))

        # Commentaires multi-lignes
        multi_line_comments = 0
        for match in re.finditer(r'/\*.*?\*/', content, re.DOTALL):
            multi_line_comments += match.group().count('\n') + 1

        return single_line_comments + multi_line_comments

    def _count_empty_lines(self, lines: List[str]) -> int:
        """Compte les lignes vides"""
        return sum(1 for line in lines if not line.strip())

    def _calculate_average_line_length(self, lines: List[str]) -> float:
        """Calcule la longueur moyenne des lignes"""
        if not lines:
            return 0.0
        total_length = sum(len(line) for line in lines)
        return total_length / len(lines)

    def _count_control_structures(self, content: str) -> Dict[str, int]:
        """Compte les structures de contrôle"""
        patterns = {
            'if': r'\bif\s*\(',
            'for': r'\bfor\s*\(',
            'while': r'\bwhile\s*\(',
            'switch': r'\bswitch\s*\(',
            'try': r'\btry\s*{',
            'catch': r'\bcatch\s*\(',
        }

        counts = {}
        for name, pattern in patterns.items():
            counts[name] = len(re.findall(pattern, content, re.MULTILINE))

        return counts

    def _calculate_complexity_score(self, content: str) -> int:
        """Calcule un score de complexité simplifié"""
        score = 0

        # Structures de contrôle
        control_patterns = [
            r'\bif\s*\(', r'\belse\b', r'\bfor\s*\(', r'\bwhile\s*\(',
            r'\bswitch\s*\(', r'\btry\s*{', r'\bcatch\s*\(', r'\bthrow\b'
        ]

        for pattern in control_patterns:
            score += len(re.findall(pattern, content, re.MULTILINE)) * 1

        # Fonctions et méthodes
        score += len(re.findall(r'function\s+\w+\s*\(|\([^)]*\)\s*=>|\b\w+\s*\([^)]*\)\s*{', content)) * 2

        # Promesses et async/await
        score += len(re.findall(r'\.then\(|\.catch\(|await\s+', content)) * 1

        return score

    def _calculate_max_nesting_depth(self, content: str) -> int:
        """Calcule la profondeur maximale d'imbrication"""
        max_depth = 0
        current_depth = 0

        for char in content:
            if char in '{[(':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char in '}])':
                current_depth -= 1

        return max_depth

    def _convert_to_code_elements(self, analysis: Dict[str, Any]) -> List[CodeElement]:
        """Convertit l'analyse brute en objets CodeElement"""
        elements = []

        # Fonctions
        for func in analysis.get('functions', []):
            elements.append(CodeElement(
                name=func.get('name', ''),
                element_type='function',
                line_start=func.get('line_number'),
                parameters=func.get('parameters', []),
                modifiers=['async'] if func.get('is_async') else [],
                metadata={'type': func.get('type', 'function')}
            ))

        # Classes
        for cls in analysis.get('classes', []):
            elements.append(CodeElement(
                name=cls.get('name', ''),
                element_type='class',
                line_start=cls.get('line_number'),
                modifiers=['extends'] if cls.get('parent') else [],
                metadata={
                    'parent': cls.get('parent'),
                    'methods': cls.get('methods', []),
                    'method_count': len(cls.get('methods', []))
                }
            ))

        return elements

    def _extract_dependencies(self, analysis: Dict[str, Any]) -> DependencyInfo:
        """Extrait les informations de dépendances"""
        imports = analysis.get('imports', [])
        exports = analysis.get('exports', [])

        # Séparer dépendances internes/externes
        internal_deps = []
        external_deps = []
        package_deps = []

        for imp in imports:
            if isinstance(imp, dict):
                module = imp.get('module', '')
                if imp.get('is_relative'):
                    internal_deps.append(module)
                elif '/' not in module or module.startswith('@'):  # Paquet npm
                    external_deps.append(module)
                    # Extraire le nom du paquet
                    pkg_name = module.split('/')[0]
                    if pkg_name and pkg_name not in package_deps:
                        package_deps.append(pkg_name)

        # Détecter le gestionnaire de paquets
        package_manager = 'npm'  # Par défaut pour JS

        # Vérifier la présence de package-lock.json ou yarn.lock dans le chemin parent
        file_path = analysis.get('file_path', '')
        if file_path:
            parent_dir = Path(file_path).parent
            if (parent_dir / 'yarn.lock').exists():
                package_manager = 'yarn'
            elif (parent_dir / 'pnpm-lock.yaml').exists():
                package_manager = 'pnpm'

        return DependencyInfo(
            imports=imports,
            exports=exports,
            internal_deps=internal_deps,
            external_deps=external_deps,
            package_deps=package_deps,
            package_manager=package_manager
        )

    def _detect_patterns(self, analysis: Dict[str, Any]) -> PatternDetection:
        """Détecte les patterns et frameworks"""
        patterns = []
        frameworks = []
        libraries = []
        architecture_hints = []

        js_analysis = analysis.get('analysis', {})

        if js_analysis.get('uses_react'):
            frameworks.append(FrameworkType.REACT)
            patterns.append('component')
            patterns.append('jsx')

        if js_analysis.get('uses_vue'):
            frameworks.append(FrameworkType.VUE)
            patterns.append('vue_component')

        if js_analysis.get('uses_node'):
            patterns.append('server_side')
            patterns.append('commonjs')
            architecture_hints.append('backend')

        if js_analysis.get('has_hooks'):
            patterns.append('react_hooks')

        if js_analysis.get('is_module'):
            patterns.append('es_modules')

        if js_analysis.get('is_typescript'):
            patterns.append('typescript')
            patterns.append('type_safety')

        if js_analysis.get('has_promises'):
            patterns.append('async_programming')
            patterns.append('promises')

        if js_analysis.get('uses_axios'):
            libraries.append('axios')
            patterns.append('http_client')

        if js_analysis.get('uses_lodash'):
            libraries.append('lodash')
            patterns.append('utility_library')

        if js_analysis.get('has_error_handling'):
            patterns.append('error_handling')

        # Détecter les patterns architecturaux
        if len(analysis.get('classes', [])) > 3:
            architecture_hints.append('oop_design')

        if len(analysis.get('functions', [])) > 10:
            architecture_hints.append('functional_style')

        return PatternDetection(
            patterns=patterns,
            frameworks=frameworks,
            libraries=libraries,
            architecture_hints=architecture_hints
        )

    def _analyze_security(self, content: str) -> SecurityAnalysis:
        """Analyse de sécurité basique"""
        warnings = []
        vulnerabilities = []

        # Détections de sécurité
        if self.eval_pattern.search(content):
            warnings.append("Utilisation de eval() détectée - risque de sécurité")
            vulnerabilities.append({
                'type': 'code_injection',
                'severity': 'high',
                'description': 'eval() peut permettre l\'exécution de code arbitraire'
            })

        if self.inner_html_pattern.search(content) and not re.search(r'DOMPurify|sanitize', content, re.IGNORECASE):
            warnings.append("innerHTML utilisé sans sanitization")
            vulnerabilities.append({
                'type': 'xss',
                'severity': 'medium',
                'description': 'innerHTML sans sanitization peut permettre des attaques XSS'
            })

        # Vérifier les clés API exposées
        api_key_patterns = [
            r'api[_-]?key\s*[:=]\s*[\'"][^\'"]+[\'"]',
            r'secret\s*[:=]\s*[\'"][^\'"]+[\'"]',
            r'token\s*[:=]\s*[\'"][^\'"]+[\'"]',
            r'password\s*[:=]\s*[\'"][^\'"]+[\'"]',
        ]

        for pattern in api_key_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                warnings.append("Possible clé API ou secret exposé dans le code")
                vulnerabilities.append({
                    'type': 'exposed_secret',
                    'severity': 'critical',
                    'description': 'Identifiants sensibles exposés dans le code'
                })
                break

        # Calculer un score de sécurité (100 = parfait)
        security_score = 100
        for vuln in vulnerabilities:
            if vuln['severity'] == 'critical':
                security_score -= 30
            elif vuln['severity'] == 'high':
                security_score -= 20
            elif vuln['severity'] == 'medium':
                security_score -= 10

        return SecurityAnalysis(
            warnings=warnings,
            vulnerabilities=vulnerabilities,
            security_score=max(0, security_score)
        )

    def get_supported_extensions(self) -> List[str]:
        """Retourne les extensions supportées"""
        return self.supported_extensions

    def get_file_type(self) -> FileType:
        """Retourne le type de fichier"""
        return self.file_type

    def can_analyze(self, file_path: str) -> bool:
        """Vérifie si le fichier peut être analysé"""
        return any(file_path.endswith(ext) for ext in self.supported_extensions)

    def cleanup(self):
        """Nettoie les ressources"""
        # Rien à nettoyer pour cette implémentation
        pass
