# java_analyzer.py
import logging
import re
from typing import Dict, List, Any
from datetime import datetime

from parsers.analyzer import Analyzer
from parsers.analysis_result import (
    AnalysisResult
)

logger = logging.getLogger(__name__)


class JavaAnalyzer(Analyzer):
    """Analyseur de fichiers Java retournant des AnalysisResult"""

    def __init__(self):
        super().__init__("java")

    def analyze_file(self, file_path: str) -> AnalysisResult:
        """Analyse un fichier Java et retourne un AnalysisResult"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()


            # Créer le résultat de base
            result = self._create_base_result(file_path)

            # Analyser le contenu
            analysis = self.analyze_content(content, file_path)

            # Mettre à jour les résultats
            self._update_result_from_analysis(result, analysis)
            result.metrics = self._calculate_java_metrics(content, analysis)



            return result

        except Exception as e:
            logger.error(f"Error analyzing Java file {file_path}: {e}")
            return self._create_error_result(file_path, str(e))

    def analyze_content(self, content: str, file_path: str) -> AnalysisResult:
        """Analyse du contenu Java (méthode interne)"""
        return {
            'package': self._extract_package(content),
            'imports': self._extract_imports(content),
            'classes': self._extract_classes(content),
            'interfaces': self._extract_interfaces(content),
            'methods': self._extract_all_methods(content),
            'fields': self._extract_fields(content),
            'annotations': self._extract_annotations(content),
            'enums': self._extract_enums(content),
            'constructors': self._extract_constructors(content),
            'analysis': self._analyze_java_structure(content),
            'raw_metrics': self._calculate_raw_metrics(content)
        }

    def _update_result_from_analysis(self, result: AnalysisResult, analysis: Dict[str, Any]) -> None:
        """Met à jour l'AnalysisResult avec les données de l'analyse Java"""

        # Ajouter les éléments de code (classes, interfaces, méthodes, champs)
        self._add_java_elements(result, analysis)

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

    def _add_java_elements(self, result: AnalysisResult, analysis: Dict[str, Any]) -> None:
        """Convertit les éléments Java en CodeElement"""

        # Classes
        for cls in analysis['classes']:
            class_element = CodeElement(
                name=cls['name'],
                element_type='class',
                modifiers=cls['modifiers'],
                metadata={
                    'is_abstract': cls['is_abstract'],
                    'is_public': cls['is_public'],
                    'is_private': cls['is_private'],
                    'is_protected': cls['is_protected'],
                    'type': 'class'
                }
            )
            result.elements.append(class_element)
            result.metrics.class_count += 1

        # Interfaces
        for interface in analysis['interfaces']:
            interface_element = CodeElement(
                name=interface['name'],
                element_type='interface',
                modifiers=interface['modifiers'],
                metadata={
                    'is_interface': True,
                    'type': 'interface'
                }
            )
            result.elements.append(interface_element)
            result.metrics.class_count += 1

        # Méthodes
        for method in analysis['methods']:
            method_element = CodeElement(
                name=method['name'],
                element_type='method',
                modifiers=method['modifiers'],
                metadata={
                    'visibility': method['visibility'],
                    'annotations': method['annotations'],
                    'is_constructor': method['is_constructor'],
                    'is_public': method['is_public'],
                    'is_private': method['is_private'],
                    'is_protected': method['is_protected'],
                    'is_static': method['is_static'],
                    'is_abstract': method['is_abstract'],
                    'is_final': method['is_final'],
                    'type': 'method'
                }
            )
            result.elements.append(method_element)
            result.metrics.function_count += 1

        # Champs
        for field in analysis['fields']:
            field_element = CodeElement(
                name=field['name'],
                element_type='variable',
                modifiers=[field['visibility']],
                metadata={
                    'data_type': field['type'],
                    'is_field': True,
                    'is_public': field['is_public'],
                    'is_private': field['is_private'],
                    'is_protected': field['is_protected'],
                    'type': 'field'
                }
            )
            result.elements.append(field_element)

        # Enums
        for enum in analysis['enums']:
            enum_element = CodeElement(
                name=enum['name'],
                element_type='enum',
                modifiers=enum['modifiers'],
                metadata={
                    'values': enum['values'],
                    'is_enum': True,
                    'type': 'enum'
                }
            )
            result.elements.append(enum_element)
            result.metrics.class_count += 1

    def _update_dependencies(self, result: AnalysisResult, analysis: Dict[str, Any]) -> None:
        """Met à jour les informations de dépendances"""
        deps = result.dependencies

        # Imports
        imports_list = []
        for imp in analysis['imports']:
            import_info = {
                'path': imp,
                'is_static': 'static' in imp,
                'is_wildcard': '*' in imp
            }

            # Détecter les dépendances internes/externes
            if imp.startswith('java.') or imp.startswith('javax.'):
                import_info['type'] = 'jdk'
                deps.external_deps.append(imp.split('.')[0])
            elif 'org.springframework' in imp:
                import_info['type'] = 'spring'
                deps.external_deps.append('spring')
            elif 'jakarta' in imp or 'javax.persistence' in imp:
                import_info['type'] = 'jpa'
                deps.external_deps.append('jpa')
            elif 'org.junit' in imp:
                import_info['type'] = 'test'
                deps.external_deps.append('junit')
            else:
                import_info['type'] = 'custom'
                # Essayer de déterminer si c'est interne
                package_parts = analysis['package'].split('.') if analysis['package'] else []
                if package_parts and imp.startswith(package_parts[0]):
                    deps.internal_deps.append(imp)
                else:
                    deps.external_deps.append(imp.split('.')[0])

            imports_list.append(import_info)

        deps.imports = imports_list

        # Déterminer le package manager
        if any('org.springframework' in imp for imp in analysis['imports']):
            deps.package_manager = 'maven'
        elif any('org.junit' in imp for imp in analysis['imports']):
            deps.package_manager = 'maven'

    def _update_patterns(self, result: AnalysisResult, analysis: Dict[str, Any]) -> None:
        """Met à jour la détection de patterns"""
        patterns = result.patterns
        structure_analysis = analysis['analysis']

        # Détecter les frameworks
        frameworks = structure_analysis.get('framework_indicators', [])
        if 'Spring' in frameworks:
            patterns.frameworks.append(FrameworkType.SPRING)
        if 'JUnit' in frameworks:
            patterns.frameworks.append(FrameworkType.SPRING)  # Pour les tests

        # Détecter les design patterns
        design_patterns = structure_analysis.get('design_patterns', [])
        patterns.patterns = design_patterns

        # Détecter les annotations Spring
        if structure_analysis.get('has_spring_annotations'):
            patterns.libraries.append('spring-framework')
        if structure_analysis.get('has_jpa_entities'):
            patterns.libraries.append('jpa')
        if structure_analysis.get('has_transactional'):
            patterns.libraries.append('spring-transaction')

        # Architecture hints
        if len(analysis['classes']) > 0:
            if any('Service' in cls['name'] for cls in analysis['classes']):
                patterns.architecture_hints.append('service-layer')
            if any('Controller' in cls['name'] or 'RestController' in cls['name'] for cls in analysis['classes']):
                patterns.architecture_hints.append('controller-layer')
            if any('Repository' in cls['name'] for cls in analysis['classes']):
                patterns.architecture_hints.append('repository-pattern')

        # Détecter les tests
        if structure_analysis.get('has_test_annotations'):
            patterns.architecture_hints.append('test-file')

    def _update_security(self, result: AnalysisResult, analysis: Dict[str, Any]) -> None:
        """Met à jour l'analyse de sécurité"""
        security = result.security

        # Vérifier les vulnérabilités potentielles
        content = "\n".join([str(v) for v in analysis.values()])

        if 'Runtime.getRuntime().exec(' in content:
            security.warnings.append("Potential command injection vulnerability with Runtime.exec()")
            security.vulnerabilities.append({
                'type': 'command_injection',
                'severity': 'high',
                'description': 'Use of Runtime.exec() may lead to command injection'
            })

        if 'System.out.println' in content and analysis['classes']:
            if any('Controller' in cls['name'] for cls in analysis['classes']):
                security.warnings.append("Consider using proper logging instead of System.out in controllers")

        # Vérifier les annotations de sécurité
        annotations = analysis['annotations']
        security_annotations = ['@Secured', '@PreAuthorize', '@RolesAllowed', '@PostAuthorize']
        found_security = [ann for ann in security_annotations if ann in annotations]

        if found_security:
            security.recommendations.append(f"Security annotations found: {', '.join(found_security)}")
        else:
            if any('Controller' in cls['name'] for cls in analysis['classes']):
                security.recommendations.append("Consider adding security annotations to controller methods")

        # Calculer un score de sécurité basique
        security.security_score = self._calculate_security_score(analysis)

    def _update_language_specific(self, result: AnalysisResult, analysis: Dict[str, Any]) -> None:
        """Met à jour les données spécifiques au Java"""
        result.language_specific = {
            'java': {
                'package': analysis['package'],
                'import_count': len(analysis['imports']),
                'class_count': len(analysis['classes']),
                'interface_count': len(analysis['interfaces']),
                'method_count': len(analysis['methods']),
                'field_count': len(analysis['fields']),
                'annotation_count': len(analysis['annotations']),
                'enum_count': len(analysis['enums']),
                'constructor_count': len(analysis['constructors']),
                'spring_detected': analysis['analysis']['has_spring_annotations'],
                'jpa_detected': analysis['analysis']['has_jpa_entities'],
                'test_file': analysis['analysis']['has_test_annotations'],
                'is_main_class': self._is_main_class(analysis)
            }
        }

    def _add_diagnostics(self, result: AnalysisResult, analysis: Dict[str, Any]) -> None:
        """Ajoute des diagnostics et recommandations"""

        # Vérifier le package
        if not analysis['package']:
            result.warnings.append("No package declaration found")

        # Vérifier le nombre de classes par fichier
        total_types = len(analysis['classes']) + len(analysis['interfaces']) + len(analysis['enums'])
        if total_types > 1:
            result.notes.append(f"Multiple types ({total_types}) in single file - consider separating")

        # Vérifier la longueur des méthodes
        if len(analysis['methods']) > 0:
            result.notes.append(f"Contains {len(analysis['methods'])} methods")

        # Vérifier les imports
        if len(analysis['imports']) > 20:
            result.warnings.append("Large number of imports - consider refactoring")

    def _extract_package(self, content: str) -> str:
        """Extrait la déclaration de package"""
        match = re.search(r'package\s+([^;]+);', content)
        return match.group(1).strip() if match else ""

    def _extract_imports(self, content: str) -> List[str]:
        """Extrait toutes les déclarations d'import"""
        imports = re.findall(r'import\s+(?:static\s+)?([^;]+);', content)
        return [imp.strip() for imp in imports]

    def _extract_classes(self, content: str) -> List[Dict]:
        """Extrait les informations des classes"""
        classes = []
        class_pattern = r'(public|private|protected|abstract|final\s+)*\s*class\s+(\w+)'

        for match in re.finditer(class_pattern, content):
            modifiers = match.group(1) or ''
            class_name = match.group(2)

            classes.append({
                'name': class_name,
                'modifiers': self._clean_modifiers(modifiers),
                'is_abstract': 'abstract' in modifiers,
                'is_public': 'public' in modifiers,
                'is_private': 'private' in modifiers,
                'is_protected': 'protected' in modifiers,
                'is_final': 'final' in modifiers
            })

        return classes

    def _extract_interfaces(self, content: str) -> List[Dict]:
        """Extrait les interfaces"""
        interfaces = []
        interface_pattern = r'(public|private|protected|abstract)?\s*interface\s+(\w+)'

        for match in re.finditer(interface_pattern, content):
            modifiers = match.group(1) or ''
            interface_name = match.group(2)

            interfaces.append({
                'name': interface_name,
                'modifiers': self._clean_modifiers(modifiers),
                'is_interface': True,
                'is_public': 'public' in modifiers,
                'is_private': 'private' in modifiers,
                'is_protected': 'protected' in modifiers
            })

        return interfaces

    def _extract_enums(self, content: str) -> List[Dict]:
        """Extrait les enums"""
        enums = []
        enum_pattern = r'(public|private|protected)?\s*enum\s+(\w+)'

        for match in re.finditer(enum_pattern, content):
            modifiers = match.group(1) or ''
            enum_name = match.group(2)

            # Extraire les valeurs de l'enum
            enum_body_match = re.search(r'enum\s+' + re.escape(enum_name) + r'\s*\{([^}]+)\}', content)
            values = []
            if enum_body_match:
                values = [v.strip() for v in enum_body_match.group(1).split(',') if v.strip()]

            enums.append({
                'name': enum_name,
                'modifiers': self._clean_modifiers(modifiers),
                'values': values,
                'is_enum': True,
                'is_public': 'public' in modifiers,
                'is_private': 'private' in modifiers,
                'is_protected': 'protected' in modifiers
            })

        return enums

    def _extract_all_methods(self, content: str) -> List[Dict]:
        """Extrait toutes les méthodes avec leurs détails"""
        methods = []
        method_pattern = r'(@[\w\.]+\s*)*\s*(public|private|protected|[\s\S]*?)\s+(\w+)\s*\([^)]*\)\s*[^{]*\{'

        for match in re.finditer(method_pattern, content, re.DOTALL):
            full_match = match.group(0)
            annotations_text = match.group(1) or ''
            modifiers_text = match.group(2) or ''
            method_name = match.group(3)

            # Extraire les paramètres
            params_match = re.search(r'\(([^)]*)\)', full_match)
            parameters = []
            if params_match:
                params_text = params_match.group(1)
                params = [p.strip() for p in params_text.split(',') if p.strip()]
                parameters = params

            # Extraire le type de retour si présent
            return_type = 'void'
            return_match = re.search(r'(\w+)\s+' + re.escape(method_name) + r'\s*\(', full_match)
            if return_match and return_match.group(1) not in ['public', 'private', 'protected', 'static', 'final', 'abstract', 'synchronized']:
                return_type = return_match.group(1)

            methods.append({
                'name': method_name,
                'visibility': self._extract_visibility(modifiers_text),
                'annotations': self._extract_method_annotations(annotations_text),
                'is_constructor': self._is_constructor(method_name, content),
                'modifiers': self._clean_modifiers(modifiers_text),
                'is_public': 'public' in modifiers_text,
                'is_private': 'private' in modifiers_text,
                'is_protected': 'protected' in modifiers_text,
                'is_static': 'static' in modifiers_text,
                'is_abstract': 'abstract' in modifiers_text,
                'is_final': 'final' in modifiers_text,
                'is_synchronized': 'synchronized' in modifiers_text,
                'parameters': parameters,
                'return_type': return_type
            })

        return methods

    def _extract_constructors(self, content: str) -> List[Dict]:
        """Extrait les constructeurs"""
        constructors = []
        classes = self._extract_classes(content)
        class_names = [cls['name'] for cls in classes]

        for class_name in class_names:
            constructor_pattern = rf'(public|private|protected)?\s*{re.escape(class_name)}\s*\(([^)]*)\)'
            for match in re.finditer(constructor_pattern, content):
                modifiers = match.group(1) or ''
                params = match.group(2) or ''

                constructors.append({
                    'class_name': class_name,
                    'modifiers': modifiers,
                    'parameters': [p.strip() for p in params.split(',') if p.strip()],
                    'is_public': 'public' in modifiers,
                    'is_private': 'private' in modifiers,
                    'is_protected': 'protected' in modifiers
                })

        return constructors

    def _extract_fields(self, content: str) -> List[Dict]:
        """Extrait les champs/attributs de classe"""
        fields = []
        field_pattern = r'(private|protected|public|static|final\s+)*\s*([\w<>,\s]+)\s+(\w+)\s*[=;]'

        for match in re.finditer(field_pattern, content):
            modifiers = match.group(1) or ''
            field_type = match.group(2).strip()
            field_name = match.group(3)

            fields.append({
                'name': field_name,
                'type': field_type,
                'modifiers': self._clean_modifiers(modifiers),
                'visibility': self._extract_visibility(modifiers),
                'is_public': 'public' in modifiers,
                'is_private': 'private' in modifiers,
                'is_protected': 'protected' in modifiers,
                'is_static': 'static' in modifiers,
                'is_final': 'final' in modifiers
            })

        return fields

    def _extract_annotations(self, content: str) -> List[str]:
        """Extrait toutes les annotations du fichier"""
        annotations = re.findall(r'@(\w+)', content)
        return list(set(annotations))

    def _extract_visibility(self, modifiers_text: str) -> str:
        """Extrait la visibilité d'une méthode/champ"""
        modifiers_text = modifiers_text or ''

        if 'public' in modifiers_text:
            return 'public'
        elif 'private' in modifiers_text:
            return 'private'
        elif 'protected' in modifiers_text:
            return 'protected'
        else:
            return 'package-private'

    def _extract_method_annotations(self, annotations_text: str) -> List[str]:
        """Extrait les annotations spécifiques à une méthode"""
        if not annotations_text:
            return []

        annotations = re.findall(r'@(\w+)', annotations_text)
        return annotations

    def _is_constructor(self, method_name: str, content: str) -> bool:
        """Détermine si une méthode est un constructeur"""
        classes = self._extract_classes(content)
        class_names = [cls['name'] for cls in classes]
        return method_name in class_names

    def _clean_modifiers(self, modifiers_text: str) -> List[str]:
        """Nettoie et formate les modificateurs"""
        if not modifiers_text:
            return []

        # Supprime les espaces multiples et split
        modifiers = re.sub(r'\s+', ' ', modifiers_text.strip()).split()
        return sorted(set(modifiers))

    def _analyze_java_structure(self, content: str) -> Dict[str, Any]:
        """Analyse la structure et les patterns Java"""
        return {
            'has_spring_annotations': any(
                ann in content for ann in ['@Service', '@RestController', '@Repository', '@Component', '@Autowired']),
            'has_jpa_entities': any(ann in content for ann in ['@Entity', '@Table', '@Id', '@Column']),
            'has_transactional': '@Transactional' in content,
            'has_test_annotations': any(ann in content for ann in ['@Test', '@Before', '@After', '@BeforeEach', '@AfterEach']),
            'has_lombok': any(ann in content for ann in ['@Data', '@Getter', '@Setter', '@Builder', '@NoArgsConstructor']),
            'design_patterns': self._detect_design_patterns(content),
            'framework_indicators': self._detect_frameworks(content),
            'architecture_style': self._detect_architecture_style(content)
        }

    def _detect_design_patterns(self, content: str) -> List[str]:
        """Détecte les design patterns potentiels"""
        patterns = []

        # Singleton pattern
        if 'getInstance()' in content or ('instance' in content and 'static' in content and 'private' in content):
            patterns.append('Singleton')

        # Factory pattern
        if 'Factory' in content or ('create' in content and 'static' in content):
            patterns.append('Factory')

        # Builder pattern
        if 'Builder' in content and 'static class' in content:
            patterns.append('Builder')

        # Observer pattern
        if 'addListener' in content or 'EventListener' in content:
            patterns.append('Observer')

        # Strategy pattern
        if 'Strategy' in content or ('execute' in content and 'interface' in content):
            patterns.append('Strategy')

        # Template Method pattern
        if 'abstract class' in content and 'protected abstract' in content:
            patterns.append('Template Method')

        return patterns

    def _detect_frameworks(self, content: str) -> List[str]:
        """Détecte les frameworks utilisés"""
        frameworks = []

        # Spring
        if any(spring in content for spring in ['@SpringBoot', '@Service', '@RestController', '@Autowired', '@Component']):
            frameworks.append('Spring')

        # Spring Boot
        if '@SpringBootApplication' in content:
            frameworks.append('Spring Boot')

        # JPA/Hibernate
        if any(jpa in content for jpa in ['@Entity', '@Table', '@Id', '@Column', '@ManyToOne', '@OneToMany']):
            frameworks.append('JPA')

        # JUnit
        if any(test in content for test in ['@Test', '@Before', '@After', 'org.junit', '@BeforeEach', '@AfterEach']):
            frameworks.append('JUnit')

        # Jackson
        if any(jackson in content for jackson in ['@JsonProperty', '@JsonIgnore', 'ObjectMapper', '@JsonSerialize']):
            frameworks.append('Jackson')

        # Lombok
        if any(lombok in content for lombok in ['@Data', '@Getter', '@Setter', '@Builder', '@NoArgsConstructor']):
            frameworks.append('Lombok')

        return frameworks

    def _detect_architecture_style(self, content: str) -> str:
        """Détecte le style d'architecture"""
        if '@RestController' in content or '@Controller' in content:
            return 'web-mvc'
        elif '@Entity' in content and '@Repository' in content:
            return 'data-access'
        elif '@Service' in content and '@Transactional' in content:
            return 'service-layer'
        elif any(test in content for test in ['@Test', '@Before', '@After']):
            return 'test'
        else:
            return 'utility'

    def _calculate_raw_metrics(self, content: str) -> Dict[str, Any]:
        """Calcule des métriques brutes sur le code"""
        lines = content.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]

        return {
            'line_count': len(lines),
            'non_empty_lines': len(non_empty_lines),
            'comment_lines': len([line for line in lines if line.strip().startswith('//') or '/*' in line]),
            'method_count': len(self._extract_all_methods(content)),
            'class_count': len(self._extract_classes(content)),
            'field_count': len(self._extract_fields(content)),
            'import_count': len(self._extract_imports(content)),
            'complexity_indicators': self._calculate_complexity(content)
        }

    def _calculate_complexity(self, content: str) -> Dict[str, int]:
        """Calcule des indicateurs de complexité"""
        return {
            'if_statements': len(re.findall(r'\bif\s*\(', content)),
            'for_loops': len(re.findall(r'\bfor\s*\(', content)),
            'while_loops': len(re.findall(r'\bwhile\s*\(', content)),
            'switch_statements': len(re.findall(r'\bswitch\s*\(', content)),
            'try_blocks': len(re.findall(r'\btry\s*\{', content)),
            'catch_blocks': len(re.findall(r'\bcatch\s*\(', content)),
            'nested_blocks': len(re.findall(r'\}\s*\{', content))
        }

    def _calculate_java_metrics(self, content: str, analysis: Dict[str, Any]) -> FileMetrics:
        """Calcule les métriques spécifiques au Java"""
        metrics = super()._calculate_metrics(content)
        raw_metrics = analysis['raw_metrics']
        complexity = raw_metrics['complexity_indicators']

        # Mettre à jour avec les métriques spécifiques au Java
        metrics.total_lines = raw_metrics['line_count']
        metrics.code_lines = raw_metrics['non_empty_lines']
        metrics.comment_lines = raw_metrics['comment_lines']
        metrics.blank_lines = metrics.total_lines - metrics.code_lines - metrics.comment_lines
        metrics.function_count = raw_metrics['method_count']
        metrics.class_count = raw_metrics['class_count']
        metrics.import_count = raw_metrics['import_count']

        # Calculer la complexité cyclomatique simplifiée
        complexity_score = (
                complexity['if_statements'] +
                complexity['for_loops'] +
                complexity['while_loops'] +
                complexity['switch_statements'] +
                complexity['catch_blocks'] +
                complexity['nested_blocks']
        )
        metrics.complexity_score = complexity_score / max(1, metrics.function_count)

        return metrics

    def _calculate_security_score(self, analysis: Dict[str, Any]) -> float:
        """Calcule un score de sécurité basique"""
        score = 100.0

        # Pénalités pour les problèmes de sécurité
        content = "\n".join([str(v) for v in analysis.values()])

        if 'Runtime.getRuntime().exec(' in content:
            score -= 30

        if 'System.out.println' in content and any('Controller' in cls['name'] for cls in analysis['classes']):
            score -= 10

        # Bonus pour les annotations de sécurité
        security_annotations = ['@Secured', '@PreAuthorize', '@RolesAllowed', '@PostAuthorize']
        found_security = len([ann for ann in security_annotations if ann in analysis['annotations']])
        score += found_security * 5

        return max(0, min(100, score))

    def _is_main_class(self, analysis: Dict[str, Any]) -> bool:
        """Détermine si c'est une classe principale"""
        for method in analysis['methods']:
            if (method['name'] == 'main' and
                    'public static void main(String[]' in str(method) and
                    method['is_public'] and method['is_static']):
                return True
        return False