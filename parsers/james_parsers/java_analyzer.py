# java_analyzer.py
import logging
import re
from typing import Dict, List, Any

from james_parsers.analyzer import Analyzer

logger = logging.getLogger(__name__)


class JavaAnalyzer(Analyzer):
    def analyze(self, file_path: str) -> Dict[str, Any]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return self.analyze_content(content, file_path)
        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {e}")
            return {
                'file_type': 'java',
                'file_path': file_path,
                'error': str(e)
            }

    def analyze_content(self, content: str, file_path: str) -> Dict[str, Any]:
        return {
            'file_type': 'java',
            'file_path': file_path,
            'package': self._extract_package(content),
            'imports': self._extract_imports(content),
            'classes': self._extract_classes(content),
            'interfaces': self._extract_interfaces(content),
            'methods': self._extract_all_methods(content),
            'fields': self._extract_fields(content),
            'annotations': self._extract_annotations(content),
            'analysis': self._analyze_java_structure(content),
            'metrics': self._calculate_metrics(content)
        }

    def _extract_package(self, content: str) -> str:
        """Extrait la déclaration de package"""
        match = re.search(r'package\s+([^;]+);', content)
        return match.group(1).strip() if match else ""

    def _extract_imports(self, content: str) -> List[str]:
        """Extrait toutes les déclarations d'import"""
        imports = re.findall(r'import\s+([^;]+);', content)
        return [imp.strip() for imp in imports]

    def _extract_classes(self, content: str) -> List[Dict]:
        """Extrait les informations des classes"""
        classes = []
        # Pattern pour les classes avec leurs modificateurs
        class_pattern = r'(public|private|protected|abstract\s+)*\s*class\s+(\w+)'

        for match in re.finditer(class_pattern, content):
            modifiers = match.group(1) or ''
            class_name = match.group(2)

            classes.append({
                'name': class_name,
                'modifiers': self._clean_modifiers(modifiers),
                'is_abstract': 'abstract' in modifiers,
                'is_public': 'public' in modifiers,
                'is_private': 'private' in modifiers,
                'is_protected': 'protected' in modifiers
            })

        return classes

    def _extract_interfaces(self, content: str) -> List[Dict]:
        """Extrait les interfaces"""
        interfaces = []
        interface_pattern = r'(public|private|protected)?\s*interface\s+(\w+)'

        for match in re.finditer(interface_pattern, content):
            interfaces.append({
                'name': match.group(2),
                'modifiers': match.group(1) or 'package-private',
                'is_interface': True
            })

        return interfaces

    def _extract_all_methods(self, content: str) -> List[Dict]:
        """Extrait toutes les méthodes avec leurs détails"""
        methods = []

        # Pattern amélioré pour capturer les méthodes
        method_pattern = r'(@[\w\.]+\s*)*\s*(public|private|protected|[\s\S]*?)\s+(\w+)\s*\([^)]*\)\s*[^{]*\{'

        for match in re.finditer(method_pattern, content, re.DOTALL):
            full_match = match.group(0)
            annotations_text = match.group(1) or ''
            modifiers_text = match.group(2) or ''
            method_name = match.group(3)

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
                'is_final': 'final' in modifiers_text
            })

        return methods

    def _extract_fields(self, content: str) -> List[Dict]:
        """Extrait les champs/attributs de classe"""
        fields = []
        # Pattern pour les déclarations de champs
        field_pattern = r'(private|protected|public)\s+([\w<>,\s]+)\s+(\w+)\s*[=;]'

        for match in re.finditer(field_pattern, content):
            fields.append({
                'name': match.group(3),
                'type': match.group(2).strip(),
                'visibility': match.group(1),
                'is_public': match.group(1) == 'public',
                'is_private': match.group(1) == 'private',
                'is_protected': match.group(1) == 'protected'
            })

        return fields

    def _extract_annotations(self, content: str) -> List[str]:
        """Extrait toutes les annotations du fichier"""
        annotations = re.findall(r'@(\w+)', content)
        return list(set(annotations))  # Déduplication

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
        # Un constructeur a le même nom que sa classe
        classes = self._extract_classes(content)
        class_names = [cls['name'] for cls in classes]
        return method_name in class_names

    def _clean_modifiers(self, modifiers_text: str) -> str:
        """Nettoie et formate les modificateurs"""
        if not modifiers_text:
            return ''

        # Supprime les espaces multiples et split
        modifiers = re.sub(r'\s+', ' ', modifiers_text.strip()).split()
        return ' '.join(sorted(set(modifiers)))  # Déduplication et tri

    def _analyze_java_structure(self, content: str) -> Dict[str, Any]:
        """Analyse la structure et les patterns Java"""
        return {
            'has_spring_annotations': any(
                ann in content for ann in ['@Service', '@RestController', '@Repository', '@Component', '@Autowired']),
            'has_jpa_entities': any(ann in content for ann in ['@Entity', '@Table', '@Id', '@Column']),
            'has_transactional': '@Transactional' in content,
            'has_test_annotations': any(ann in content for ann in ['@Test', '@Before', '@After']),
            'design_patterns': self._detect_design_patterns(content),
            'framework_indicators': self._detect_frameworks(content)
        }

    def _detect_design_patterns(self, content: str) -> List[str]:
        """Détecte les design patterns potentiels"""
        patterns = []

        # Singleton pattern
        if 'getInstance()' in content or 'instance' in content and 'static' in content:
            patterns.append('Singleton')

        # Factory pattern
        if 'Factory' in content or 'create' in content and 'static' in content:
            patterns.append('Factory')

        # Builder pattern
        if 'Builder' in content and 'static class' in content:
            patterns.append('Builder')

        # Observer pattern
        if 'addListener' in content or 'EventListener' in content:
            patterns.append('Observer')

        # Dependency Injection
        if '@Autowired' in content or '@Inject' in content:
            patterns.append('Dependency Injection')

        return patterns

    def _detect_frameworks(self, content: str) -> List[str]:
        """Détecte les frameworks utilisés"""
        frameworks = []

        # Spring
        if any(spring in content for spring in ['@SpringBoot', '@Service', '@RestController', '@Autowired']):
            frameworks.append('Spring')

        # JPA/Hibernate
        if any(jpa in content for jpa in ['@Entity', '@Table', '@Id', '@Column']):
            frameworks.append('JPA')

        # JUnit
        if any(test in content for test in ['@Test', '@Before', '@After', 'org.junit']):
            frameworks.append('JUnit')

        # Jackson
        if any(jackson in content for jackson in ['@JsonProperty', '@JsonIgnore', 'ObjectMapper']):
            frameworks.append('Jackson')

        return frameworks

    def _calculate_metrics(self, content: str) -> Dict[str, Any]:
        """Calcule des métriques sur le code"""
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
            'catch_blocks': len(re.findall(r'\bcatch\s*\(', content))
        }
