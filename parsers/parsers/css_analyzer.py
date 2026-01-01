# james_parsers/css_analyzer.py
import re
import logging
from typing import Dict, List, Any
from .analyzer import Analyzer

logger = logging.getLogger(__name__)

class CSSAnalyzer(Analyzer):
    def analyze(self, file_path: str) -> Dict[str, Any]:
        """Analyse un fichier CSS"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return self.analyze_content(content, file_path)
        except Exception as e:
            logger.error(f"Error reading CSS file {file_path}: {e}")
            return self._create_error_response(file_path, str(e))

    def analyze_content(self, content: str, file_path: str = "") -> Dict[str, Any]:
        """Analyse le contenu CSS"""
        try:
            return {
                'file_type': 'css',
                'file_path': file_path,
                'rules': self._extract_rules(content),
                'selectors': self._extract_selectors(content),
                'properties': self._extract_properties(content),
                'media_queries': self._extract_media_queries(content),
                'analysis': self._analyze_css_patterns(content)
            }
        except Exception as e:
            logger.error(f"Error analyzing CSS content {file_path}: {e}")
            return self._create_error_response(file_path, str(e))

    def _create_error_response(self, file_path: str, error: str) -> Dict[str, Any]:
        """Crée une réponse d'erreur standardisée"""
        return {
            'file_type': 'css',
            'file_path': file_path,
            'error': error,
            'rules': [],
            'selectors': [],
            'properties': {},
            'media_queries': [],
            'analysis': {
                'total_rules': 0,
                'total_selectors': 0,
                'selector_types': {},
                'common_properties': [],
                'has_animations': False,
                'has_flexbox': False,
                'has_grid': False,
                'specificity_analysis': {'average': 0, 'max': 0, 'min': 0}
            }
        }

    def _extract_rules(self, content: str) -> List[Dict]:
        """Extrait les règles CSS"""
        rules = []
        rule_pattern = r'([^{]+)\{([^}]+)\}'

        for match in re.finditer(rule_pattern, content):
            selectors = match.group(1).strip()
            declarations = match.group(2).strip()

            rules.append({
                'selectors': [s.strip() for s in selectors.split(',')],
                'declarations': self._parse_declarations(declarations),
                'specificity': self._calculate_specificity(selectors)
            })

        return rules

    def _parse_declarations(self, declarations: str) -> List[Dict]:
        """Parse les déclarations CSS (propriété: valeur)"""
        parsed = []
        # Sépare les déclarations par ; et filtre les vides
        decl_list = [d.strip() for d in declarations.split(';') if d.strip()]

        for decl in decl_list:
            if ':' in decl:
                prop, _, value = decl.partition(':')
                parsed.append({
                    'property': prop.strip(),
                    'value': value.strip()
                })

        return parsed

    def _extract_selectors(self, content: str) -> List[Dict]:
        """Extrait les sélecteurs CSS avec spécificité"""
        selectors = []
        rule_pattern = r'([^{]+)\{'

        for match in re.finditer(rule_pattern, content):
            selector_text = match.group(1).strip()
            individual_selectors = [s.strip() for s in selector_text.split(',')]

            for selector in individual_selectors:
                selectors.append({
                    'selector': selector,
                    'type': self._classify_selector(selector),
                    'complexity': self._calculate_selector_complexity(selector),
                    'specificity': self._calculate_specificity(selector)
                })

        return selectors

    def _classify_selector(self, selector: str) -> str:
        """Classifie le type de sélecteur CSS"""
        if selector.startswith('.'):
            return 'class'
        elif selector.startswith('#'):
            return 'id'
        elif selector.startswith('@'):
            return 'at_rule'
        elif re.match(r'^[a-zA-Z]', selector):
            return 'element'
        elif selector.startswith('['):
            return 'attribute'
        else:
            return 'complex'

    def _calculate_selector_complexity(self, selector: str) -> int:
        """Calcule la complexité d'un sélecteur"""
        # Nombre de combinateurs et pseudo-classes
        combinators = len(re.findall(r'[\s>+~]', selector))
        pseudo_classes = len(re.findall(r':\w', selector))
        return combinators + pseudo_classes

    def _calculate_specificity(self, selector: str) -> int:
        """Calcule la spécificité d'un sélecteur CSS"""
        try:
            id_count = len(re.findall(r'#\w', selector))
            class_count = len(re.findall(r'\.\w', selector))
            element_count = len(re.findall(r'(^|[\s>+~])\w', selector))
            return id_count * 100 + class_count * 10 + element_count
        except:
            return 0

    def _extract_properties(self, content: str) -> Dict[str, int]:
        """Extrait et compte les propriétés CSS utilisées"""
        properties = {}
        prop_pattern = r'([\w-]+)\s*:'

        for match in re.finditer(prop_pattern, content):
            prop = match.group(1)
            properties[prop] = properties.get(prop, 0) + 1

        return properties

    def _extract_media_queries(self, content: str) -> List[Dict]:
        """Extrait les media queries"""
        media_queries = []
        media_pattern = r'@media\s*([^{]+)\{([^}]+)\}'

        for match in re.finditer(media_pattern, content):
            media_queries.append({
                'conditions': match.group(1).strip(),
                'rules_count': len(re.findall(r'\{[^}]+\}', match.group(2)))
            })

        return media_queries

    def _analyze_css_patterns(self, content: str) -> Dict[str, Any]:
        """Analyse les patterns CSS avec gestion d'erreurs"""
        try:
            rules = self._extract_rules(content)
            selectors = self._extract_selectors(content)

            return {
                'total_rules': len(rules),
                'total_selectors': len(selectors),
                'selector_types': self._count_selector_types(selectors),
                'common_properties': self._get_common_properties(content),
                'has_animations': self._has_animations(content),
                'has_flexbox': self._has_flexbox(content),
                'has_grid': self._has_grid(content),
                'specificity_analysis': self._analyze_specificity(selectors)
            }
        except Exception as e:
            logger.warning(f"Error in CSS pattern analysis: {e}")
            return {
                'total_rules': 0,
                'total_selectors': 0,
                'selector_types': {},
                'common_properties': [],
                'has_animations': False,
                'has_flexbox': False,
                'has_grid': False,
                'specificity_analysis': {'average': 0, 'max': 0, 'min': 0}
            }

    def _count_selector_types(self, selectors: List[Dict]) -> Dict[str, int]:
        """Compte les types de sélecteurs"""
        type_count = {}
        for selector in selectors:
            sel_type = selector.get('type', 'unknown')
            type_count[sel_type] = type_count.get(sel_type, 0) + 1
        return type_count

    def _get_common_properties(self, content: str) -> List[str]:
        """Retourne les propriétés CSS les plus utilisées"""
        properties = self._extract_properties(content)
        # Retourne les 10 propriétés les plus utilisées
        sorted_props = sorted(properties.items(), key=lambda x: x[1], reverse=True)
        return [prop for prop, count in sorted_props[:10]]

    def _has_animations(self, content: str) -> bool:
        """Vérifie la présence d'animations"""
        return '@keyframes' in content or 'animation:' in content

    def _has_flexbox(self, content: str) -> bool:
        """Vérifie la présence de Flexbox"""
        return bool(re.search(r'display:\s*flex', content))

    def _has_grid(self, content: str) -> bool:
        """Vérifie la présence de Grid"""
        return bool(re.search(r'display:\s*grid', content))

    def _analyze_specificity(self, selectors: List[Dict]) -> Dict[str, Any]:
        """Analyse la spécificité des sélecteurs"""
        try:
            if not selectors:
                return {'average': 0, 'max': 0, 'min': 0}

            specificities = [s.get('specificity', 0) for s in selectors]
            return {
                'average': sum(specificities) / len(specificities),
                'max': max(specificities),
                'min': min(specificities)
            }
        except:
            return {'average': 0, 'max': 0, 'min': 0}