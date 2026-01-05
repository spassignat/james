# parsers/css_analyzer.py
import logging
import re
from typing import Dict, List, Any

import time

from parsers.analysis_result import AnalysisResult, FileType, FileMetrics, CodeElement, PatternDetection
from parsers.analyzer import Analyzer

logger = logging.getLogger(__name__)


class CSSAnalyzer(Analyzer):
    def __init__(self):
        super().__init__()
        self.file_type = FileType.CSS
        self.supported_extensions = ['.css', '.scss', '.sass', '.less']

    def analyze_file(self, file_path: str) -> AnalysisResult:
        """Analyse un fichier CSS et retourne un AnalysisResult"""
        start_time = time.time() * 1000

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            result = self._analyze_css_content(content, file_path)
            result.processing_time_ms = int(time.time() * 1000 - start_time)
            return result

        except Exception as e:
            logger.error(f"Erreur analyse CSS {file_path}: {e}")
            result = self._create_error_result(file_path, str(e))
            result.processing_time_ms = int(time.time() * 1000 - start_time)
            return result

    def _analyze_css_content(self, content: str, file_path: str) -> AnalysisResult:
        """Analyse le contenu CSS et crée un AnalysisResult"""
        result = self._create_base_result(file_path)

        # Analyse CSS complète
        css_analysis = self.analyze_content(content, file_path)

        # Remplir les champs du résultat
        result.metrics = self._calculate_css_metrics(content)
        result.elements = self._convert_to_code_elements(css_analysis)
        result.patterns = self._detect_css_patterns(css_analysis)
        result.language_specific = css_analysis

        return result

    def analyze_content(self, content: str, file_path: str = "") -> Dict[str, Any]:
        """Analyse le contenu CSS (méthode existante adaptée)"""
        try:
            rules = self._extract_rules(content)
            selectors = self._extract_selectors(content)

            return {
                'file_type': 'css',
                'file_path': file_path,
                'rules': rules,
                'selectors': selectors,
                'properties': self._extract_properties(content),
                'media_queries': self._extract_media_queries(content),
                'keyframes': self._extract_keyframes(content),
                'variables': self._extract_css_variables(content),
                'analysis': self._analyze_css_patterns(rules, selectors, content),
                'metrics': self._calculate_css_metrics_dict(content)
            }
        except Exception as e:
            logger.error(f"Error analyzing CSS content {file_path}: {e}")
            return self._create_error_analysis(file_path, str(e))

    def _create_error_analysis(self, file_path: str, error: str) -> Dict[str, Any]:
        """Crée une analyse d'erreur"""
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
        # Pattern amélioré pour gérer les règles imbriquées (SASS/LESS)
        rule_pattern = r'([^{]+)\{([^{}]*?)\}'

        for match in re.finditer(rule_pattern, content):
            selectors = match.group(1).strip()
            declarations = match.group(2).strip()

            rules.append({
                'selectors': [s.strip() for s in selectors.split(',')],
                'declarations': self._parse_declarations(declarations),
                'specificity': self._calculate_specificity(selectors),
                'line_start': self._get_line_number(content, match.start()),
                'line_end': self._get_line_number(content, match.end())
            })

        return rules

    def _parse_declarations(self, declarations: str) -> List[Dict]:
        """Parse les déclarations CSS"""
        parsed = []
        decl_list = [d.strip() for d in declarations.split(';') if d.strip()]

        for decl in decl_list:
            if ':' in decl:
                prop, _, value = decl.partition(':')
                parsed.append({
                    'property': prop.strip(),
                    'value': value.strip(),
                    'important': '!important' in value
                })

        return parsed

    def _extract_selectors(self, content: str) -> List[Dict]:
        """Extrait les sélecteurs CSS"""
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
                    'specificity': self._calculate_specificity(selector),
                    'line_number': self._get_line_number(content, match.start())
                })

        return selectors

    def _classify_selector(self, selector: str) -> str:
        """Classifie le type de sélecteur CSS"""
        if selector.startswith('.'):
            return 'class'
        elif selector.startswith('#'):
            return 'id'
        elif selector.startswith('@'):
            if selector.startswith('@media'):
                return 'media_query'
            elif selector.startswith('@keyframes'):
                return 'keyframes'
            elif selector.startswith('@supports'):
                return 'supports'
            else:
                return 'at_rule'
        elif re.match(r'^[a-zA-Z]', selector):
            return 'element'
        elif selector.startswith('['):
            return 'attribute'
        elif selector.startswith(':'):
            return 'pseudo'
        else:
            return 'complex'

    def _calculate_selector_complexity(self, selector: str) -> int:
        """Calcule la complexité d'un sélecteur"""
        # Nombre de combinateurs
        combinators = len(re.findall(r'[\s>+~]', selector))
        # Nombre de pseudo-classes/elements
        pseudo_elements = len(re.findall(r'::?\w', selector))
        # Nombre de classes/ids
        classes_ids = len(re.findall(r'[.#]', selector))

        return combinators + pseudo_elements + classes_ids

    def _calculate_specificity(self, selector: str) -> List[int]:
        """Calcule la spécificité CSS (a, b, c, d)"""
        # Format: [inline, id, class, element]
        specificity = [0, 0, 0, 0]

        # Compter les IDs
        specificity[1] = len(re.findall(r'#\w', selector))

        # Compter les classes, attributs, pseudo-classes
        classes = len(re.findall(r'\.\w', selector))
        attributes = len(re.findall(r'\[[^\]]+\]', selector))
        pseudo_classes = len(re.findall(r':\w', selector))
        specificity[2] = classes + attributes + pseudo_classes

        # Compter les éléments et pseudo-éléments
        elements = len(re.findall(r'(^|[\s>+~])\w', selector))
        pseudo_elements = len(re.findall(r'::\w', selector))
        specificity[3] = elements + pseudo_elements

        return specificity

    def _extract_properties(self, content: str) -> Dict[str, Any]:
        """Extrait et analyse les propriétés CSS"""
        properties = {}
        prop_pattern = r'([\w-]+)\s*:\s*([^;]+);'

        for match in re.finditer(prop_pattern, content):
            prop = match.group(1).strip()
            value = match.group(2).strip()

            if prop not in properties:
                properties[prop] = {
                    'count': 0,
                    'values': [],
                    'important_count': 0
                }

            properties[prop]['count'] += 1
            properties[prop]['values'].append(value)

            if '!important' in value:
                properties[prop]['important_count'] += 1

        return properties

    def _extract_media_queries(self, content: str) -> List[Dict]:
        """Extrait les media queries"""
        media_queries = []
        media_pattern = r'@media\s*([^{]+)\{([^{}]+)\}'

        for match in re.finditer(media_pattern, content):
            conditions = match.group(1).strip()
            body = match.group(2)

            media_queries.append({
                'conditions': conditions,
                'rules': self._extract_rules(body),
                'rules_count': len(re.findall(r'\{[^}]+\}', body)),
                'features': self._extract_media_features(conditions),
                'line_start': self._get_line_number(content, match.start()),
                'line_end': self._get_line_number(content, match.end())
            })

        return media_queries

    def _extract_media_features(self, conditions: str) -> List[str]:
        """Extrait les features des media queries"""
        features = []
        common_features = ['min-width', 'max-width', 'orientation', 'prefers-color-scheme',
                           'hover', 'pointer', 'aspect-ratio', 'resolution']

        for feature in common_features:
            if feature in conditions:
                features.append(feature)

        return features

    def _extract_keyframes(self, content: str) -> List[Dict]:
        """Extrait les animations @keyframes"""
        keyframes = []
        keyframe_pattern = r'@keyframes\s+(\w+)\s*\{([^{}]+)\}'

        for match in re.finditer(keyframe_pattern, content):
            name = match.group(1)
            body = match.group(2)

            keyframes.append({
                'name': name,
                'steps': self._extract_keyframe_steps(body),
                'line_start': self._get_line_number(content, match.start()),
                'line_end': self._get_line_number(content, match.end())
            })

        return keyframes

    def _extract_keyframe_steps(self, body: str) -> List[Dict]:
        """Extrait les étapes des animations"""
        steps = []
        step_pattern = r'(\d+%|[a-z]+)\s*\{([^}]+)\}'

        for match in re.finditer(step_pattern, body):
            step = match.group(1)
            declarations = match.group(2)

            steps.append({
                'step': step,
                'declarations': self._parse_declarations(declarations)
            })

        return steps

    def _extract_css_variables(self, content: str) -> Dict[str, Any]:
        """Extrait les variables CSS (custom properties)"""
        variables = {'declared': {}, 'used': []}

        # Variables déclarées
        var_decl_pattern = r'--([\w-]+)\s*:\s*([^;]+);'
        for match in re.finditer(var_decl_pattern, content):
            var_name = match.group(1)
            var_value = match.group(2).strip()
            variables['declared'][var_name] = var_value

        # Variables utilisées
        var_use_pattern = r'var\s*\(\s*--([\w-]+)'
        for match in re.finditer(var_use_pattern, content):
            var_name = match.group(1)
            if var_name not in variables['used']:
                variables['used'].append(var_name)

        return variables

    def _analyze_css_patterns(self, rules: List[Dict], selectors: List[Dict], content: str) -> Dict[str, Any]:
        """Analyse les patterns CSS"""
        try:
            selector_types = self._count_selector_types(selectors)
            specificity_values = [self._specificity_to_score(s.get('specificity', [0, 0, 0, 0]))
                                  for s in selectors if s.get('specificity')]

            return {
                'total_rules': len(rules),
                'total_selectors': len(selectors),
                'selector_types': selector_types,
                'common_properties': self._get_common_properties(content),
                'has_animations': self._has_animations(content),
                'has_flexbox': self._has_flexbox(content),
                'has_grid': self._has_grid(content),
                'has_custom_properties': bool(self._extract_css_variables(content)['declared']),
                'has_media_queries': bool(self._extract_media_queries(content)),
                'specificity_analysis': self._analyze_specificity(specificity_values),
                'complexity_score': self._calculate_css_complexity(rules, selectors)
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

    def _specificity_to_score(self, specificity: List[int]) -> int:
        """Convertit la spécificité en score numérique"""
        return specificity[0] * 1000 + specificity[1] * 100 + specificity[2] * 10 + specificity[3]

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
        # Trier par utilisation
        sorted_props = sorted(
            [(prop, data['count']) for prop, data in properties.items()],
            key=lambda x: x[1],
            reverse=True
        )
        return [prop for prop, count in sorted_props[:10]]

    def _has_animations(self, content: str) -> bool:
        """Vérifie la présence d'animations"""
        return '@keyframes' in content or 'animation:' in content

    def _has_flexbox(self, content: str) -> bool:
        """Vérifie la présence de Flexbox"""
        return bool(re.search(r'display:\s*flex', content, re.IGNORECASE))

    def _has_grid(self, content: str) -> bool:
        """Vérifie la présence de Grid"""
        return bool(re.search(r'display:\s*grid', content, re.IGNORECASE))

    def _analyze_specificity(self, specificities: List[int]) -> Dict[str, Any]:
        """Analyse la spécificité des sélecteurs"""
        try:
            if not specificities:
                return {'average': 0, 'max': 0, 'min': 0}

            return {
                'average': sum(specificities) / len(specificities),
                'max': max(specificities),
                'min': min(specificities),
                'distribution': self._calculate_specificity_distribution(specificities)
            }
        except:
            return {'average': 0, 'max': 0, 'min': 0}

    def _calculate_specificity_distribution(self, specificities: List[int]) -> Dict[str, int]:
        """Calcule la distribution de spécificité"""
        distribution = {
            'low': 0,  # 0-10
            'medium': 0,  # 11-100
            'high': 0,  # 101-1000
            'very_high': 0  # 1001+
        }

        for spec in specificities:
            if spec <= 10:
                distribution['low'] += 1
            elif spec <= 100:
                distribution['medium'] += 1
            elif spec <= 1000:
                distribution['high'] += 1
            else:
                distribution['very_high'] += 1

        return distribution

    def _calculate_css_complexity(self, rules: List[Dict], selectors: List[Dict]) -> float:
        """Calcule la complexité CSS"""
        if not rules:
            return 0.0

        # Facteurs de complexité
        avg_selectors_per_rule = len(selectors) / len(rules) if rules else 0
        avg_declarations_per_rule = sum(len(r.get('declarations', [])) for r in rules) / len(rules) if rules else 0
        avg_selector_complexity = sum(s.get('complexity', 0) for s in selectors) / len(selectors) if selectors else 0

        return (avg_selectors_per_rule * 0.3 +
                avg_declarations_per_rule * 0.4 +
                avg_selector_complexity * 0.3)

    def _calculate_css_metrics_dict(self, content: str) -> Dict[str, Any]:
        """Calcule les métriques CSS sous forme de dict"""
        lines = content.split('\n')

        return {
            'total_lines': len(lines),
            'code_lines': len([l for l in lines if l.strip()]),
            'comment_lines': len([l for l in lines if '/*' in l or l.strip().startswith('//')]),
            'blank_lines': len([l for l in lines if not l.strip()]),
            'file_size_bytes': len(content.encode('utf-8'))
        }

    def _calculate_css_metrics(self, content: str) -> FileMetrics:
        """Calcule les métriques CSS pour AnalysisResult"""
        metrics_dict = self._calculate_css_metrics_dict(content)

        return FileMetrics(
            total_lines=metrics_dict['total_lines'],
            code_lines=metrics_dict['code_lines'],
            comment_lines=metrics_dict['comment_lines'],
            blank_lines=metrics_dict['blank_lines'],
            file_size_bytes=metrics_dict['file_size_bytes']
        )

    def _convert_to_code_elements(self, css_analysis: Dict[str, Any]) -> List[CodeElement]:
        """Convertit l'analyse CSS en objets CodeElement"""
        elements = []

        # Règles CSS
        for rule in css_analysis.get('rules', []):
            for selector in rule.get('selectors', []):
                elements.append(CodeElement(
                    name=selector,
                    element_type='css_rule',
                    line_start=rule.get('line_start'),
                    line_end=rule.get('line_end'),
                    metadata={
                        'declarations_count': len(rule.get('declarations', [])),
                        'specificity': rule.get('specificity'),
                        'rule_type': 'css_rule'
                    }
                ))

        # Media Queries
        for media_query in css_analysis.get('media_queries', []):
            elements.append(CodeElement(
                name=f"@media {media_query.get('conditions', '')}",
                element_type='css_media_query',
                line_start=media_query.get('line_start'),
                line_end=media_query.get('line_end'),
                metadata={
                    'rules_count': media_query.get('rules_count', 0),
                    'features': media_query.get('features', []),
                    'rule_type': 'media_query'
                }
            ))

        # Keyframes
        for keyframe in css_analysis.get('keyframes', []):
            elements.append(CodeElement(
                name=f"@keyframes {keyframe.get('name', '')}",
                element_type='css_keyframes',
                line_start=keyframe.get('line_start'),
                line_end=keyframe.get('line_end'),
                metadata={
                    'steps_count': len(keyframe.get('steps', [])),
                    'rule_type': 'keyframes'
                }
            ))

        return elements

    def _detect_css_patterns(self, css_analysis: Dict[str, Any]) -> PatternDetection:
        """Détecte les patterns CSS"""
        patterns = []
        frameworks = []
        libraries = []

        analysis = css_analysis.get('analysis', {})

        # Détecter des patterns CSS
        if analysis.get('has_flexbox'):
            patterns.append('flexbox_layout')

        if analysis.get('has_grid'):
            patterns.append('css_grid')

        if analysis.get('has_animations'):
            patterns.append('css_animations')

        if analysis.get('has_custom_properties'):
            patterns.append('css_variables')

        if analysis.get('has_media_queries'):
            patterns.append('responsive_design')

        # Détecter des préprocesseurs (basé sur l'extension)
        if css_analysis.get('file_path', '').endswith('.scss'):
            patterns.append('scss_preprocessor')
        elif css_analysis.get('file_path', '').endswith('.sass'):
            patterns.append('sass_preprocessor')
        elif css_analysis.get('file_path', '').endswith('.less'):
            patterns.append('less_preprocessor')

        return PatternDetection(
            patterns=patterns,
            frameworks=frameworks,
            libraries=libraries,
            architecture_hints=[]
        )

    def _get_line_number(self, content: str, position: int) -> int:
        """Retourne le numéro de ligne pour une position donnée"""
        return content[:position].count('\n') + 1
