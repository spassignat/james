# parsers/vuejs_analyzer.py
import logging
from typing import Dict, Any, List, Optional

import time

from parsers.analyzer import Analyzer
from .css_analyzer import CSSAnalyzer
from .html_analyzer import HTMLAnalyzer
from .javascript_analyzer import JavaScriptAnalyzer
from ..analysis_result import AnalysisResult, SectionAnalysis, CodeElement, FileMetrics, FileType

logger = logging.getLogger(__name__)


class VueJSAnalyzer(Analyzer):
    file_type = FileType.VUE
    js_analyzer: JavaScriptAnalyzer
    html_analyzer: HTMLAnalyzer
    css_analyzer: CSSAnalyzer

    def __init__(self):
        super().__init__()
        self.file_type = FileType.VUE
        self.js_analyzer = JavaScriptAnalyzer()
        self.html_analyzer = HTMLAnalyzer()
        self.css_analyzer = CSSAnalyzer()

    def analyze_file(self, file_path: str) -> AnalysisResult:
        """Analyse un fichier Vue.js"""
        start_time = time.time() * 1000

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            result = self._analyze_vue_content(content, file_path)
            result.processing_time_ms = int(time.time() * 1000 - start_time)
            return result

        except Exception as e:
            logger.error(f"Erreur analyse Vue.js {file_path}: {e}")
            result = self._create_error_result(file_path, str(e))
            result.processing_time_ms = int(time.time() * 1000 - start_time)
            return result

    def _analyze_vue_content(self, content: str, file_path: str) -> AnalysisResult:
        """Analyse le contenu d'un fichier Vue.js"""
        result = self._create_base_result(file_path)
        result.language_specific = {}

        # Extraire les sections
        sections = self._extract_sections(content)
        result.sections = self._analyze_sections(sections, file_path)

        # Analyser les composants Vue
        vue_analysis = self.analyze_content(content, file_path)
        result.language_specific.update(vue_analysis)

        # Combiner les éléments de toutes les sections
        all_elements = []
        for section_name, section in result.sections.items():
            all_elements.extend(section.elements)

        result.elements = all_elements

        # Calculer les métriques combinées
        result.metrics = self._combine_section_metrics(result.sections)

        return result

    def _extract_sections(self, content: str) -> Dict[str, str]:
        """Extrait les sections du fichier Vue"""
        sections = {}

        # Extraire template
        import re
        template_match = re.search(r'<template>([\s\S]*?)</template>', content)
        if template_match:
            sections['template'] = template_match.group(1).strip()

        # Extraire script
        script_match = re.search(r'<script[^>]*>([\s\S]*?)</script>', content)
        if script_match:
            sections['script'] = script_match.group(1).strip()

        # Extraire style
        style_match = re.search(r'<style[^>]*>([\s\S]*?)</style>', content)
        if style_match:
            sections['style'] = style_match.group(1).strip()

        return sections

    def _analyze_sections(self, sections: Dict[str, str], file_path: str) -> Dict[str, SectionAnalysis]:
        """Analyse chaque section avec l'analyseur approprié"""
        section_results = {}

        # Analyser le template
        if 'template' in sections:
            html_analysis = self.html_analyzer.analyze_content(sections['template'], file_path)
            section_results['template'] = SectionAnalysis(
                section_type='template',
                content=sections['template'][:1000],  # Limiter la taille
                language='html',
                elements=self._convert_html_elements(html_analysis),
                metrics=self.html_analyzer._calculate_metrics(sections['template']),
                analysis=html_analysis.get('analysis', {})
            )

        # Analyser le script
        if 'script' in sections:
            js_analysis = self.js_analyzer.analyze_content(sections['script'], file_path)
            section_results['script'] = SectionAnalysis(
                section_type='script',
                content=sections['script'][:1000],
                language='javascript',
                elements=self.js_analyzer._convert_to_code_elements(js_analysis),
                metrics=self.js_analyzer._calculate_metrics(sections['script']),
                analysis=js_analysis.get('analysis', {})
            )

        # Analyser le style
        if 'style' in sections:
            css_analysis = self.css_analyzer.analyze_content(sections['style'], file_path)
            section_results['style'] = SectionAnalysis(
                section_type='style',
                content=sections['style'][:1000],
                language='css',
                metrics=self.css_analyzer._calculate_metrics(sections['style']),
                analysis=css_analysis.get('analysis', {})
            )

        return section_results

    def _convert_html_elements(self, html_analysis: Dict[str, Any]) -> List[CodeElement]:
        """Convertit l'analyse HTML en CodeElements"""
        elements = []

        for element in html_analysis.get('elements', []):
            elements.append(CodeElement(
                name=element.get('tag', ''),
                element_type='html_element',
                metadata={
                    'attributes': element.get('attributes', {}),
                    'has_children': element.get('has_children', False)
                }
            ))

        return elements

    def _combine_section_metrics(self, sections: Dict[str, SectionAnalysis]) -> FileMetrics:
        """Combine les métriques de toutes les sections"""
        total_lines = sum(s.metrics.total_lines for s in sections.values())
        code_lines = sum(s.metrics.code_lines for s in sections.values())

        return FileMetrics(
            total_lines=total_lines,
            code_lines=code_lines,
            comment_lines=sum(s.metrics.comment_lines for s in sections.values()),
            blank_lines=sum(s.metrics.blank_lines for s in sections.values()),
            file_size_bytes=sum(s.metrics.file_size_bytes for s in sections.values())
        )

    # Garder votre méthode analyze_content existante pour la compatibilité
    def analyze_content(self, content: str, file_path: str = "") -> Dict[str, Any]:
        """Méthode existante pour la compatibilité"""
        # Votre code existant ici...
        return {
            'vue_version': self._detect_vue_version(content),
            'sections': self._extract_sections_dict(content),
            'components': self._analyze_components(content),
            'composables': self._extract_composables(content)
        }

    # parsers/vuejs_analyzer.py (fonctions manquantes)

    def _detect_vue_version(self, content: str) -> str:
        """Détecte la version de Vue.js"""
        # Vue 3 avec Composition API et <script setup>
        if '<script setup>' in content or 'defineComponent' in content or 'createApp' in content:
            return 'vue3'
        # Vue 2 avec Options API
        elif 'export default' in content and 'Vue.component' not in content:
            return 'vue2'
        # Vue 2 avec API globale
        elif 'Vue.component' in content or 'new Vue(' in content:
            return 'vue2_legacy'
        return 'unknown'

    def _extract_sections_dict(self, content: str) -> Dict[str, Any]:
        """Extrait les sections et leurs métadonnées en format dict"""
        template_section = self._extract_template_section(content)
        script_section = self._extract_script_section(content)
        style_sections = self._extract_style_sections(content)

        return {
            'template': {
                'present': bool(template_section['content']),
                'attributes': template_section['attributes'],
                'length': template_section['length']
            },
            'script': {
                'present': bool(script_section['content']),
                'attributes': script_section['attributes'],
                'setup': script_section['setup'],
                'language': script_section['language'],
                'length': script_section['length']
            },
            'styles': {
                'present': len(style_sections) > 0,
                'count': len(style_sections),
                'sections': [{
                    'language': style['language'],
                    'scoped': style['scoped'],
                    'module': style['module'],
                    'length': style['length']
                } for style in style_sections]
            },
            'vue_version': self._detect_vue_version(content),
            'has_setup_script': script_section['setup'],
            'has_scoped_styles': any(style['scoped'] for style in style_sections)
        }

    def _extract_template_section(self, content: str) -> Dict[str, Any]:
        """Extrait la section template avec ses attributs"""
        import re

        # Chercher le template avec ses attributs
        template_match = re.search(r'<template([^>]*)>([\s\S]*?)</template>', content)

        if not template_match:
            return {'content': '', 'attributes': {}}

        attrs_str = template_match.group(1)
        template_content = template_match.group(2).strip()

        # Parser les attributs
        attributes = {}
        if attrs_str:
            # Extraire les attributs simples
            attr_pattern = r'(\w+)(?:\s*=\s*["\']([^"\']*)["\'])?'
            for match in re.finditer(attr_pattern, attrs_str):
                attr_name = match.group(1)
                attr_value = match.group(2) if match.group(2) else True
                attributes[attr_name] = attr_value

        return {
            'content': template_content,
            'attributes': attributes,
            'length': len(template_content)
        }

    def _extract_script_section(self, content: str) -> Dict[str, Any]:
        """Extrait la section script avec ses attributs"""
        import re

        # Chercher toutes les sections script
        script_matches = list(re.finditer(r'<script([^>]*)>([\s\S]*?)</script>', content))

        if not script_matches:
            return {'content': '', 'attributes': {}, 'setup': False}

        # Prendre la première section script (généralement la principale)
        script_match = script_matches[0]
        attrs_str = script_match.group(1)
        script_content = script_match.group(2).strip()

        # Parser les attributs
        attributes = {}
        setup = False
        lang = 'javascript'

        if attrs_str:
            # Extraire les attributs
            attr_pattern = r'(\w+)(?:\s*=\s*["\']([^"\']*)["\'])?'
            for match in re.finditer(attr_pattern, attrs_str):
                attr_name = match.group(1).strip()
                attr_value = match.group(2).strip() if match.group(2) else True

                if attr_name == 'setup':
                    setup = True
                elif attr_name == 'lang':
                    lang = attr_value

                attributes[attr_name] = attr_value

        # Vérifier si c'est un script setup
        if not setup and 'setup' in attrs_str:
            setup = True

        return {
            'content': script_content,
            'attributes': attributes,
            'setup': setup,
            'language': lang,
            'length': len(script_content)
        }

    def _extract_style_sections(self, content: str) -> List[Dict[str, Any]]:
        """Extrait toutes les sections style"""
        import re

        styles = []
        style_matches = re.finditer(r'<style([^>]*)>([\s\S]*?)</style>', content)

        for match in style_matches:
            attrs_str = match.group(1)
            style_content = match.group(2).strip()

            # Parser les attributs
            attributes = {}
            scoped = False
            module = False
            lang = 'css'

            if attrs_str:
                attr_pattern = r'(\w+)(?:\s*=\s*["\']([^"\']*)["\'])?'
                for attr_match in re.finditer(attr_pattern, attrs_str):
                    attr_name = attr_match.group(1).strip()
                    attr_value = attr_match.group(2).strip() if attr_match.group(2) else True

                    if attr_name == 'scoped':
                        scoped = True
                    elif attr_name == 'module':
                        module = True
                    elif attr_name == 'lang':
                        lang = attr_value

                    attributes[attr_name] = attr_value

            styles.append({
                'content': style_content,
                'attributes': attributes,
                'scoped': scoped,
                'module': module,
                'language': lang,
                'length': len(style_content)
            })

        return styles

    def _analyze_components(self, content: str) -> List[Dict[str, Any]]:
        """Analyse les composants Vue dans le fichier"""
        import re
        components = []

        # Détecter le type de composant
        vue_version = self._detect_vue_version(content)

        if vue_version.startswith('vue3'):
            # Composition API
            components.extend(self._analyze_composition_api(content))

            # Options API avec defineComponent
            if 'defineComponent' in content:
                components.extend(self._analyze_define_component(content))

        elif vue_version.startswith('vue2'):
            # Options API classique
            components.extend(self._analyze_options_api(content))

        # Détecter les composants globaux (Vue 2)
        global_components = re.findall(r'Vue\.component\s*\(\s*[\'"]([^\'"]+)[\'"]', content)
        for comp_name in global_components:
            components.append({
                'name': comp_name,
                'type': 'global_component',
                'api': 'vue2_global',
                'registration': 'global'
            })

        # Détecter les composants asynchrones
        async_components = re.findall(r'defineAsyncComponent\s*\(', content)
        if async_components:
            components.append({
                'name': 'async_component',
                'type': 'async_component',
                'api': 'vue3',
                'is_async': True
            })

        return components

    def _analyze_composition_api(self, content: str) -> List[Dict[str, Any]]:
        """Analyse les composants avec Composition API"""
        import re
        components = []

        # Script setup simple
        if '<script setup>' in content:
            # Extraire le nom du composant du fichier ou des commentaires
            component_name = self._extract_component_name(content)

            components.append({
                'name': component_name,
                'type': 'composition_api',
                'api': 'vue3_composition',
                'setup': True,
                'features': self._extract_setup_features(content)
            })

        # Composition API sans setup (avec setup() function)
        setup_func_match = re.search(r'setup\s*\(([^)]*)\)\s*{', content)
        if setup_func_match:
            component_name = self._extract_component_name(content)

            components.append({
                'name': component_name,
                'type': 'composition_api',
                'api': 'vue3_composition',
                'setup': False,
                'features': self._extract_composition_features(content)
            })

        return components

    def _analyze_define_component(self, content: str) -> List[Dict[str, Any]]:
        """Analyse les composants avec defineComponent"""
        import re
        components = []

        # Chercher defineComponent calls
        define_comp_pattern = r'defineComponent\s*\(([\s\S]*?)\)\s*(?:;|$)'

        for match in re.finditer(define_comp_pattern, content):
            component_content = match.group(1)

            # Essayer d'extraire le nom
            component_name = self._extract_component_name(content)

            # Analyser les options
            options = self._analyze_component_options(component_content)

            components.append({
                'name': component_name,
                'type': 'options_api',
                'api': 'vue3_options',
                'options': options
            })

        return components

    def _analyze_options_api(self, content: str) -> List[Dict[str, Any]]:
        """Analyse les composants avec Options API (Vue 2)"""
        import re
        components = []

        # Chercher export default { ... }
        export_default_pattern = r'export\s+default\s*{([\s\S]*?)\n}'

        for match in re.finditer(export_default_pattern, content):
            options_content = match.group(1)
            component_name = self._extract_component_name(content)

            # Analyser les options
            options = self._analyze_component_options(options_content)

            components.append({
                'name': component_name,
                'type': 'options_api',
                'api': 'vue2_options',
                'options': options
            })

        return components

    def _extract_component_name(self, content: str) -> str:
        """Extrait le nom du composant"""
        import re

        # Chercher dans les commentaires
        comment_pattern = r'//\s*@component\s+(\w+)|/\*\s*@component\s+(\w+)\s*\*/'
        comment_match = re.search(comment_pattern, content)
        if comment_match:
            return comment_match.group(1) or comment_match.group(2)

        # Chercher le nom dans le script
        name_patterns = [
            r'name\s*:\s*[\'"]([^\'"]+)[\'"]',
            r'"name"\s*:\s*[\'"]([^\'"]+)[\'"]',
            r"'name'\s*:\s*[\'\"]([^\\\'\"]+)[\'\"]"
        ]

        for pattern in name_patterns:
            name_match = re.search(pattern, content)
        if name_match:
            return name_match.group(1)

        # Retourner un nom par défaut
        return 'AnonymousComponent'

    def _extract_setup_features(self, content: str) -> Dict[str, Any]:
        """Extrait les features d'un script setup"""
        import re

        features = {
            'imports': [],
            'refs': [],
            'reactives': [],
            'computed': [],
            'watchers': [],
            'props': [],
            'emits': [],
            'slots': [],
            'exposes': []
        }

        # Extraire les imports
        import_matches = re.findall(r'import\s+{([^}]+)}\s+from', content)
        for match in import_matches:
            features['imports'].extend([imp.strip() for imp in match.split(',')])

        # Compter les refs
        ref_matches = re.findall(r'\b(\w+)\s*=\s*ref\(', content)
        features['refs'] = ref_matches

        # Compter les reactives
        reactive_matches = re.findall(r'\b(\w+)\s*=\s*reactive\(', content)
        features['reactives'] = reactive_matches

        # Compter les computed
        computed_matches = re.findall(r'\b(\w+)\s*=\s*computed\(', content)
        features['computed'] = computed_matches

        # Détecter les props avec defineProps
        if 'defineProps' in content:
            props_match = re.search(r'defineProps\s*\(([^)]*)\)', content)
            if props_match:
                features['props'] = self._parse_props_definition(props_match.group(1))

        # Détecter les emits avec defineEmits
        if 'defineEmits' in content:
            features['emits'] = True

        # Détecter les slots avec useSlots
        if 'useSlots' in content:
            features['slots'] = True

        # Détecter les exposes avec defineExpose
        if 'defineExpose' in content:
            features['exposes'] = True

        return features

    def _extract_composition_features(self, content: str) -> Dict[str, Any]:
        """Extrait les features de Composition API (sans setup)"""

        features = {
            'lifecycle_hooks': [],
            'provide_inject': False,
            'template_refs': False
        }

        # Détecter les lifecycle hooks
        lifecycle_hooks = [
            'onBeforeMount', 'onMounted', 'onBeforeUpdate', 'onUpdated',
            'onBeforeUnmount', 'onUnmounted', 'onErrorCaptured', 'onRenderTracked',
            'onRenderTriggered', 'onActivated', 'onDeactivated'
        ]

        for hook in lifecycle_hooks:
            if hook in content:
                features['lifecycle_hooks'].append(hook)

        # Détecter provide/inject
        if 'provide(' in content or 'inject(' in content:
            features['provide_inject'] = True

        # Détecter les template refs
        if 'templateRef' in content or 'ref(' in content:
            features['template_refs'] = True

        return features

    def _analyze_component_options(self, options_content: str) -> Dict[str, Any]:
        """Analyse les options d'un composant"""
        import re

        options = {
            'has_data': False,
            'has_props': False,
            'has_computed': False,
            'has_watch': False,
            'has_methods': False,
            'has_lifecycle': False,
            'has_components': False,
            'has_directives': False,
            'has_filters': False,
            'has_mixins': False
        }

        # Vérifier chaque option
        option_patterns = {
            'has_data': r'data\s*\(\)\s*:|data\s*:\s*{',
            'has_props': r'props\s*:\s*{',
            'has_computed': r'computed\s*:\s*{',
            'has_watch': r'watch\s*:\s*{',
            'has_methods': r'methods\s*:\s*{',
            'has_lifecycle': r'(?:beforeCreate|created|beforeMount|mounted|beforeUpdate|updated|beforeDestroy|destroyed|activated|deactivated|errorCaptured)\s*\(\)',
            'has_components': r'components\s*:\s*{',
            'has_directives': r'directives\s*:\s*{',
            'has_filters': r'filters\s*:\s*{',
            'has_mixins': r'mixins\s*:\s*\['
        }

        for option, pattern in option_patterns.items():
            if re.search(pattern, options_content):
                options[option] = True

        return options

    def _parse_props_definition(self, props_content: str) -> List[Dict[str, Any]]:
        """Parse une définition de props"""
        import re
        import json

        props = []

        # Essayer de parser comme un objet TypeScript/JavaScript
        try:
            # Nettoyer le contenu
            clean_content = props_content.strip()

            # Si c'est un objet TypeScript avec interface
            if clean_content.startswith('<'):
                # Extraire le type générique
                type_match = re.search(r'<([^>]+)>', clean_content)
                if type_match:
                    props.append({
                        'type': type_match.group(1),
                        'definition': 'typescript_generic'
                    })

            # Si c'est un objet littéral
            elif clean_content.startswith('{'):
                # Essayer de parser comme JSON (simplifié)
                # Remplace les clés non citées par des clés citées
                json_like = re.sub(r'(\w+)\s*:', r'"\1":', clean_content)
                json_like = re.sub(r':\s*(\w+)(?=[,}])', r': "\1"', json_like)

                try:
                    props_obj = json.loads(json_like)
                    for prop_name, prop_def in props_obj.items():
                        if isinstance(prop_def, dict):
                            props.append({
                                'name': prop_name,
                                'type': prop_def.get('type', 'any'),
                                'required': prop_def.get('required', False),
                                'default': prop_def.get('default')
                            })
                        else:
                            props.append({
                                'name': prop_name,
                                'type': str(prop_def),
                                'required': False
                            })
                except:
                    # Fallback: extraction simple
                    prop_matches = re.findall(r'"(\w+)"\s*:\s*{([^}]+)}', clean_content)
                    for prop_name, prop_def in prop_matches:
                        props.append({
                            'name': prop_name,
                            'definition': prop_def.strip()
                        })

        except Exception as e:
            # En cas d'erreur, retourner une analyse basique
            props.append({
                'error': str(e),
                'raw_content': props_content[:200]
            })

        return props

    def _extract_composables(self, content: str) -> List[Dict[str, Any]]:
        """Extrait les composables (hooks Vue 3)"""
        import re

        composables = []

        # Pattern pour les fonctions useXXX (composables)
        composable_pattern = r'(?:export\s+)?(?:function|const)\s+(use[A-Z][a-zA-Z0-9]*)\s*[=(]'

        for match in re.finditer(composable_pattern, content):
            composable_name = match.group(1)

            # Trouver le corps de la fonction
            func_start = match.end()
            func_body = self._extract_function_body(content, func_start)

            composables.append({
                'name': composable_name,
                'type': 'composable',
                'length': len(func_body) if func_body else 0,
                'has_return': 'return ' in func_body if func_body else False
            })

        return composables

    def _extract_function_body(self, content: str, start_pos: int) -> Optional[str]:
        """Extrait le corps d'une fonction à partir d'une position"""
        brace_count = 0
        body_start = -1

        for i in range(start_pos, len(content)):
            char = content[i]

            if char == '{':
                if brace_count == 0:
                    body_start = i + 1
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and body_start != -1:
                    return content[body_start:i]

            if brace_count < 0:
                break

        return None

    def _extract_directives(self, content: str) -> List[Dict[str, Any]]:
        """Extrait les directives personnalisées"""
        import re

        directives = []

        # Directives dans les options
        directives_match = re.search(r'directives\s*:\s*{([^}]+)}', content)
        if directives_match:
            directives_content = directives_match.group(1)
            # Extraire les noms de directives
            directive_names = re.findall(r'"(\w+)"\s*:\s*{', directives_content)
            for name in directive_names:
                directives.append({
                    'name': name,
                    'type': 'local_directive',
                    'scope': 'component'
                })

        # Directives globales (Vue 2)
        global_directives = re.findall(r'Vue\.directive\s*\(\s*[\'"]([^\'"]+)[\'"]', content)
        for name in global_directives:
            directives.append({
                'name': name,
                'type': 'global_directive',
                'scope': 'global'
            })

        # Directives avec v-prefix dans le template
        if '<template>' in content:
            template_content = self._extract_template_section(content)['content']
            directive_usage = re.findall(r'v-([a-zA-Z][a-zA-Z0-9\-]*)', template_content)
            for directive in directive_usage:
                if directive not in ['for', 'if', 'else', 'else-if', 'show', 'model', 'bind', 'on', 'html', 'text',
                                     'cloak', 'once', 'pre', 'memo']:
                    directives.append({
                        'name': directive,
                        'type': 'custom_directive_usage',
                        'scope': 'template'
                    })

        return directives

    def _extract_stores(self, content: str) -> List[Dict[str, Any]]:
        """Extrait les stores (Pinia, Vuex)"""
        import re

        stores = []

        # Pinia stores
        if 'defineStore' in content:
            store_matches = re.finditer(r'defineStore\s*\(\s*[\'"]([^\'"]+)[\'"]\s*,', content)
            for match in store_matches:
                store_name = match.group(1)

                stores.append({
                    'name': store_name,
                    'type': 'pinia_store',
                    'library': 'pinia'
                })

        # Vuex stores
        vuex_patterns = [
            r'new\s+Vuex\.Store\s*\(',
            r'createStore\s*\(',
            r'useStore\s*\('
        ]

        for pattern in vuex_patterns:
            if re.search(pattern, content):
                stores.append({
                    'type': 'vuex_store',
                    'library': 'vuex',
                    'detected_by': pattern
                })

        return stores
