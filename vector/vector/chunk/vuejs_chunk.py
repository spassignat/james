# src/chunk_strategies.py
import re
from typing import List, Dict, Any, Optional
import logging

from vector.chunk.chunk_strategy import ChunkStrategy
from vector.chunk.css_chunk import CSSChunkStrategy
from vector.chunk.html_chunk import HTMLChunkStrategy
from vector.chunk.javascript_chunk import JavaScriptChunkStrategy

logger = logging.getLogger(__name__)

class VueJSChunkStrategy(ChunkStrategy):
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150):
        super().__init__(chunk_size, chunk_overlap)
        # Stratégies de délégation
        self.js_chunk_strategy = JavaScriptChunkStrategy(chunk_size, chunk_overlap)
        self.html_chunk_strategy = HTMLChunkStrategy(chunk_size, chunk_overlap)
        self.css_chunk_strategy = CSSChunkStrategy(chunk_size, chunk_overlap)

    def create_chunks(self, analysis: Dict[str, Any], file_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        chunks = []

        # Détection du type d'API Vue
        api_type = self._detect_vue_api_type(analysis)

        # Chunk template (délégation à HTMLChunkStrategy)
        if analysis.get('template_analysis'):
            template_chunks = self._create_template_chunks(analysis, file_info, api_type)
            chunks.extend(template_chunks)

        # Chunk script selon le type d'API
        if analysis.get('script_analysis'):
            script_chunks = self._create_script_chunks(analysis, file_info, api_type)
            chunks.extend(script_chunks)

        # Chunk style (délégation à CSSChunkStrategy)
        if analysis.get('style_analysis'):
            style_chunks = self._create_style_chunks(analysis, file_info)
            chunks.extend(style_chunks)

        # Chunk composants Vue spécifiques
        component_chunks = self._create_component_chunks(analysis, file_info, api_type)
        chunks.extend(component_chunks)

        # Chunk données globales du composant
        global_chunks = self._create_global_component_chunks(analysis, file_info, api_type)
        chunks.extend(global_chunks)

        return chunks

    def _detect_vue_api_type(self, analysis: Dict[str, Any]) -> str:
        """Détecte le type d'API Vue utilisée"""
        script_analysis = analysis.get('script_analysis', {})
        components = analysis.get('components', [])

        # Détection Composition API
        if any(comp.get('type') == 'composition_api' for comp in components):
            return 'composition'

        if script_analysis.get('imports'):
            imports = ' '.join(script_analysis.get('imports', []))
            if any(keyword in imports for keyword in ['ref', 'reactive', 'computed', 'watch', 'defineComponent']):
                return 'composition'

        # Détection Options API
        if any(comp.get('type') == 'options_api' for comp in components):
            return 'options'

        if script_analysis.get('content'):
            content = script_analysis.get('content', '')
            if any(pattern in content for pattern in ['export default {', 'data()', 'methods:', 'computed:']):
                return 'options'

        # Détection par version Vue
        vue_version = analysis.get('vue_version', '')
        if vue_version == 'vue3':
            return 'composition'  # Par défaut pour Vue 3
        else:
            return 'options'  # Par défaut pour Vue 2

    def _create_template_chunks(self, analysis: Dict, file_info: Dict, api_type: str) -> List[Dict[str, Any]]:
        """Crée des chunks pour le template avec délégation HTML"""
        template_analysis = analysis.get('template_analysis', {})
        chunks = []

        # Chunk principal template
        content = f"VUE TEMPLATE - {api_type.upper()} API: {file_info['filename']}\n"
        content += f"FICHIER: {file_info['relative_path']}\n"
        content += f"TYPE API: {api_type}\n"

        # CORRECTION : Gestion correcte des éléments (peut être dict ou list)
        elements = template_analysis.get('elements', {})
        if elements:
            if isinstance(elements, list):
                # Si c'est une liste de strings
                element_names = [str(elem) for elem in elements[:8]]
                content += f"ÉLÉMENTS PRINCIPAUX: {', '.join(element_names)}\n"
            elif isinstance(elements, dict):
                # Si c'est un dictionnaire, prendre les clés
                element_names = list(elements.keys())[:8]
                content += f"ÉLÉMENTS PRINCIPAUX: {', '.join(element_names)}\n"
            else:
                # Autres types
                content += f"ÉLÉMENTS: {str(elements)[:200]}\n"

        # Caractéristiques spécifiques au template
        features = []
        if template_analysis.get('has_forms'):
            features.append('formulaires')
        if template_analysis.get('has_conditional_rendering'):
            features.append('rendu conditionnel')
        if template_analysis.get('has_loops'):
            features.append('boucles')

        if features:
            content += f"FONCTIONNALITÉS: {', '.join(features)}\n"

        chunks.append({
            'content': content,
            'type': 'vue_template_overview',
            'metadata': {
                'file_path': file_info['path'],
                'api_type': api_type,
                'element_count': self._get_element_count(template_analysis),
                'has_forms': template_analysis.get('has_forms', False)
            }
        })

        # Délégation à HTMLChunkStrategy pour l'analyse détaillée
        if template_analysis.get('content'):
            html_chunks = self.html_chunk_strategy.create_chunks(template_analysis, file_info)
            for chunk in html_chunks:
                chunk['type'] = 'vue_template_' + chunk.get('type', 'html')
                chunk['metadata']['api_type'] = api_type
            chunks.extend(html_chunks)

        return chunks

    def _get_element_count(self, template_analysis: Dict[str, Any]) -> int:
        """Compte le nombre d'éléments de manière sécurisée"""
        elements = template_analysis.get('elements', {})
        if isinstance(elements, list):
            return len(elements)
        elif isinstance(elements, dict):
            return len(elements)
        else:
            return 0

    def _create_script_chunks(self, analysis: Dict, file_info: Dict, api_type: str) -> List[Dict[str, Any]]:
        """Crée des chunks pour le script selon le type d'API"""
        script_analysis = analysis.get('script_analysis', {})
        chunks = []

        # Chunk principal script
        content = f"VUE SCRIPT - {api_type.upper()} API: {file_info['filename']}\n"
        content += f"FICHIER: {file_info['relative_path']}\n"
        content += f"VERSION: {analysis.get('vue_version', 'unknown')}\n"
        content += f"TYPE API: {api_type}\n"

        chunks.append({
            'content': content,
            'type': f'vue_script_{api_type}_overview',
            'metadata': {
                'file_path': file_info['path'],
                'vue_version': analysis.get('vue_version'),
                'api_type': api_type
            }
        })

        # Chunks spécifiques selon l'API
        if api_type == 'composition':
            composition_chunks = self._create_composition_api_chunks(analysis, file_info, script_analysis)
            chunks.extend(composition_chunks)
        else:
            options_chunks = self._create_options_api_chunks(analysis, file_info, script_analysis)
            chunks.extend(options_chunks)

        # Délégation à JavaScriptChunkStrategy pour l'analyse JS générique
        if script_analysis:
            # S'assurer que script_analysis a la structure attendue par JavaScriptChunkStrategy
            normalized_script_analysis = self._normalize_script_analysis(script_analysis)
            js_chunks = self.js_chunk_strategy.create_chunks(normalized_script_analysis, file_info)
            for chunk in js_chunks:
                chunk['type'] = f'vue_script_{api_type}_' + chunk.get('type', 'js')
                chunk['metadata']['api_type'] = api_type
            chunks.extend(js_chunks)

        return chunks

    def _normalize_script_analysis(self, script_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Normalise l'analyse du script pour la délégation JavaScript"""
        normalized = script_analysis.copy()

        # S'assurer que les champs attendus existent
        if 'functions' not in normalized:
            normalized['functions'] = []
        if 'imports' not in normalized:
            normalized['imports'] = []
        if 'variables' not in normalized:
            normalized['variables'] = []

        return normalized

    def _create_composition_api_chunks(self, analysis: Dict, file_info: Dict, script_analysis: Dict) -> List[Dict[str, Any]]:
        """Crée des chunks spécifiques à Composition API"""
        chunks = []
        components = analysis.get('components', [])

        for component in components:
            if component.get('type') == 'composition_api':
                content = f"COMPOSITION API - COMPOSANT: {file_info['filename']}\n"
                content += f"FICHIER: {file_info['relative_path']}\n"

                features = component.get('features', {})

                # CORRECTION : Gestion sécurisée des imports
                imports = features.get('imports', [])
                if imports and isinstance(imports, list):
                    content += f"IMPORTS: {', '.join(imports[:5])}\n"

                if features.get('refs', 0) > 0:
                    content += f"RÉFÉRENCES: {features.get('refs')} ref(s)\n"
                if features.get('reactives', 0) > 0:
                    content += f"RÉACTIFS: {features.get('reactives')} reactive(s)\n"
                if features.get('computed', 0) > 0:
                    content += f"COMPUTED: {features.get('computed')} computed\n"
                if features.get('watchers', 0) > 0:
                    content += f"WATCHERS: {features.get('watchers')} watch\n"

                chunks.append({
                    'content': content,
                    'type': 'vue_composition_api',
                    'metadata': {
                        'file_path': file_info['path'],
                        'refs_count': features.get('refs', 0),
                        'reactives_count': features.get('reactives', 0),
                        'computed_count': features.get('computed', 0),
                        'watchers_count': features.get('watchers', 0)
                    }
                })

        return chunks

    def _create_options_api_chunks(self, analysis: Dict, file_info: Dict, script_analysis: Dict) -> List[Dict[str, Any]]:
        """Crée des chunks spécifiques à Options API"""
        chunks = []
        components = analysis.get('components', [])

        for component in components:
            if component.get('type') == 'options_api':
                content = f"OPTIONS API - COMPOSANT: {file_info['filename']}\n"
                content += f"FICHIER: {file_info['relative_path']}\n"

                features = component.get('features', {})

                # Sections de l'Options API
                sections = []
                if features.get('data'):
                    sections.append('data')
                if features.get('methods'):
                    sections.append('methods')
                if features.get('computed'):
                    sections.append('computed')
                if features.get('props'):
                    sections.append('props')
                if features.get('emits'):
                    sections.append('emits')
                if features.get('lifecycle'):
                    sections.append('lifecycle')

                if sections:
                    content += f"SECTIONS: {', '.join(sections)}\n"

                chunks.append({
                    'content': content,
                    'type': 'vue_options_api',
                    'metadata': {
                        'file_path': file_info['path'],
                        'sections': sections,
                        'has_data': features.get('data', False),
                        'has_methods': features.get('methods', False),
                        'has_computed': features.get('computed', False)
                    }
                })

        return chunks

    def _create_style_chunks(self, analysis: Dict, file_info: Dict) -> List[Dict[str, Any]]:
        """Crée des chunks pour les styles avec délégation CSS"""
        style_analysis = analysis.get('style_analysis', {})
        chunks = []

        # Chunk principal style
        content = f"VUE STYLES: {file_info['filename']}\n"
        content += f"FICHIER: {file_info['relative_path']}\n"

        if style_analysis.get('preprocessors'):
            content += f"PRÉPROCESSEUR: {style_analysis.get('preprocessors')}\n"

        if style_analysis.get('scoped'):
            content += "SCOPE: Styles scopés\n"

        chunks.append({
            'content': content,
            'type': 'vue_style_overview',
            'metadata': {
                'file_path': file_info['path'],
                'scoped': style_analysis.get('scoped', False),
                'preprocessor': style_analysis.get('preprocessors')
            }
        })

        # Délégation à CSSChunkStrategy pour l'analyse détaillée
        if style_analysis:
            normalized_style_analysis = self._normalize_style_analysis(style_analysis)
            css_chunks = self.css_chunk_strategy.create_chunks(normalized_style_analysis, file_info)
            for chunk in css_chunks:
                chunk['type'] = 'vue_style_' + chunk.get('type', 'css')
            chunks.extend(css_chunks)

        return chunks

    def _normalize_style_analysis(self, style_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Normalise l'analyse des styles pour la délégation CSS"""
        normalized = style_analysis.copy()

        # S'assurer que les champs attendus existent
        if 'rules' not in normalized:
            normalized['rules'] = []
        if 'selectors' not in normalized:
            normalized['selectors'] = []

        return normalized

    def _create_component_chunks(self, analysis: Dict, file_info: Dict, api_type: str) -> List[Dict[str, Any]]:
        """Crée des chunks pour les composables et utilitaires"""
        chunks = []

        # Composables (Vue 3)
        composables = analysis.get('composables', [])
        if composables and isinstance(composables, list):
            for composable in composables[:5]:  # Limiter aux 5 premiers
                content = f"COMPOSABLE VUE: {composable}\n"
                content += f"FICHIER: {file_info['relative_path']}\n"
                content += f"TYPE API: {api_type}\n"

                chunks.append({
                    'content': content,
                    'type': 'vue_composable',
                    'metadata': {
                        'file_path': file_info['path'],
                        'composable_name': composable,
                        'api_type': api_type
                    }
                })

        # Directives personnalisées
        directives = analysis.get('directives', [])
        if directives and isinstance(directives, list):
            for directive in directives[:3]:
                content = f"DIRECTIVE VUE: {directive}\n"
                content += f"FICHIER: {file_info['relative_path']}\n"

                chunks.append({
                    'content': content,
                    'type': 'vue_directive',
                    'metadata': {
                        'file_path': file_info['path'],
                        'directive_name': directive
                    }
                })

        return chunks

    def _create_global_component_chunks(self, analysis: Dict, file_info: Dict, api_type: str) -> List[Dict[str, Any]]:
        """Crée des chunks pour les données globales du composant"""
        chunks = []

        # Chunk métadonnées du composant
        content = f"COMPOSANT VUE GLOBAL: {file_info['filename']}\n"
        content += f"FICHIER: {file_info['relative_path']}\n"
        content += f"VERSION: {analysis.get('vue_version', 'unknown')}\n"
        content += f"TYPE API: {api_type}\n"

        # Sections détectées
        sections = analysis.get('sections', {})
        if sections.get('has_template'):
            content += "SECTION: Template\n"
        if sections.get('has_script'):
            content += "SECTION: Script\n"
        if sections.get('has_style'):
            content += "SECTION: Style\n"
        if sections.get('has_setup'):
            content += "SCRIPT: Setup (Composition API)\n"

        chunks.append({
            'content': content,
            'type': 'vue_component_global',
            'metadata': {
                'file_path': file_info['path'],
                'vue_version': analysis.get('vue_version'),
                'api_type': api_type,
                'has_template': sections.get('has_template', False),
                'has_script': sections.get('has_script', False),
                'has_style': sections.get('has_style', False),
                'has_setup': sections.get('has_setup', False)
            }
        })

        return chunks