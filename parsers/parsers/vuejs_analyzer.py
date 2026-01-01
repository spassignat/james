# james_parsers/vuejs_analyzer.py
import re
import logging
from typing import Dict, List, Any
from .analyzer import Analyzer

logger = logging.getLogger(__name__)

class VueJSAnalyzer(Analyzer):
    def __init__(self):
        # Import des analyseurs de base (éviter les imports circulaires)
        from .javascript_analyzer import JavaScriptAnalyzer
        from .html_analyzer import HTMLAnalyzer
        from .css_analyzer import CSSAnalyzer

        self.js_analyzer = JavaScriptAnalyzer()
        self.html_analyzer = HTMLAnalyzer()
        self.css_analyzer = CSSAnalyzer()

    def analyze(self, file_path: str) -> Dict[str, Any]:
        """Analyse un fichier Vue.js complet"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            return self.analyze_content(content, file_path)

        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return {
                'file_type': 'vue',
                'file_path': file_path,
                'error': str(e),
                'sections': {},
                'components': []
            }

    def analyze_content(self, content: str, file_path: str = "") -> Dict[str, Any]:
        """Analyse le contenu d'un fichier Vue.js"""
        try:
            analysis = {
                'file_type': 'vue',
                'file_path': file_path,
                'vue_version': self._detect_vue_version(content),
                'sections': self._extract_sections(content),
                'components': self._analyze_components(content),
                'composables': self._extract_composables(content),
                'directives': self._extract_directives(content),
                'stores': self._extract_stores(content)
            }

            # Analyse spécifique par section avec gestion d'erreurs robuste
            template_content = self._extract_template(content)
            if template_content:
                analysis['template_analysis'] = self._safe_analyze_template(template_content, file_path)

            script_content = self._extract_script(content)
            if script_content:
                analysis['script_analysis'] = self._safe_analyze_script(script_content, file_path)

            style_content = self._extract_style(content)
            if style_content:
                analysis['style_analysis'] = self._safe_analyze_style(style_content, file_path)

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing Vue.js content {file_path}: {e}")
            return {
                'file_type': 'vue',
                'file_path': file_path,
                'error': str(e),
                'sections': {},
                'components': []
            }

    def _safe_analyze_template(self, template_content: str, file_path: str) -> Dict[str, Any]:
        """Analyse sécurisée du template"""
        try:
            return self.html_analyzer.analyze_content(template_content, file_path)
        except Exception as e:
            logger.warning(f"Template analysis failed for {file_path}: {e}")
            return {
                'file_type': 'html',
                'content': template_content[:1000],
                'error': str(e),
                'elements': [],
                'has_forms': False,
                'has_tables': False
            }

    def _safe_analyze_script(self, script_content: str, file_path: str) -> Dict[str, Any]:
        """Analyse sécurisée du script"""
        try:
            return self.js_analyzer.analyze_content(script_content, file_path)
        except Exception as e:
            logger.warning(f"Script analysis failed for {file_path}: {e}")
            return {
                'file_type': 'javascript',
                'content': script_content[:1000],
                'error': str(e),
                'functions': [],
                'imports': [],
                'exports': []
            }

    def _safe_analyze_style(self, style_content: str, file_path: str) -> Dict[str, Any]:
        """Analyse sécurisée des styles"""
        try:
            return self.css_analyzer.analyze_content(style_content, file_path)
        except Exception as e:
            logger.warning(f"Style analysis failed for {file_path}: {e}")
            return {
                'file_type': 'css',
                'content': style_content[:1000],
                'error': str(e),
                'rules': [],
                'selectors': [],
                'properties': {},
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

    def _detect_vue_version(self, content: str) -> str:
        """Détecte la version de Vue.js"""
        if '<script setup>' in content or 'defineComponent' in content:
            return 'vue3'
        elif 'export default' in content:
            return 'vue2'
        return 'unknown'

    def _extract_sections(self, content: str) -> Dict[str, bool]:
        """Extrait les sections présentes dans le fichier"""
        return {
            'has_template': '<template>' in content,
            'has_script': '<script' in content,
            'has_style': '<style' in content,
            'has_setup': '<script setup>' in content
        }

    def _extract_template(self, content: str) -> str:
        """Extrait le contenu du template"""
        match = re.search(r'<template>([\s\S]*?)</template>', content)
        return match.group(1).strip() if match else ""

    def _extract_script(self, content: str) -> str:
        """Extrait le contenu du script"""
        match = re.search(r'<script[^>]*>([\s\S]*?)</script>', content)
        return match.group(1).strip() if match else ""

    def _extract_style(self, content: str) -> str:
        """Extrait le contenu des styles"""
        match = re.search(r'<style[^>]*>([\s\S]*?)</style>', content)
        return match.group(1).strip() if match else ""

    def _analyze_components(self, content: str) -> List[Dict]:
        """Analyse les composants Vue"""
        components = []

        # Composition API
        if '<script setup>' in content:
            components.append({
                'type': 'composition_api',
                'features': self._extract_composition_features(content)
            })

        # Options API
        if 'export default' in content:
            options_match = re.search(r'export\s+default\s*{([\s\S]*?)}', content)
            if options_match:
                options_content = options_match.group(1)
                components.append({
                    'type': 'options_api',
                    'features': self._extract_options_features(options_content)
                })

        return components

    def _extract_composition_features(self, content: str) -> Dict[str, Any]:
        """Extrait les features de Composition API"""
        return {
            'imports': re.findall(r'import\s+{([^}]+)}', content),
            'refs': len(re.findall(r'ref\(', content)),
            'reactives': len(re.findall(r'reactive\(', content)),
            'computed': len(re.findall(r'computed\(', content)),
            'watchers': len(re.findall(r'watch\(', content))
        }

    def _extract_options_features(self, content: str) -> Dict[str, bool]:
        """Extrait les features de Options API"""
        return {
            'data': 'data()' in content,
            'methods': 'methods:' in content,
            'computed': 'computed:' in content,
            'props': 'props:' in content,
            'emits': 'emits:' in content,
            'lifecycle': any(hook in content for hook in ['mounted()', 'created()', 'beforeMount()'])
        }

    def _extract_composables(self, content: str) -> List[str]:
        """Extrait les composables (Vue 3)"""
        composables = re.findall(r'const\s+use(\w+)\s*=', content)
        return composables

    def _extract_directives(self, content: str) -> List[str]:
        """Extrait les directives personnalisées"""
        directives = re.findall(r'directives\s*:\s*{([^}]+)}', content)
        return directives

    def _extract_stores(self, content: str) -> List[str]:
        """Extrait les stores (Pinia, Vuex)"""
        stores = []
        if 'defineStore' in content:
            stores.extend(re.findall(r'defineStore\s*\(\s*[\'"]([^\'"]+)[\'"]', content))
        return stores