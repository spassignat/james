# json_analyzer.py
import json
import logging
# Nécessaire pour os.path.commonprefix
import os
import re
from datetime import datetime
from typing import Dict, Any, List

from parsers.analysis_result import (
    AnalysisResult, AnalysisStatus, FileType, FrameworkType,
    CodeElement, FileMetrics
)
from parsers.analyzer import Analyzer

logger = logging.getLogger(__name__)


class JSONAnalyzer(Analyzer):
    """Analyseur de fichiers JSON retournant des AnalysisResult"""

    def __init__(self):
        super().__init__()
        self.file_type = FileType.JSON

    def analyze_file(self, file_path: str) -> AnalysisResult:
        """Analyse un fichier JSON et retourne un AnalysisResult"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Mesurer le temps de traitement
            start_time = datetime.now()

            # Créer le résultat de base
            result = self._create_base_result(file_path)
            result.analyzer_name = "JSONAnalyzer"

            # Analyser le contenu
            analysis = self.analyze_content(content, file_path)

            # Mettre à jour les résultats
            self._update_result_from_analysis(result, analysis)
            result.metrics = self._calculate_json_metrics(content, analysis)

            # Calculer le temps de traitement
            result.processing_time_ms = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )

            return result

        except Exception as e:
            logger.error(f"Error analyzing JSON file {file_path}: {e}")
            return self._create_error_result(file_path, str(e))

    def analyze_content(self, content: str, file_path: str) -> Dict[str, Any]:
        """Analyse du contenu JSON (méthode interne)"""
        try:
            data = json.loads(content)
            return {
                'is_valid': True,
                'data': data,
                'structure': self._analyze_structure(data),
                'schema_type': self._detect_schema_type(data),
                'size_metrics': self._calculate_metrics(data),
                'key_patterns': self._detect_key_patterns(data),
                'value_patterns': self._detect_value_patterns(data)
            }
        except json.JSONDecodeError as e:
            return {
                'is_valid': False,
                'error': str(e),
                'structure': {'type': 'invalid'},
                'schema_type': 'invalid',
                'size_metrics': {'total_elements': 0, 'depth': 0}
            }

    def _update_result_from_analysis(self, result: AnalysisResult, analysis: Dict[str, Any]) -> None:
        """Met à jour l'AnalysisResult avec les données de l'analyse JSON"""

        if not analysis.get('is_valid', False):
            result.status = AnalysisStatus.ERROR
            result.errors.append(f"Invalid JSON: {analysis.get('error', 'Unknown error')}")
            return

        # Ajouter les éléments de code (clés JSON comme éléments)
        self._add_json_elements(result, analysis)

        # Mettre à jour les patterns
        self._update_patterns(result, analysis)

        # Mettre à jour les dépendances
        self._update_dependencies(result, analysis)

        # Mettre à jour la sécurité
        self._update_security(result, analysis)

        # Mettre à jour les données spécifiques au langage
        self._update_language_specific(result, analysis)

        # Ajouter des diagnostics
        self._add_diagnostics(result, analysis)

    def _add_json_elements(self, result: AnalysisResult, analysis: Dict[str, Any]) -> None:
        """Convertit les éléments JSON en CodeElement"""
        data = analysis.get('data', {})

        if isinstance(data, dict):
            for key, value in data.items():
                element_type = self._determine_json_element_type(value)
                metadata = self._create_json_element_metadata(key, value)

                element = CodeElement(
                    name=str(key),
                    element_type=element_type,
                    metadata=metadata
                )
                result.elements.append(element)

                # Compter les éléments
                if element_type == 'object':
                    result.metrics.class_count += 1
                elif element_type == 'array':
                    result.metrics.import_count += 1  # Utilisé pour compter les arrays
        elif isinstance(data, list):
            # Pour les arrays JSON, créer un élément pour l'array
            element = CodeElement(
                name="root_array",
                element_type='array',
                metadata={
                    'length': len(data),
                    'item_types': analysis.get('structure', {}).get('item_types', {}),
                    'type': 'json_array'
                }
            )
            result.elements.append(element)
            result.metrics.import_count = len(data)

    def _update_patterns(self, result: AnalysisResult, analysis: Dict[str, Any]) -> None:
        """Met à jour la détection de patterns"""
        patterns = result.patterns
        schema_type = analysis.get('schema_type', 'generic')

        # Détecter le type de schéma
        if schema_type == 'json_schema':
            patterns.patterns.append('json_schema')
        elif schema_type == 'openapi':
            patterns.patterns.append('openapi_spec')
            patterns.frameworks.append(FrameworkType.EXPRESS)  # OpenAPI souvent utilisé avec Express
        elif schema_type == 'package_json':
            patterns.patterns.append('node_package')
            patterns.frameworks.append(FrameworkType.VANILLA)

        # Détecter les patterns de clés
        key_patterns = analysis.get('key_patterns', {})
        if key_patterns.get('has_nesting', False):
            patterns.architecture_hints.append('nested_structure')

        if key_patterns.get('has_config_keys', False):
            patterns.architecture_hints.append('configuration_file')

        # Détecter les bibliothèques basées sur les clés
        libraries = []
        if self._is_tsconfig_file(analysis):
            libraries.append('typescript')
        if self._is_eslint_config(analysis):
            libraries.append('eslint')
        if self._is_jest_config(analysis):
            libraries.append('jest')

        patterns.libraries.extend(libraries)

    def _update_dependencies(self, result: AnalysisResult, analysis: Dict[str, Any]) -> None:
        """Met à jour les informations de dépendances"""
        deps = result.dependencies
        data = analysis.get('data', {})

        if isinstance(data, dict):
            # Détecter package.json
            if analysis.get('schema_type') == 'package_json':
                deps.package_manager = 'npm'

                # Extraire les dépendances
                for dep_type in ['dependencies', 'devDependencies', 'peerDependencies', 'optionalDependencies']:
                    if dep_type in data and isinstance(data[dep_type], dict):
                        deps.external_deps.extend(list(data[dep_type].keys()))

                # Extraire les scripts
                if 'scripts' in data and isinstance(data['scripts'], dict):
                    for script_name, script_cmd in data['scripts'].items():
                        deps.imports.append({
                            'name': script_name,
                            'type': 'npm_script',
                            'command': script_cmd
                        })

            # Détecter composer.json (PHP)
            if 'require' in data and isinstance(data['require'], dict):
                deps.package_manager = 'composer'
                deps.external_deps.extend(list(data['require'].keys()))

            # Détecter requirements.txt style
            if 'requirements' in data and isinstance(data['requirements'], list):
                deps.package_manager = 'pip'
                deps.external_deps.extend(data['requirements'])

    def _update_security(self, result: AnalysisResult, analysis: Dict[str, Any]) -> None:
        """Met à jour l'analyse de sécurité"""
        security = result.security
        data = analysis.get('data', {})

        # Vérifier les problèmes de sécurité courants dans JSON
        if isinstance(data, dict):
            # Vérifier les secrets potentiels
            secret_keys = ['password', 'secret', 'token', 'key', 'credential', 'auth']
            found_secrets = []

            def check_for_secrets(obj, path=""):
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        full_path = f"{path}.{key}" if path else key
                        if any(secret in key.lower() for secret in secret_keys):
                            found_secrets.append(full_path)
                        check_for_secrets(value, full_path)
                elif isinstance(obj, list):
                    for i, item in enumerate(obj):
                        check_for_secrets(item, f"{path}[{i}]")

            check_for_secrets(data)

            if found_secrets:
                security.warnings.append(f"Potential secrets found in keys: {', '.join(found_secrets[:5])}")
                if len(found_secrets) > 5:
                    security.warnings.append(f"... and {len(found_secrets) - 5} more")

                security.vulnerabilities.append({
                    'type': 'potential_secret_exposure',
                    'severity': 'high',
                    'description': 'JSON file may contain secrets in key names',
                    'locations': found_secrets[:10]
                })

            # Vérifier les URLs non sécurisées
            def check_insecure_urls(obj):
                urls = []
                if isinstance(obj, dict):
                    for value in obj.values():
                        urls.extend(check_insecure_urls(value))
                elif isinstance(obj, list):
                    for item in obj:
                        urls.extend(check_insecure_urls(item))
                elif isinstance(obj, str):
                    if obj.startswith('http://'):
                        urls.append(obj)
                return urls

            insecure_urls = check_insecure_urls(data)
            if insecure_urls:
                security.warnings.append(f"Insecure HTTP URLs found: {len(insecure_urls)}")
                security.recommendations.append("Consider using HTTPS URLs instead of HTTP")

        # Calculer un score de sécurité basique
        security.security_score = self._calculate_security_score(analysis)

    def _update_language_specific(self, result: AnalysisResult, analysis: Dict[str, Any]) -> None:
        """Met à jour les données spécifiques au JSON"""
        result.language_specific = {
            'json': {
                'is_valid': analysis.get('is_valid', False),
                'schema_type': analysis.get('schema_type', 'generic'),
                'structure_type': analysis.get('structure', {}).get('type', 'unknown'),
                'size_metrics': analysis.get('size_metrics', {}),
                'key_patterns': analysis.get('key_patterns', {}),
                'value_patterns': analysis.get('value_patterns', {}),
                'is_config_file': self._is_configuration_file(analysis),
                'is_package_file': analysis.get('schema_type') == 'package_json',
                'is_api_spec': analysis.get('schema_type') in ['openapi', 'json_schema']
            }
        }

    def _add_diagnostics(self, result: AnalysisResult, analysis: Dict[str, Any]) -> None:
        """Ajoute des diagnostics et recommandations"""

        if not analysis.get('is_valid', False):
            return

        structure = analysis.get('structure', {})
        size_metrics = analysis.get('size_metrics', {})

        # Vérifier la profondeur
        depth = size_metrics.get('depth', 0)
        if depth > 5:
            result.warnings.append(f"Deeply nested JSON structure (depth: {depth}) - consider flattening")

        # Vérifier la taille
        total_elements = size_metrics.get('total_elements', 0)
        if total_elements > 1000:
            result.notes.append(f"Large JSON file with {total_elements} elements")

        # Vérifier les types de valeurs
        if structure.get('type') == 'object':
            value_types = structure.get('value_types', {})
            if 'dict' in value_types and value_types['dict'] > 10:
                result.notes.append("Complex object with many nested dictionaries")

        # Recommandations pour les fichiers de configuration
        if self._is_configuration_file(analysis):
            result.recommendations.append("Consider adding comments (though not standard JSON) or using JSONC")

    def _analyze_structure(self, data: Any) -> Dict[str, Any]:
        """Analyse la structure du JSON"""
        if isinstance(data, dict):
            return {
                'type': 'object',
                'keys': list(data.keys()),
                'key_count': len(data),
                'nested_objects': self._count_nested_objects(data),
                'value_types': self._analyze_value_types(data),
                'key_pattern_analysis': self._analyze_key_patterns(data.keys())
            }
        elif isinstance(data, list):
            return {
                'type': 'array',
                'length': len(data),
                'item_types': self._analyze_array_types(data),
                'uniform_array': self._is_uniform_array(data)
            }
        else:
            return {
                'type': 'primitive',
                'value_type': type(data).__name__,
                'value': str(data)[:100] if len(str(data)) > 100 else str(data)
            }

    def _count_nested_objects(self, data: dict) -> int:
        """Compte les objets imbriqués"""
        count = 0
        for value in data.values():
            if isinstance(value, dict):
                count += 1 + self._count_nested_objects(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        count += 1 + self._count_nested_objects(item)
        return count

    def _analyze_value_types(self, data: dict) -> Dict[str, int]:
        """Analyse les types de valeurs dans un objet"""
        type_count = {}
        for value in data.values():
            value_type = type(value).__name__
            type_count[value_type] = type_count.get(value_type, 0) + 1
        return type_count

    def _analyze_array_types(self, data: list) -> Dict[str, int]:
        """Analyse les types dans un tableau"""
        if not data:
            return {'empty': 1}

        type_count = {}
        for item in data:
            item_type = type(item).__name__
            type_count[item_type] = type_count.get(item_type, 0) + 1
        return type_count

    def _is_uniform_array(self, data: list) -> bool:
        """Vérifie si un tableau contient des éléments uniformes"""
        if len(data) <= 1:
            return True

        first_type = type(data[0]).__name__
        return all(type(item).__name__ == first_type for item in data[1:])

    def _detect_schema_type(self, data: Any) -> str:
        """Détecte le type de schéma JSON"""
        if isinstance(data, dict):
            # JSON Schema
            if 'schema' in data or '$schema' in data:
                return 'json_schema'

            # OpenAPI/Swagger
            elif 'openapi' in str(data).lower() or 'swagger' in str(data).lower():
                return 'openapi'
            elif 'paths' in data and ('components' in data or 'definitions' in data):
                return 'openapi'

            # package.json
            elif all(key in data for key in ['name', 'version']):
                if 'dependencies' in data or 'scripts' in data:
                    return 'package_json'

            # tsconfig.json
            elif 'compilerOptions' in data:
                return 'tsconfig'

            # eslint config
            elif 'extends' in data and isinstance(data['extends'], str) and 'eslint' in data['extends']:
                return 'eslint_config'

            # jest config
            elif 'testMatch' in data or 'testEnvironment' in data:
                return 'jest_config'

            # Docker compose
            elif 'services' in data and 'version' in data:
                return 'docker_compose'

        return 'generic'

    def _detect_key_patterns(self, data: Any) -> Dict[str, Any]:
        """Détecte les patterns dans les clés JSON"""
        patterns = {
            'has_nesting': False,
            'has_config_keys': False,
            'common_patterns': [],
            'key_statistics': {}
        }

        def analyze_keys(obj, depth=0):
            if isinstance(obj, dict):
                patterns['has_nesting'] = patterns['has_nesting'] or depth > 0

                for key in obj.keys():
                    key_str = str(key)

                    # Statistiques
                    patterns['key_statistics'][key_str] = patterns['key_statistics'].get(key_str, 0) + 1

                    # Patterns communs
                    if key_str.endswith('_url') or key_str.endswith('Url'):
                        patterns['common_patterns'].append('url_pattern')
                    elif key_str.endswith('_id') or key_str.endswith('Id'):
                        patterns['common_patterns'].append('id_pattern')
                    elif key_str.endswith('_name') or key_str.endswith('Name'):
                        patterns['common_patterns'].append('name_pattern')

                    # Clés de configuration
                    config_keys = ['host', 'port', 'database', 'username', 'password', 'config', 'settings']
                    if any(config_key in key_str.lower() for config_key in config_keys):
                        patterns['has_config_keys'] = True

                    # Analyser récursivement
                    analyze_keys(obj[key], depth + 1)

            elif isinstance(obj, list):
                for item in obj:
                    analyze_keys(item, depth)

        analyze_keys(data)

        # Dédupliquer les patterns
        patterns['common_patterns'] = list(set(patterns['common_patterns']))

        return patterns

    def _detect_value_patterns(self, data: Any) -> Dict[str, Any]:
        """Détecte les patterns dans les valeurs JSON"""
        patterns = {
            'string_patterns': [],
            'numeric_ranges': {},
            'boolean_distribution': {'true': 0, 'false': 0}
        }

        def analyze_values(obj):
            if isinstance(obj, dict):
                for value in obj.values():
                    analyze_values(value)
            elif isinstance(obj, list):
                for item in obj:
                    analyze_values(item)
            elif isinstance(obj, str):
                # Patterns de string
                if obj.startswith('http://') or obj.startswith('https://'):
                    patterns['string_patterns'].append('url')
                elif '@' in obj and '.' in obj:
                    patterns['string_patterns'].append('email')
                elif len(obj) == 36 and '-' in obj:  # UUID pattern
                    patterns['string_patterns'].append('uuid')
                elif obj.isdigit() and len(obj) == 10:  # Timestamp
                    patterns['string_patterns'].append('timestamp')

            elif isinstance(obj, (int, float)):
                # Ranges numériques
                if obj < 0:
                    patterns['numeric_ranges']['negative'] = patterns['numeric_ranges'].get('negative', 0) + 1
                elif obj < 100:
                    patterns['numeric_ranges']['small'] = patterns['numeric_ranges'].get('small', 0) + 1
                else:
                    patterns['numeric_ranges']['large'] = patterns['numeric_ranges'].get('large', 0) + 1

            elif isinstance(obj, bool):
                if obj:
                    patterns['boolean_distribution']['true'] += 1
                else:
                    patterns['boolean_distribution']['false'] += 1

        analyze_values(data)

        # Dédupliquer
        patterns['string_patterns'] = list(set(patterns['string_patterns']))

        return patterns

    def _calculate_metrics(self, data: Any) -> Dict[str, int]:
        """Calcule des métriques sur les données JSON"""

        def count_elements(obj):
            if isinstance(obj, dict):
                return 1 + sum(count_elements(v) for v in obj.values())
            elif isinstance(obj, list):
                return 1 + sum(count_elements(item) for item in obj)
            else:
                return 1

        return {
            'total_elements': count_elements(data),
            'depth': self._calculate_depth(data)
        }

    def _calculate_depth(self, obj, current_depth=0):
        """Calcule la profondeur maximale du JSON"""
        if isinstance(obj, dict):
            if obj:
                return max(self._calculate_depth(v, current_depth + 1) for v in obj.values())
            else:
                return current_depth + 1
        elif isinstance(obj, list):
            if obj:
                return max(self._calculate_depth(item, current_depth + 1) for item in obj)
            else:
                return current_depth + 1
        else:
            return current_depth

    def _determine_json_element_type(self, value: Any) -> str:
        """Détermine le type d'élément JSON"""
        if isinstance(value, dict):
            return 'object'
        elif isinstance(value, list):
            return 'array'
        elif isinstance(value, str):
            return 'string'
        elif isinstance(value, (int, float)):
            return 'number'
        elif isinstance(value, bool):
            return 'boolean'
        elif value is None:
            return 'null'
        else:
            return 'unknown'

    def _create_json_element_metadata(self, key: str, value: Any) -> Dict[str, Any]:
        """Crée les métadonnées pour un élément JSON"""
        metadata = {
            'json_type': self._determine_json_element_type(value),
            'key_name': key
        }

        if isinstance(value, dict):
            metadata.update({
                'key_count': len(value),
                'keys': list(value.keys())[:10]  # Limiter pour éviter des métadonnées trop grandes
            })
        elif isinstance(value, list):
            metadata.update({
                'length': len(value),
                'item_type': type(value[0]).__name__ if value else 'empty'
            })
        elif isinstance(value, str):
            metadata.update({
                'length': len(value),
                'is_long': len(value) > 100
            })

        return metadata

    def _calculate_json_metrics(self, content: str, analysis: Dict[str, Any]) -> FileMetrics:
        """Calcule les métriques spécifiques au JSON"""
        metrics = super()._calculate_metrics(content)

        if not analysis.get('is_valid', False):
            return metrics

        size_metrics = analysis.get('size_metrics', {})
        structure = analysis.get('structure', {})

        # Mettre à jour avec les métriques spécifiques au JSON
        metrics.total_elements = size_metrics.get('total_elements', 0)
        metrics.depth = size_metrics.get('depth', 0)

        # Compter les classes (objets) et imports (arrays)
        if structure.get('type') == 'object':
            metrics.class_count = 1
            metrics.import_count = structure.get('key_count', 0)
        elif structure.get('type') == 'array':
            metrics.import_count = structure.get('length', 0)

        # Calculer la complexité basée sur la profondeur et le nombre d'éléments
        metrics.complexity_score = (size_metrics.get('depth', 0) * 2) + (size_metrics.get('total_elements', 0) / 100)

        return metrics

    def _calculate_security_score(self, analysis: Dict[str, Any]) -> float:
        """Calcule un score de sécurité basique pour JSON"""
        score = 100.0

        if not analysis.get('is_valid', False):
            return 0.0

        data = analysis.get('data', {})

        # Pénalités pour les problèmes de sécurité
        def check_security_issues(obj, path=""):
            issues = 0
            if isinstance(obj, dict):
                for key, value in obj.items():
                    key_str = str(key).lower()

                    # Secrets dans les clés
                    if any(secret in key_str for secret in ['password', 'secret', 'token', 'key', 'credential']):
                        issues += 1

                    # Secrets dans les valeurs (strings simples)
                    if isinstance(value, str) and len(value) > 20:
                        # Vérifier si ça ressemble à un token
                        if value.isalnum() and len(value) > 32:
                            issues += 1

                    # URLs non sécurisées
                    if isinstance(value, str) and value.startswith('http://'):
                        issues += 0.5

                    issues += check_security_issues(value, f"{path}.{key}")

            elif isinstance(obj, list):
                for item in obj:
                    issues += check_security_issues(item, f"{path}[]")

            return issues

        security_issues = check_security_issues(data)
        score -= security_issues * 10

        return max(0, min(100, score))

    def _is_configuration_file(self, analysis: Dict[str, Any]) -> bool:
        """Détermine si c'est un fichier de configuration"""
        schema_type = analysis.get('schema_type', '')
        config_types = ['tsconfig', 'eslint_config', 'jest_config', 'package_json']
        return schema_type in config_types

    def _is_tsconfig_file(self, analysis: Dict[str, Any]) -> bool:
        """Détermine si c'est un tsconfig.json"""
        return analysis.get('schema_type') == 'tsconfig'

    def _is_eslint_config(self, analysis: Dict[str, Any]) -> bool:
        """Détermine si c'est une config ESLint"""
        return analysis.get('schema_type') == 'eslint_config'

    def _is_jest_config(self, analysis: Dict[str, Any]) -> bool:
        """Détermine si c'est une config Jest"""
        return analysis.get('schema_type') == 'jest_config'

    def _analyze_key_patterns(self, keys: List[str]) -> Dict[str, Any]:
        """Analyse les patterns dans les noms de clés"""
        patterns = {
            'casing_style': self._detect_casing_style(keys),
            'separator_style': self._detect_separator_style(keys),
            'common_prefixes': self._find_common_prefixes(keys),
            'common_suffixes': self._find_common_suffixes(keys)
        }
        return patterns

    def _detect_casing_style(self, keys: List[str]) -> str:
        """Détecte le style de casse des clés"""
        if not keys:
            return 'unknown'

        camel_case = sum(1 for k in keys if re.match(r'^[a-z]+[A-Za-z]*$', k))
        snake_case = sum(1 for k in keys if '_' in k)
        kebab_case = sum(1 for k in keys if '-' in k)
        pascal_case = sum(1 for k in keys if re.match(r'^[A-Z][A-Za-z]*$', k))

        counts = {
            'camelCase': camel_case,
            'snake_case': snake_case,
            'kebab-case': kebab_case,
            'PascalCase': pascal_case
        }

        return max(counts.items(), key=lambda x: x[1])[0]

    def _detect_separator_style(self, keys: List[str]) -> str:
        """Détecte le style de séparateur des clés"""
        if not keys:
            return 'none'

        has_dots = any('.' in k for k in keys)
        has_dashes = any('-' in k for k in keys)
        has_underscores = any('_' in k for k in keys)

        if has_dots:
            return 'dot_notation'
        elif has_dashes:
            return 'kebab_case'
        elif has_underscores:
            return 'snake_case'
        else:
            return 'simple'

    def _find_common_prefixes(self, keys: List[str]) -> List[str]:
        """Trouve les préfixes communs dans les clés"""
        if len(keys) < 2:
            return []

        # Trouver les préfixes communs
        prefixes = {}
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                prefix = os.path.commonprefix([keys[i], keys[j]])
                if len(prefix) > 2 and not prefix.endswith(('_', '-', '.')):
                    prefixes[prefix] = prefixes.get(prefix, 0) + 1

        # Retourner les préfixes les plus communs
        common_prefixes = sorted(prefixes.items(), key=lambda x: x[1], reverse=True)[:5]
        return [prefix for prefix, count in common_prefixes if count > 1]

    def _find_common_suffixes(self, keys: List[str]) -> List[str]:
        """Trouve les suffixes communs dans les clés"""
        if len(keys) < 2:
            return []

        # Inverser les clés pour trouver les suffixes
        reversed_keys = [k[::-1] for k in keys]

        suffixes = {}
        for i in range(len(reversed_keys)):
            for j in range(i + 1, len(reversed_keys)):
                suffix = os.path.commonprefix([reversed_keys[i], reversed_keys[j]])
                if len(suffix) > 2 and not suffix.endswith(('_', '-', '.')):
                    normal_suffix = suffix[::-1]
                    suffixes[normal_suffix] = suffixes.get(normal_suffix, 0) + 1

        # Retourner les suffixes les plus communs
        common_suffixes = sorted(suffixes.items(), key=lambda x: x[1], reverse=True)[:5]
        return [suffix for suffix, count in common_suffixes if count > 1]
