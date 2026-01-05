# properties_analyzer.py
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


class PropertiesAnalyzer(Analyzer):
    """Analyseur de fichiers de propriétés retournant des AnalysisResult"""

    def __init__(self):
        super().__init__()
        self.file_type = FileType.YAML  # Utiliser YAML comme approximation pour les fichiers de propriétés

    def analyze_file(self, file_path: str) -> AnalysisResult:
        """Analyse un fichier de propriétés et retourne un AnalysisResult"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Mesurer le temps de traitement
            start_time = datetime.now()

            # Créer le résultat de base
            result = self._create_base_result(file_path)
            result.analyzer_name = "PropertiesAnalyzer"

            # Analyser le contenu
            analysis = self.analyze_content(content, file_path)

            # Mettre à jour les résultats
            self._update_result_from_analysis(result, analysis)
            result.metrics = self._calculate_properties_metrics(content, analysis)

            # Calculer le temps de traitement
            result.processing_time_ms = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )

            return result

        except Exception as e:
            logger.error(f"Error analyzing properties file {file_path}: {e}")
            return self._create_error_result(file_path, str(e))

    def analyze_content(self, content: str, file_path: str) -> Dict[str, Any]:
        """Analyse du contenu des propriétés (méthode interne)"""
        return {
            'properties': self._extract_properties(content),
            'sections': self._extract_sections(content),
            'comments': self._extract_comments(content),
            'analysis': self._analyze_properties_patterns(content),
            'file_type': self._detect_properties_format(content),
            'has_includes': self._detect_includes(content)
        }

    def _update_result_from_analysis(self, result: AnalysisResult, analysis: Dict[str, Any]) -> None:
        """Met à jour l'AnalysisResult avec les données de l'analyse des propriétés"""

        # Ajouter les éléments de code (propriétés comme éléments)
        self._add_properties_elements(result, analysis)

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

    def _add_properties_elements(self, result: AnalysisResult, analysis: Dict[str, Any]) -> None:
        """Convertit les propriétés en CodeElement"""

        # Propriétés simples
        for prop in analysis['properties']:
            element = CodeElement(
                name=prop['key'],
                element_type='property',
                line_start=prop.get('line'),
                metadata={
                    'value': prop['value'],
                    'has_expression': prop.get('has_expression', False),
                    'expression_type': prop.get('expression_type'),
                    'is_commented': prop.get('is_commented', False),
                    'section': prop.get('section'),
                    'type': 'configuration_property'
                }
            )
            result.elements.append(element)
            result.metrics.import_count += 1  # Utiliser import_count pour compter les propriétés

        # Sections (groupes de propriétés)
        for section in analysis['sections']:
            element = CodeElement(
                name=section['name'],
                element_type='section',
                line_start=section.get('start_line'),
                line_end=section.get('end_line'),
                metadata={
                    'property_count': len(section.get('properties', [])),
                    'type': 'configuration_section',
                    'is_nested': section.get('is_nested', False)
                }
            )
            result.elements.append(element)
            result.metrics.class_count += 1  # Utiliser class_count pour les sections

    def _update_dependencies(self, result: AnalysisResult, analysis: Dict[str, Any]) -> None:
        """Met à jour les informations de dépendances"""
        deps = result.dependencies

        # Détecter les fichiers inclus
        if analysis.get('has_includes'):
            include_patterns = [
                r'include\s*=\s*(.+)',
                r'import\s*=\s*(.+)',
                r'source\s*=\s*(.+)'
            ]

            content = "\n".join([str(p.get('value', '')) for p in analysis.get('properties', [])])
            for pattern in include_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches:
                    deps.internal_deps.append(match.strip().strip('"\''))

        # Détecter les références externes (URLs, hosts, etc.)
        for prop in analysis.get('properties', []):
            value = prop.get('value', '')

            # URLs
            if re.match(r'https?://', value):
                deps.external_deps.append(self._extract_domain_from_url(value))

            # JDBC URLs
            elif 'jdbc:' in value.lower():
                jdbc_match = re.match(r'jdbc:(\w+):', value.lower())
                if jdbc_match:
                    deps.external_deps.append(f"jdbc:{jdbc_match.group(1)}")

            # Noms de serveurs/hôtes
            elif prop.get('key', '').lower() in ['host', 'server', 'endpoint']:
                deps.external_deps.append(value.split(':')[0] if ':' in value else value)

        # Déterminer le type de fichier de configuration
        file_type = analysis.get('file_type', '')
        if file_type == 'spring':
            deps.package_manager = 'maven'
        elif file_type == 'dotenv':
            deps.package_manager = 'npm'  # Pour Node.js projects

    def _update_patterns(self, result: AnalysisResult, analysis: Dict[str, Any]) -> None:
        """Met à jour la détection de patterns"""
        patterns = result.patterns
        prop_analysis = analysis.get('analysis', {})

        # Détecter le type de configuration
        file_type = analysis.get('file_type', '')
        if file_type:
            patterns.patterns.append(f'{file_type}_config')

        # Détecter les frameworks via les propriétés
        if file_type == 'spring':
            patterns.frameworks.append(FrameworkType.SPRING)
        elif file_type == 'dotenv':
            patterns.frameworks.append(FrameworkType.VANILLA)  # Node.js

        # Patterns communs détectés
        common_patterns = prop_analysis.get('common_patterns', [])
        patterns.patterns.extend(common_patterns)

        # Catégories de propriétés
        categories = prop_analysis.get('property_categories', {})
        for category, count in categories.items():
            if count > 0:
                patterns.architecture_hints.append(f'{category}_config')

        # Détecter les bibliothèques spécifiques
        if any('kafka' in prop.get('key', '').lower() or 'kafka' in prop.get('value', '').lower()
               for prop in analysis.get('properties', [])):
            patterns.libraries.append('apache_kafka')

        if any('redis' in prop.get('key', '').lower() or 'redis' in prop.get('value', '').lower()
               for prop in analysis.get('properties', [])):
            patterns.libraries.append('redis')

        if any('elasticsearch' in prop.get('key', '').lower() or 'elastic' in prop.get('value', '').lower()
               for prop in analysis.get('properties', [])):
            patterns.libraries.append('elasticsearch')

    def _update_security(self, result: AnalysisResult, analysis: Dict[str, Any]) -> None:
        """Met à jour l'analyse de sécurité"""
        security = result.security

        # Détecter les secrets dans les propriétés
        secrets_found = []
        sensitive_properties = []

        for prop in analysis.get('properties', []):
            key = prop.get('key', '').lower()
            value = prop.get('value', '')

            # Détecter les clés sensibles
            sensitive_keywords = ['password', 'secret', 'key', 'token', 'credential', 'auth']
            if any(keyword in key for keyword in sensitive_keywords):
                sensitive_properties.append(key)

                # Vérifier si la valeur semble être un secret
                if self._looks_like_secret(value):
                    secrets_found.append({
                        'key': prop.get('key'),
                        'line': prop.get('line'),
                        'severity': 'high'
                    })
                    security.vulnerabilities.append({
                        'type': 'hardcoded_secret',
                        'severity': 'high',
                        'description': f'Hardcoded secret found in property: {prop.get("key")}',
                        'recommendation': 'Use environment variables or secure secret management',
                        'location': f'line {prop.get("line")}'
                    })

        if secrets_found:
            security.warnings.append(f"Hardcoded secrets detected: {len(secrets_found)}")

        if sensitive_properties:
            security.notes.append(f"Sensitive configuration keys found: {', '.join(sensitive_properties[:5])}")
            if len(sensitive_properties) > 5:
                security.notes.append(f"... and {len(sensitive_properties) - 5} more")

        # Détecter les configurations non sécurisées
        for prop in analysis.get('properties', []):
            key = prop.get('key', '').lower()
            value = prop.get('value', '').lower()

            # SSL/TLS désactivé
            if 'ssl' in key and 'false' in value:
                security.warnings.append(f"SSL/TLS disabled: {prop.get('key')}")
                security.recommendations.append("Enable SSL/TLS for secure communication")

            # Authentication désactivée
            if 'auth' in key and 'false' in value:
                security.warnings.append(f"Authentication disabled: {prop.get('key')}")
                security.recommendations.append("Enable authentication for security")

            # Debug mode activé
            if 'debug' in key and 'true' in value:
                security.warnings.append(f"Debug mode enabled: {prop.get('key')}")
                security.recommendations.append("Disable debug mode in production")

        # Calculer le score de sécurité
        security.security_score = self._calculate_security_score(analysis)

    def _update_language_specific(self, result: AnalysisResult, analysis: Dict[str, Any]) -> None:
        """Met à jour les données spécifiques aux propriétés"""
        result.language_specific = {
            'properties': {
                'format': analysis.get('file_type', 'generic'),
                'property_count': len(analysis.get('properties', [])),
                'section_count': len(analysis.get('sections', [])),
                'comment_count': len(analysis.get('comments', [])),
                'has_expressions': analysis.get('analysis', {}).get('has_expressions', False),
                'has_includes': analysis.get('has_includes', False),
                'is_environment_file': self._is_environment_file(analysis),
                'is_application_config': self._is_application_config(analysis),
                'is_framework_config': analysis.get('file_type') in ['spring', 'dotenv'],
                'encoding_detected': self._detect_encoding(analysis)
            }
        }

    def _add_diagnostics(self, result: AnalysisResult, analysis: Dict[str, Any]) -> None:
        """Ajoute des diagnostics et recommandations"""

        properties = analysis.get('properties', [])
        sections = analysis.get('sections', [])

        # Vérifier le nombre de propriétés
        if len(properties) == 0:
            result.notes.append("Empty configuration file")
        elif len(properties) > 100:
            result.warnings.append(f"Large configuration file with {len(properties)} properties")

        # Vérifier les expressions non résolues
        expressions = [p for p in properties if p.get('has_expression')]
        if expressions:
            result.notes.append(f"Configuration uses expressions: {len(expressions)} property(ies)")

        # Vérifier les sections imbriquées
        nested_sections = [s for s in sections if s.get('is_nested')]
        if nested_sections:
            result.notes.append(f"Nested sections detected: {len(nested_sections)}")

        # Recommandations
        if not any('logging' in cat for cat in analysis.get('analysis', {}).get('property_categories', {})):
            result.recommendations.append("Consider adding logging configuration")

        if analysis.get('analysis', {}).get('has_expressions'):
            result.recommendations.append("Ensure environment variables are properly set for expressions")

    def _extract_properties(self, content: str) -> List[Dict]:
        """Extrait les propriétés du fichier"""
        properties = []
        lines = content.split('\n')

        current_section = None
        in_multiline_value = False
        multiline_buffer = []
        multiline_key = None

        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()

            # Ignorer les lignes vides
            if not stripped:
                continue

            # Gérer les valeurs multilignes
            if in_multiline_value:
                if stripped.endswith('\\'):
                    multiline_buffer.append(stripped.rstrip('\\'))
                    continue
                else:
                    multiline_buffer.append(stripped)
                    value = ' '.join(multiline_buffer)
                    properties.append(self._create_property_info(
                        multiline_key, value, line_num - len(multiline_buffer),
                        current_section, content
                    ))
                    in_multiline_value = False
                    multiline_buffer = []
                    multiline_key = None
                    continue

            # Commentaires
            if stripped.startswith('#') or stripped.startswith('!'):
                # Commentaires sur la même ligne qu'une propriété
                if '=' in stripped:
                    comment_index = stripped.find('#') if '#' in stripped else stripped.find('!')
                    equal_index = stripped.find('=')
                    if comment_index > equal_index:
                        # C'est un commentaire après la valeur
                        prop_part = stripped[:comment_index].rstrip()
                        if '=' in prop_part:
                            key, _, value = prop_part.partition('=')
                            prop_info = self._create_property_info(
                                key.strip(), value.strip(), line_num,
                                current_section, content
                            )
                            prop_info['is_commented'] = True
                            properties.append(prop_info)
                continue

            # Sections (pour les fichiers .ini, .cfg)
            if stripped.startswith('[') and stripped.endswith(']'):
                current_section = stripped[1:-1]
                continue

            # Propriétés avec =
            if '=' in stripped:
                key, _, value = stripped.partition('=')
                key = key.strip()
                value = value.strip()

                # Vérifier les valeurs multilignes
                if value.endswith('\\'):
                    in_multiline_value = True
                    multiline_key = key
                    multiline_buffer.append(value.rstrip('\\'))
                    continue

                properties.append(self._create_property_info(
                    key, value, line_num, current_section, content
                ))

            # Propriétés avec : (format alternatif)
            elif ':' in stripped and not stripped.startswith('#'):
                key, _, value = stripped.partition(':')
                properties.append(self._create_property_info(
                    key.strip(), value.strip(), line_num, current_section, content
                ))

        return properties

    def _create_property_info(self, key: str, value: str, line_num: int,
                              section: Optional[str], content: str) -> Dict[str, Any]:
        """Crée un dictionnaire d'information pour une propriété"""
        has_expression, expr_type = self._detect_expression(value)

        return {
            'key': key,
            'value': value,
            'line': line_num,
            'section': section,
            'has_expression': has_expression,
            'expression_type': expr_type,
            'is_commented': False,
            'value_type': self._detect_value_type(value),
            'is_sensitive': self._is_sensitive_key(key)
        }

    def _extract_sections(self, content: str) -> List[Dict]:
        """Extrait les sections du fichier"""
        sections = []
        lines = content.split('\n')

        current_section = None
        section_properties = []
        section_start = 0

        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()

            # Nouvelle section
            if stripped.startswith('[') and stripped.endswith(']'):
                # Sauvegarder la section précédente
                if current_section is not None:
                    sections.append({
                        'name': current_section,
                        'start_line': section_start,
                        'end_line': line_num - 1,
                        'properties': section_properties.copy(),
                        'property_count': len(section_properties),
                        'is_nested': '.' in current_section
                    })

                # Commencer une nouvelle section
                current_section = stripped[1:-1]
                section_properties = []
                section_start = line_num

            # Ajouter les propriétés à la section courante
            elif current_section is not None and '=' in stripped and not stripped.startswith('#'):
                key, _, value = stripped.partition('=')
                section_properties.append({
                    'key': key.strip(),
                    'value': value.strip(),
                    'line': line_num
                })

        # Ajouter la dernière section
        if current_section is not None:
            sections.append({
                'name': current_section,
                'start_line': section_start,
                'end_line': len(lines),
                'properties': section_properties,
                'property_count': len(section_properties),
                'is_nested': '.' in current_section
            })

        return sections

    def _extract_comments(self, content: str) -> List[Dict]:
        """Extrait les commentaires du fichier"""
        comments = []
        lines = content.split('\n')

        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()

            if stripped.startswith('#') or stripped.startswith('!'):
                comment_text = stripped[1:].strip() if stripped.startswith('#') else stripped[1:].strip()
                comments.append({
                    'text': comment_text,
                    'line': line_num,
                    'is_inline': '#' in line and '=' in line and line.find('#') > line.find('='),
                    'length': len(comment_text)
                })

        return comments

    def _analyze_properties_patterns(self, content: str) -> Dict[str, Any]:
        """Analyse les patterns dans les propriétés"""
        properties = self._extract_properties(content)

        return {
            'total_properties': len(properties),
            'commented_lines': len([l for l in content.split('\n') if l.strip().startswith('#') or l.strip().startswith('!')]),
            'has_expressions': any(p['has_expression'] for p in properties),
            'expression_count': len([p for p in properties if p['has_expression']]),
            'common_patterns': self._detect_common_patterns(properties),
            'property_categories': self._categorize_properties(properties),
            'value_distribution': self._analyze_value_distribution(properties)
        }

    def _detect_common_patterns(self, properties: List[Dict]) -> List[str]:
        """Détecte les patterns communs"""
        patterns = []
        keys = [p['key'].lower() for p in properties]
        values = [p['value'] for p in properties]

        # Détection des patterns
        if any('jdbc:' in value.lower() for value in values):
            patterns.append('database_configuration')

        if any('url' in key or 'uri' in key for key in keys):
            patterns.append('url_configuration')

        if any('password' in key or 'secret' in key or 'key' in key for key in keys):
            patterns.append('security_configuration')

        if any('port' in key for key in keys):
            patterns.append('network_configuration')

        if any('timeout' in key or 'retry' in key for key in keys):
            patterns.append('timeout_configuration')

        if any('log' in key or 'logging' in key for key in keys):
            patterns.append('logging_configuration')

        if any('cache' in key for key in keys):
            patterns.append('cache_configuration')

        if any('queue' in key or 'topic' in key for key in keys):
            patterns.append('messaging_configuration')

        return patterns

    def _categorize_properties(self, properties: List[Dict]) -> Dict[str, int]:
        """Catégorise les propriétés"""
        categories = {
            'database': 0,
            'server': 0,
            'security': 0,
            'logging': 0,
            'external_services': 0,
            'performance': 0,
            'messaging': 0,
            'cache': 0,
            'monitoring': 0,
            'other': 0
        }

        for prop in properties:
            key = prop['key'].lower()

            if any(db_word in key for db_word in ['db', 'database', 'jdbc', 'sql', 'hibernate']):
                categories['database'] += 1
            elif any(server_word in key for server_word in ['server', 'port', 'host', 'address', 'bind']):
                categories['server'] += 1
            elif any(sec_word in key for sec_word in ['password', 'secret', 'key', 'auth', 'ssl', 'tls', 'cert', 'encrypt']):
                categories['security'] += 1
            elif any(log_word in key for log_word in ['log', 'logging', 'logger', 'appender']):
                categories['logging'] += 1
            elif any(ext_word in key for ext_word in ['api', 'service', 'endpoint', 'client', 'external']):
                categories['external_services'] += 1
            elif any(perf_word in key for perf_word in ['timeout', 'retry', 'pool', 'thread', 'performance']):
                categories['performance'] += 1
            elif any(msg_word in key for msg_word in ['queue', 'topic', 'kafka', 'rabbit', 'jms', 'messaging']):
                categories['messaging'] += 1
            elif any(cache_word in key for cache_word in ['cache', 'redis', 'memcached']):
                categories['cache'] += 1
            elif any(mon_word in key for mon_word in ['metric', 'monitor', 'health', 'prometheus']):
                categories['monitoring'] += 1
            else:
                categories['other'] += 1

        # Supprimer les catégories vides
        return {k: v for k, v in categories.items() if v > 0}

    def _analyze_value_distribution(self, properties: List[Dict]) -> Dict[str, Any]:
        """Analyse la distribution des valeurs"""
        value_types = {}
        value_lengths = []
        numeric_values = []
        boolean_values = {'true': 0, 'false': 0}

        for prop in properties:
            value = prop['value']

            # Type de valeur
            value_type = prop.get('value_type', 'string')
            value_types[value_type] = value_types.get(value_type, 0) + 1

            # Longueur
            value_lengths.append(len(value))

            # Valeurs numériques
            if value_type == 'number':
                try:
                    numeric_values.append(float(value))
                except ValueError:
                    pass

            # Valeurs booléennes
            if value.lower() in ['true', 'false']:
                boolean_values[value.lower()] += 1

        return {
            'value_types': value_types,
            'avg_value_length': sum(value_lengths) / len(value_lengths) if value_lengths else 0,
            'max_value_length': max(value_lengths) if value_lengths else 0,
            'numeric_values': {
                'count': len(numeric_values),
                'avg': sum(numeric_values) / len(numeric_values) if numeric_values else 0
            },
            'boolean_values': boolean_values
        }

    def _detect_properties_format(self, content: str) -> str:
        """Détecte le format du fichier de propriétés"""
        lines = content.split('\n')

        # Spring Boot application.properties/yml
        if any('spring.' in line for line in lines):
            return 'spring'

        # .env files
        if any('export ' in line for line in lines) or content.count('=') > content.count('\n') * 0.8:
            return 'dotenv'

        # INI files with sections
        if any(line.strip().startswith('[') and line.strip().endswith(']') for line in lines):
            return 'ini'

        # Java properties files
        if any('=' in line and not line.strip().startswith('#') for line in lines[:10]):
            return 'java_properties'

        # YAML-like (avec indentations)
        if any(':' in line and line.count('  ') > 0 for line in lines[:10]):
            return 'yaml'

        return 'generic'

    def _detect_includes(self, content: str) -> bool:
        """Détecte les inclusions d'autres fichiers"""
        include_patterns = [
            r'include\s*[=:]\s*',
            r'import\s*[=:]\s*',
            r'source\s*[=:]\s*',
            r'@include\s+',
            r'#include\s+'
        ]

        for pattern in include_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return True

        return False

    def _detect_expression(self, value: str) -> tuple[bool, Optional[str]]:
        """Détecte les expressions dans les valeurs"""
        expression_patterns = {
            'env_variable': r'\$\{?(\w+)\}?',
            'spring_expression': r'#\{.*?\}',
            'system_property': r'\$\{systemProperty\[.*?\]\}',
            'command': r'\$\(.*?\)'
        }

        for expr_type, pattern in expression_patterns.items():
            if re.search(pattern, value):
                return True, expr_type

        return False, None

    def _detect_value_type(self, value: str) -> str:
        """Détecte le type de la valeur"""
        if not value:
            return 'empty'

        # Booléen
        if value.lower() in ['true', 'false', 'yes', 'no', 'on', 'off']:
            return 'boolean'

        # Nombre
        try:
            float(value)
            return 'number'
        except ValueError:
            pass

        # Liste séparée par des virgules
        if ',' in value and not re.search(r'\$\{.*\}', value):
            return 'list'

        # URL
        if re.match(r'https?://', value) or re.match(r'jdbc:', value, re.IGNORECASE):
            return 'url'

        # Chemin de fichier
        if '/' in value or '\\' in value or value.endswith(('.properties', '.yml', '.yaml', '.xml', '.json')):
            return 'path'

        # Expression
        if self._detect_expression(value)[0]:
            return 'expression'

        return 'string'

    def _is_sensitive_key(self, key: str) -> bool:
        """Détermine si une clé est sensible"""
        sensitive_patterns = [
            r'password', r'secret', r'key', r'token', r'credential',
            r'auth', r'private', r'certificate', r'jwt', r'api[._-]?key'
        ]

        key_lower = key.lower()
        return any(re.search(pattern, key_lower) for pattern in sensitive_patterns)

    def _looks_like_secret(self, value: str) -> bool:
        """Détermine si une valeur ressemble à un secret"""
        # Longueur typique des tokens/secrets
        if 20 <= len(value) <= 100 and value.isalnum():
            return True

        # UUID
        if re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', value, re.IGNORECASE):
            return True

        # JWT-like
        if re.match(r'^[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+$', value):
            return True

        # Base64-like
        if re.match(r'^[A-Za-z0-9+/]+={0,2}$', value) and len(value) > 20:
            return True

        return False

    def _extract_domain_from_url(self, url: str) -> str:
        """Extrait le domaine d'une URL"""
        match = re.match(r'https?://([^/:]+)', url)
        return match.group(1) if match else url

    def _calculate_properties_metrics(self, content: str, analysis: Dict[str, Any]) -> FileMetrics:
        """Calcule les métriques spécifiques aux propriétés"""
        metrics = super()._calculate_metrics(content)

        # Mettre à jour avec les métriques spécifiques
        metrics.import_count = len(analysis.get('properties', []))
        metrics.class_count = len(analysis.get('sections', []))

        # Compter les lignes de configuration (exclure les commentaires vides)
        lines = content.split('\n')
        config_lines = len([l for l in lines if l.strip() and not l.strip().startswith('#')])
        metrics.code_lines = config_lines

        # Compter les commentaires
        metrics.comment_lines = len(analysis.get('comments', []))

        # Complexité basée sur le nombre de sections et d'expressions
        sections = analysis.get('sections', [])
        properties = analysis.get('properties', [])
        expressions = [p for p in properties if p.get('has_expression')]

        complexity = len(sections) * 0.5 + len(expressions) * 0.3
        metrics.complexity_score = complexity

        return metrics

    def _calculate_security_score(self, analysis: Dict[str, Any]) -> float:
        """Calcule un score de sécurité pour les fichiers de propriétés"""
        score = 100.0
        properties = analysis.get('properties', [])

        # Pénalités pour les problèmes de sécurité
        sensitive_count = 0
        hardcoded_secrets = 0
        insecure_configs = 0

        for prop in properties:
            key = prop.get('key', '').lower()
            value = prop.get('value', '').lower()

            # Secrets sensibles
            if self._is_sensitive_key(key):
                sensitive_count += 1
                if self._looks_like_secret(prop.get('value', '')):
                    hardcoded_secrets += 1

            # Configurations non sécurisées
            if ('ssl' in key or 'tls' in key) and 'false' in value:
                insecure_configs += 1

            if 'auth' in key and 'false' in value:
                insecure_configs += 1

            if 'debug' in key and 'true' in value:
                insecure_configs += 1

        # Appliquer les pénalités
        score -= hardcoded_secrets * 20
        score -= insecure_configs * 15

        if sensitive_count > 5:
            score -= 10

        # Bonus pour les bonnes pratiques
        if any('env:' in prop.get('value', '') or '${' in prop.get('value', '') for prop in properties):
            score += 5  # Utilisation de variables d'environnement

        return max(0, min(100, score))

    def _is_environment_file(self, analysis: Dict[str, Any]) -> bool:
        """Détermine si c'est un fichier d'environnement"""
        file_type = analysis.get('file_type', '')
        return file_type == 'dotenv' or '.env' in analysis.get('file_path', '').lower()

    def _is_application_config(self, analysis: Dict[str, Any]) -> bool:
        """Détermine si c'est un fichier de configuration d'application"""
        file_path = analysis.get('file_path', '').lower()
        config_patterns = [
            'application', 'config', 'settings', 'properties',
            '.conf', '.cfg', '.ini', '.toml'
        ]

        return any(pattern in file_path for pattern in config_patterns)

    def _detect_encoding(self, analysis: Dict[str, Any]) -> str:
        """Détecte l'encodage probable"""
        # Analyse simplifiée - dans une vraie implémentation, utiliser chardet
        content = "\n".join([str(p.get('value', '')) for p in analysis.get('properties', [])])

        # Vérifier les caractères non-ASCII
        non_ascii = sum(1 for c in content if ord(c) > 127)
        if non_ascii > 0:
            return 'utf-8'

        return 'ascii'