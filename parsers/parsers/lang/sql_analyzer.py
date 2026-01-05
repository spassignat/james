# sql_analyzer.py
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


class SQLAnalyzer(Analyzer):
    """Analyseur de fichiers SQL retournant des AnalysisResult"""

    def __init__(self):
        super().__init__()
        self.file_type = FileType.SQL

    def analyze_file(self, file_path: str) -> AnalysisResult:
        """Analyse un fichier SQL et retourne un AnalysisResult"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Mesurer le temps de traitement
            start_time = datetime.now()

            # Créer le résultat de base
            result = self._create_base_result(file_path)
            result.analyzer_name = "SQLAnalyzer"

            # Analyser le contenu
            analysis = self.analyze_content(content, file_path)

            # Mettre à jour les résultats
            self._update_result_from_analysis(result, analysis)
            result.metrics = self._calculate_sql_metrics(content, analysis)

            # Calculer le temps de traitement
            result.processing_time_ms = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )

            return result

        except Exception as e:
            logger.error(f"Error analyzing SQL file {file_path}: {e}")
            return self._create_error_result(file_path, str(e))

    def analyze_content(self, content: str, file_path: str) -> Dict[str, Any]:
        """Analyse du contenu SQL (méthode interne)"""
        return {
            'statements': self._extract_statements(content),
            'tables': self._extract_tables(content),
            'operations': self._analyze_operations(content),
            'constraints': self._extract_constraints(content),
            'columns': self._extract_columns(content),
            'indexes': self._extract_indexes(content),
            'triggers': self._extract_triggers(content),
            'views': self._extract_views(content),
            'functions': self._extract_functions(content),
            'procedures': self._extract_procedures(content),
            'analysis': self._analyze_sql_patterns(content),
            'database_type': self._detect_database_type(content),
            'security_issues': self._detect_security_issues(content)
        }

    def _update_result_from_analysis(self, result: AnalysisResult, analysis: Dict[str, Any]) -> None:
        """Met à jour l'AnalysisResult avec les données de l'analyse SQL"""

        # Ajouter les éléments de code (tables, vues, fonctions, etc.)
        self._add_sql_elements(result, analysis)

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

    def _add_sql_elements(self, result: AnalysisResult, analysis: Dict[str, Any]) -> None:
        """Convertit les éléments SQL en CodeElement"""

        # Tables
        for table in analysis['tables']:
            element = CodeElement(
                name=table['name'],
                element_type='table',
                metadata={
                    'operation': table.get('operation', 'unknown'),
                    'columns': table.get('columns', []),
                    'constraints': table.get('constraints', []),
                    'is_view': False
                }
            )
            result.elements.append(element)
            result.metrics.class_count += 1

        # Vues
        for view in analysis['views']:
            element = CodeElement(
                name=view['name'],
                element_type='view',
                metadata={
                    'definition': view.get('definition', ''),
                    'is_view': True
                }
            )
            result.elements.append(element)
            result.metrics.class_count += 1

        # Fonctions
        for func in analysis['functions']:
            element = CodeElement(
                name=func['name'],
                element_type='function',
                parameters=func.get('parameters', []),
                return_type=func.get('return_type', 'void'),
                metadata={
                    'language': func.get('language', 'sql'),
                    'is_deterministic': func.get('is_deterministic', False),
                    'type': 'sql_function'
                }
            )
            result.elements.append(element)
            result.metrics.function_count += 1

        # Procédures
        for proc in analysis['procedures']:
            element = CodeElement(
                name=proc['name'],
                element_type='procedure',
                parameters=proc.get('parameters', []),
                metadata={
                    'language': proc.get('language', 'sql'),
                    'type': 'sql_procedure'
                }
            )
            result.elements.append(element)
            result.metrics.function_count += 1

        # Déclencheurs (Triggers)
        for trigger in analysis['triggers']:
            element = CodeElement(
                name=trigger['name'],
                element_type='trigger',
                metadata={
                    'table': trigger.get('table', ''),
                    'timing': trigger.get('timing', ''),
                    'event': trigger.get('event', ''),
                    'type': 'sql_trigger'
                }
            )
            result.elements.append(element)
            result.metrics.function_count += 1

        # Index
        for index in analysis['indexes']:
            element = CodeElement(
                name=index['name'],
                element_type='index',
                metadata={
                    'table': index.get('table', ''),
                    'columns': index.get('columns', []),
                    'is_unique': index.get('is_unique', False),
                    'type': 'sql_index'
                }
            )
            result.elements.append(element)

    def _update_patterns(self, result: AnalysisResult, analysis: Dict[str, Any]) -> None:
        """Met à jour la détection de patterns"""
        patterns = result.patterns
        sql_analysis = analysis['analysis']

        # Détecter le type de base de données
        db_type = analysis.get('database_type', 'generic')
        if db_type != 'generic':
            patterns.libraries.append(db_type)

        # Détecter les patterns SQL
        if sql_analysis.get('has_transactions'):
            patterns.patterns.append('transaction_management')

        if sql_analysis.get('has_subqueries'):
            patterns.patterns.append('subqueries')

        if sql_analysis.get('has_functions'):
            patterns.patterns.append('aggregate_functions')

        if sql_analysis.get('has_cte'):
            patterns.patterns.append('common_table_expressions')

        if sql_analysis.get('has_window_functions'):
            patterns.patterns.append('window_functions')

        # Architecture hints
        if len(analysis['tables']) > 0:
            patterns.architecture_hints.append('database_schema')

        if len(analysis['views']) > 0:
            patterns.architecture_hints.append('data_abstraction')

        if len(analysis['functions']) > 0 or len(analysis['procedures']) > 0:
            patterns.architecture_hints.append('business_logic_in_db')

        if len(analysis['triggers']) > 0:
            patterns.architecture_hints.append('data_integrity_rules')

    def _update_dependencies(self, result: AnalysisResult, analysis: Dict[str, Any]) -> None:
        """Met à jour les informations de dépendances"""
        deps = result.dependencies

        # Tables référencées
        tables = analysis['tables']
        for table in tables:
            if table.get('operation') == 'REFERENCE':
                deps.internal_deps.append(f"table:{table['name']}")

        # Vues référencées
        for view in analysis['views']:
            deps.internal_deps.append(f"view:{view['name']}")

        # Fonctions et procédures référencées
        for func in analysis['functions']:
            deps.internal_deps.append(f"function:{func['name']}")

        for proc in analysis['procedures']:
            deps.internal_deps.append(f"procedure:{proc['name']}")

        # Dépendances externes (extensions, types de données spéciaux)
        db_type = analysis.get('database_type', 'generic')
        if db_type in ['postgresql', 'mysql', 'oracle', 'sqlserver']:
            deps.external_deps.append(db_type)

        # Détecter les types de données spéciaux
        content = "\n".join(str(v) for v in analysis.values())
        if 'SERIAL' in content or 'BIGSERIAL' in content:
            deps.external_deps.append('postgresql_serial')
        if 'JSONB' in content or 'JSON' in content:
            deps.external_deps.append('json_support')
        if 'GEOMETRY' in content or 'GEOGRAPHY' in content:
            deps.external_deps.append('spatial_support')

    def _update_security(self, result: AnalysisResult, analysis: Dict[str, Any]) -> None:
        """Met à jour l'analyse de sécurité"""
        security = result.security
        security_issues = analysis.get('security_issues', {})

        # Vulnérabilités détectées
        if security_issues.get('has_sql_injection_risk'):
            security.vulnerabilities.append({
                'type': 'sql_injection_risk',
                'severity': 'high',
                'description': 'Potential SQL injection vulnerability detected',
                'recommendation': 'Use parameterized queries or stored procedures'
            })
            security.warnings.append("Potential SQL injection risk detected")

        if security_issues.get('has_hardcoded_credentials'):
            security.vulnerabilities.append({
                'type': 'hardcoded_credentials',
                'severity': 'high',
                'description': 'Hardcoded credentials detected',
                'recommendation': 'Use environment variables or secure credential storage'
            })
            security.warnings.append("Hardcoded credentials detected")

        if security_issues.get('has_dangerous_permissions'):
            security.vulnerabilities.append({
                'type': 'dangerous_permissions',
                'severity': 'medium',
                'description': 'Dangerous database permissions granted',
                'recommendation': 'Follow principle of least privilege'
            })

        # Recommandations générales
        if len(analysis['tables']) > 0:
            security.recommendations.append("Consider adding audit columns (created_at, updated_at, created_by)")

        if not any('password' in col.lower() for table in analysis['tables'] for col in table.get('columns', [])):
            security.recommendations.append("Ensure password columns are properly hashed")

        # Calculer le score de sécurité
        security.security_score = self._calculate_security_score(analysis)

    def _update_language_specific(self, result: AnalysisResult, analysis: Dict[str, Any]) -> None:
        """Met à jour les données spécifiques au SQL"""
        result.language_specific = {
            'sql': {
                'database_type': analysis.get('database_type', 'generic'),
                'statement_count': len(analysis['statements']),
                'table_count': len(analysis['tables']),
                'view_count': len(analysis['views']),
                'function_count': len(analysis['functions']),
                'procedure_count': len(analysis['procedures']),
                'trigger_count': len(analysis['triggers']),
                'index_count': len(analysis['indexes']),
                'constraint_count': len(analysis['constraints']),
                'operations': analysis['operations'],
                'complexity_level': analysis['analysis'].get('complexity_level', 'LOW'),
                'is_migration_file': self._is_migration_file(analysis),
                'is_seed_file': self._is_seed_file(analysis),
                'is_schema_file': self._is_schema_file(analysis)
            }
        }

    def _add_diagnostics(self, result: AnalysisResult, analysis: Dict[str, Any]) -> None:
        """Ajoute des diagnostics et recommandations"""

        # Vérifier le nombre de tables
        table_count = len(analysis['tables'])
        if table_count == 0:
            result.notes.append("No table definitions found")
        elif table_count > 20:
            result.warnings.append(f"Large number of tables ({table_count}) - consider splitting schema")

        # Vérifier les contraintes
        constraint_count = len(analysis['constraints'])
        if constraint_count == 0 and table_count > 0:
            result.warnings.append("No constraints defined - consider adding primary/foreign keys")

        # Vérifier la complexité
        complexity = analysis['analysis'].get('complexity_level', 'LOW')
        if complexity == 'HIGH':
            result.notes.append("Complex SQL detected - consider simplifying or adding comments")

        # Vérifier les fonctions/procédures
        if len(analysis['functions']) > 10 or len(analysis['procedures']) > 10:
            result.warnings.append("Large amount of business logic in database - consider moving to application layer")

    def _extract_statements(self, content: str) -> List[Dict]:
        """Extrait les statements SQL"""
        statements = []
        # Normaliser le contenu
        normalized = re.sub(r'--.*$', '', content, flags=re.MULTILINE)
        normalized = re.sub(r'/\*.*?\*/', '', normalized, flags=re.DOTALL)

        # Détection des statements avec plus de détails
        statement_patterns = {
            'CREATE_TABLE': r'(CREATE\s+(?:TEMPORARY\s+)?TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)\s*\()',
            'CREATE_VIEW': r'(CREATE\s+(?:OR\s+REPLACE\s+)?VIEW\s+(\w+)\s+AS)',
            'CREATE_FUNCTION': r'(CREATE\s+(?:OR\s+REPLACE\s+)?FUNCTION\s+(\w+)\s*\()',
            'CREATE_PROCEDURE': r'(CREATE\s+(?:OR\s+REPLACE\s+)?PROCEDURE\s+(\w+)\s*\()',
            'CREATE_TRIGGER': r'(CREATE\s+(?:OR\s+REPLACE\s+)?TRIGGER\s+(\w+)\s+)',
            'CREATE_INDEX': r'(CREATE\s+(?:UNIQUE\s+)?INDEX\s+(\w+)\s+ON\s+(\w+))',
            'SELECT': r'(SELECT\s+.*?FROM\s+(\w+))',
            'INSERT': r'(INSERT\s+(?:INTO\s+)?(\w+))',
            'UPDATE': r'(UPDATE\s+(\w+)\s+SET)',
            'DELETE': r'(DELETE\s+FROM\s+(\w+))',
            'ALTER': r'(ALTER\s+TABLE\s+(\w+))',
            'DROP': r'(DROP\s+(?:TABLE|VIEW|FUNCTION|PROCEDURE|TRIGGER|INDEX)\s+(\w+))'
        }

        for stmt_type, pattern in statement_patterns.items():
            for match in re.finditer(pattern, normalized, re.IGNORECASE | re.DOTALL):
                groups = match.groups()
                statement_info = {
                    'type': stmt_type,
                    'full_statement': match.group(0)[:500],
                    'length': len(match.group(0))
                }

                # Ajouter des informations spécifiques selon le type
                if stmt_type == 'CREATE_TABLE' and len(groups) > 1:
                    statement_info['table_name'] = groups[1]
                elif stmt_type == 'CREATE_INDEX' and len(groups) > 2:
                    statement_info['index_name'] = groups[1]
                    statement_info['table_name'] = groups[2]
                elif len(groups) > 1:
                    statement_info['target_name'] = groups[1]

                statements.append(statement_info)

        return statements

    def _extract_tables(self, content: str) -> List[Dict]:
        """Extrait les informations sur les tables"""
        tables = []

        # Tables créées avec leurs colonnes
        create_pattern = r'CREATE\s+(?:TEMPORARY\s+)?TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)\s*\((.*?)\)(?:\s*;|$)'

        for match in re.finditer(create_pattern, content, re.IGNORECASE | re.DOTALL):
            table_name = match.group(1)
            columns_text = match.group(2)

            # Extraire les colonnes
            columns = []
            column_pattern = r'\s*(\w+)\s+([\w\(\)\s,]+)(?:\s+(?:NOT\s+NULL|NULL|PRIMARY\s+KEY|UNIQUE|DEFAULT\s+[^,]+))?'

            for col_match in re.finditer(column_pattern, columns_text, re.IGNORECASE):
                columns.append({
                    'name': col_match.group(1),
                    'type': col_match.group(2).strip(),
                    'definition': col_match.group(0).strip()
                })

            # Extraire les contraintes
            constraints = []
            constraint_patterns = {
                'PRIMARY_KEY': r'PRIMARY\s+KEY\s*\(([^)]+)\)',
                'FOREIGN_KEY': r'FOREIGN\s+KEY\s*\(([^)]+)\)\s+REFERENCES\s+(\w+)\s*\(([^)]+)\)',
                'UNIQUE': r'UNIQUE\s*\(([^)]+)\)',
                'CHECK': r'CHECK\s*\(([^)]+)\)'
            }

            for const_type, pattern in constraint_patterns.items():
                for const_match in re.finditer(pattern, columns_text, re.IGNORECASE):
                    constraint_info = {'type': const_type}
                    if const_type == 'FOREIGN_KEY':
                        constraint_info.update({
                            'columns': [c.strip() for c in const_match.group(1).split(',')],
                            'references_table': const_match.group(2),
                            'references_columns': [c.strip() for c in const_match.group(3).split(',')]
                        })
                    else:
                        constraint_info['columns'] = [c.strip() for c in const_match.group(1).split(',')]
                    constraints.append(constraint_info)

            tables.append({
                'name': table_name,
                'operation': 'CREATE',
                'columns': columns,
                'constraints': constraints,
                'column_count': len(columns),
                'constraint_count': len(constraints)
            })

        # Tables référencées
        ref_patterns = [
            r'FROM\s+(\w+)',
            r'JOIN\s+(\w+)',
            r'INTO\s+(\w+)',
            r'UPDATE\s+(\w+)'
        ]

        for pattern in ref_patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                table_name = match.group(1)
                # Éviter les doublons
                if not any(t['name'] == table_name for t in tables):
                    tables.append({
                        'name': table_name,
                        'operation': 'REFERENCE'
                    })

        return tables

    def _extract_columns(self, content: str) -> List[Dict]:
        """Extrait les informations sur les colonnes"""
        columns = []

        # Rechercher les définitions de colonnes dans CREATE TABLE
        table_matches = re.finditer(r'CREATE\s+TABLE\s+\w+\s*\((.*?)\)', content, re.IGNORECASE | re.DOTALL)

        for match in table_matches:
            table_body = match.group(1)
            # Extraire les lignes de colonnes
            column_lines = re.findall(r'\s*(\w+)\s+([\w\(\)]+)([^,]*)(?:,|$)', table_body)

            for col_name, col_type, col_constraints in column_lines:
                columns.append({
                    'name': col_name,
                    'type': col_type,
                    'constraints': self._parse_column_constraints(col_constraints),
                    'full_definition': f"{col_name} {col_type}{col_constraints}"
                })

        return columns

    def _parse_column_constraints(self, constraints_text: str) -> Dict[str, Any]:
        """Parse les contraintes de colonne"""
        constraints = {
            'not_null': 'NOT NULL' in constraints_text.upper(),
            'unique': 'UNIQUE' in constraints_text.upper(),
            'primary_key': 'PRIMARY KEY' in constraints_text.upper(),
            'has_default': 'DEFAULT' in constraints_text.upper(),
            'references': None
        }

        # Extraire la valeur par défaut
        default_match = re.search(r'DEFAULT\s+([^,\s]+)', constraints_text, re.IGNORECASE)
        if default_match:
            constraints['default_value'] = default_match.group(1)

        # Extraire les références FOREIGN KEY
        fk_match = re.search(r'REFERENCES\s+(\w+)\s*\((\w+)\)', constraints_text, re.IGNORECASE)
        if fk_match:
            constraints['references'] = {
                'table': fk_match.group(1),
                'column': fk_match.group(2)
            }

        return constraints

    def _extract_indexes(self, content: str) -> List[Dict]:
        """Extrait les informations sur les index"""
        indexes = []

        index_pattern = r'CREATE\s+(UNIQUE\s+)?INDEX\s+(\w+)\s+ON\s+(\w+)\s*\(([^)]+)\)'

        for match in re.finditer(index_pattern, content, re.IGNORECASE):
            indexes.append({
                'name': match.group(2),
                'table': match.group(3),
                'columns': [col.strip() for col in match.group(4).split(',')],
                'is_unique': bool(match.group(1))
            })

        return indexes

    def _extract_triggers(self, content: str) -> List[Dict]:
        """Extrait les informations sur les triggers"""
        triggers = []

        trigger_pattern = r'CREATE\s+(?:OR\s+REPLACE\s+)?TRIGGER\s+(\w+)\s+(BEFORE|AFTER|INSTEAD\s+OF)\s+(INSERT|UPDATE|DELETE)\s+ON\s+(\w+)'

        for match in re.finditer(trigger_pattern, content, re.IGNORECASE):
            triggers.append({
                'name': match.group(1),
                'timing': match.group(2),
                'event': match.group(3),
                'table': match.group(4)
            })

        return triggers

    def _extract_views(self, content: str) -> List[Dict]:
        """Extrait les informations sur les vues"""
        views = []

        view_pattern = r'CREATE\s+(?:OR\s+REPLACE\s+)?VIEW\s+(\w+)\s+AS\s+(.*?)(?=\s*(?:CREATE|ALTER|DROP|$))'

        for match in re.finditer(view_pattern, content, re.IGNORECASE | re.DOTALL):
            views.append({
                'name': match.group(1),
                'definition': match.group(2).strip()[:500],
                'definition_length': len(match.group(2))
            })

        return views

    def _extract_functions(self, content: str) -> List[Dict]:
        """Extrait les informations sur les fonctions"""
        functions = []

        func_pattern = r'CREATE\s+(?:OR\s+REPLACE\s+)?FUNCTION\s+(\w+)\s*\((.*?)\)\s*RETURNS\s+([\w\(\)]+)'

        for match in re.finditer(func_pattern, content, re.IGNORECASE | re.DOTALL):
            functions.append({
                'name': match.group(1),
                'parameters': self._parse_parameters(match.group(2)),
                'return_type': match.group(3).strip(),
                'language': self._extract_function_language(match.group(0))
            })

        return functions

    def _extract_procedures(self, content: str) -> List[Dict]:
        """Extrait les informations sur les procédures"""
        procedures = []

        proc_pattern = r'CREATE\s+(?:OR\s+REPLACE\s+)?PROCEDURE\s+(\w+)\s*\((.*?)\)'

        for match in re.finditer(proc_pattern, content, re.IGNORECASE | re.DOTALL):
            procedures.append({
                'name': match.group(1),
                'parameters': self._parse_parameters(match.group(2))
            })

        return procedures

    def _parse_parameters(self, params_text: str) -> List[str]:
        """Parse les paramètres de fonction/procédure"""
        if not params_text.strip():
            return []

        params = []
        param_parts = re.split(r'\s*,\s*', params_text)

        for param in param_parts:
            # Format: param_name param_type
            param_match = re.match(r'(\w+)\s+([\w\(\)]+)', param.strip())
            if param_match:
                params.append(f"{param_match.group(1)} {param_match.group(2)}")
            else:
                params.append(param.strip())

        return params

    def _extract_function_language(self, function_text: str) -> str:
        """Extrait le langage de la fonction"""
        language_match = re.search(r'LANGUAGE\s+(\w+)', function_text, re.IGNORECASE)
        return language_match.group(1).lower() if language_match else 'sql'

    def _analyze_operations(self, content: str) -> Dict[str, int]:
        """Analyse les opérations SQL"""
        operations = {
            'SELECT': len(re.findall(r'\bSELECT\b', content, re.IGNORECASE)),
            'INSERT': len(re.findall(r'\bINSERT\b', content, re.IGNORECASE)),
            'UPDATE': len(re.findall(r'\bUPDATE\b', content, re.IGNORECASE)),
            'DELETE': len(re.findall(r'\bDELETE\b', content, re.IGNORECASE)),
            'CREATE': len(re.findall(r'\bCREATE\b', content, re.IGNORECASE)),
            'ALTER': len(re.findall(r'\bALTER\b', content, re.IGNORECASE)),
            'DROP': len(re.findall(r'\bDROP\b', content, re.IGNORECASE)),
            'JOIN': len(re.findall(r'\bJOIN\b', content, re.IGNORECASE)),
            'UNION': len(re.findall(r'\bUNION\b', content, re.IGNORECASE)),
            'WHERE': len(re.findall(r'\bWHERE\b', content, re.IGNORECASE)),
            'GROUP_BY': len(re.findall(r'\bGROUP\s+BY\b', content, re.IGNORECASE)),
            'ORDER_BY': len(re.findall(r'\bORDER\s+BY\b', content, re.IGNORECASE)),
            'HAVING': len(re.findall(r'\bHAVING\b', content, re.IGNORECASE))
        }
        return operations

    def _extract_constraints(self, content: str) -> List[Dict]:
        """Extrait les contraintes"""
        constraints = []
        constraint_patterns = {
            'PRIMARY_KEY': (r'PRIMARY\s+KEY\s*(?:\(([^)]+)\))?', ['columns']),
            'FOREIGN_KEY': (r'FOREIGN\s+KEY\s*\(([^)]+)\)\s+REFERENCES\s+(\w+)\s*\(([^)]+)\)', ['columns', 'references_table', 'references_columns']),
            'UNIQUE': (r'UNIQUE\s*(?:\(([^)]+)\))?', ['columns']),
            'NOT_NULL': (r'NOT\s+NULL', []),
            'CHECK': (r'CHECK\s*\(([^)]+)\)', ['condition'])
        }

        for const_type, (pattern, groups) in constraint_patterns.items():
            for match in re.finditer(pattern, content, re.IGNORECASE):
                constraint_info = {'type': const_type}

                for i, group_name in enumerate(groups, 1):
                    if i <= len(match.groups()) and match.group(i):
                        if group_name == 'columns' or group_name == 'references_columns':
                            constraint_info[group_name] = [c.strip() for c in match.group(i).split(',')]
                        else:
                            constraint_info[group_name] = match.group(i)

                constraints.append(constraint_info)

        return constraints

    def _analyze_sql_patterns(self, content: str) -> Dict[str, Any]:
        """Analyse les patterns SQL"""
        return {
            'has_transactions': bool(re.search(r'\b(BEGIN|START TRANSACTION|COMMIT|ROLLBACK)\b', content, re.IGNORECASE)),
            'has_subqueries': bool(re.search(r'\(.*SELECT.*\)', content, re.IGNORECASE | re.DOTALL)),
            'has_functions': len(re.findall(r'\b(COUNT|SUM|AVG|MAX|MIN|COALESCE|NULLIF)\b', content, re.IGNORECASE)) > 0,
            'has_cte': bool(re.search(r'WITH\s+\w+\s+AS', content, re.IGNORECASE)),
            'has_window_functions': bool(re.search(r'\b(OVER|PARTITION BY|RANK|ROW_NUMBER)\b', content, re.IGNORECASE)),
            'has_dynamic_sql': bool(re.search(r'EXECUTE\s+IMMEDIATE|EXEC\s*\(|sp_executesql', content, re.IGNORECASE)),
            'complexity_level': self._calculate_complexity(content)
        }

    def _calculate_complexity(self, content: str) -> str:
        """Calcule le niveau de complexité"""
        join_count = len(re.findall(r'\bJOIN\b', content, re.IGNORECASE))
        subquery_count = len(re.findall(r'\(.*SELECT.*\)', content, re.IGNORECASE | re.DOTALL))
        union_count = len(re.findall(r'\bUNION\b', content, re.IGNORECASE))
        cte_count = len(re.findall(r'\bWITH\b', content, re.IGNORECASE))

        total_complexity = join_count + (subquery_count * 2) + union_count + cte_count

        if total_complexity > 8:
            return 'HIGH'
        elif total_complexity > 3:
            return 'MEDIUM'
        else:
            return 'LOW'

    def _detect_database_type(self, content: str) -> str:
        """Détecte le type de base de données"""
        # PostgreSQL
        if re.search(r'\b(SERIAL|BIGSERIAL|JSONB|ARRAY|INTERVAL)\b', content, re.IGNORECASE):
            return 'postgresql'

        # MySQL
        if re.search(r'\b(ENGINE=InnoDB|AUTO_INCREMENT|UNSIGNED)\b', content, re.IGNORECASE):
            return 'mysql'

        # SQL Server
        if re.search(r'\b(GO|NVARCHAR|DATETIME2|WITH\s+\(NOLOCK\))\b', content, re.IGNORECASE):
            return 'sqlserver'

        # Oracle
        if re.search(r'\b(NUMBER\(|VARCHAR2|NVL|TO_DATE)\b', content, re.IGNORECASE):
            return 'oracle'

        # SQLite
        if re.search(r'\b(AUTOINCREMENT|WITHOUT ROWID)\b', content, re.IGNORECASE):
            return 'sqlite'

        return 'generic'

    def _detect_security_issues(self, content: str) -> Dict[str, bool]:
        """Détecte les problèmes de sécurité"""
        return {
            'has_sql_injection_risk': bool(re.search(r'\$\d+|%s|\{\w+\}', content)),  # Placeholders non paramétrés
            'has_hardcoded_credentials': bool(re.search(r'PASSWORD\s*=\s*[\'"][^\'"]+[\'"]', content, re.IGNORECASE)),
            'has_dangerous_permissions': bool(re.search(r'GRANT\s+(ALL|SUPERUSER|DBA)\s+TO', content, re.IGNORECASE)),
            'has_dynamic_sql_execution': bool(re.search(r'EXECUTE\s+IMMEDIATE|EXEC\s*\(|sp_executesql', content, re.IGNORECASE)),
            'has_weak_crypto': bool(re.search(r'MD5\(|SHA1\(', content, re.IGNORECASE))
        }

    def _calculate_sql_metrics(self, content: str, analysis: Dict[str, Any]) -> FileMetrics:
        """Calcule les métriques spécifiques au SQL"""
        metrics = super()._calculate_metrics(content)

        # Mettre à jour avec les métriques spécifiques au SQL
        metrics.function_count = len(analysis['functions']) + len(analysis['procedures']) + len(analysis['triggers'])
        metrics.class_count = len(analysis['tables']) + len(analysis['views'])
        metrics.import_count = len(analysis['statements'])

        # Calculer la complexité
        complexity_level = analysis['analysis'].get('complexity_level', 'LOW')
        complexity_scores = {'LOW': 1.0, 'MEDIUM': 3.0, 'HIGH': 6.0}
        metrics.complexity_score = complexity_scores.get(complexity_level, 1.0)

        # Compter les lignes de code SQL (exclure les commentaires)
        lines = content.split('\n')
        sql_lines = len([l for l in lines if l.strip() and not l.strip().startswith('--') and not l.strip().startswith('/*')])
        metrics.code_lines = sql_lines

        return metrics

    def _calculate_security_score(self, analysis: Dict[str, Any]) -> float:
        """Calcule un score de sécurité pour SQL"""
        score = 100.0
        security_issues = analysis.get('security_issues', {})

        # Pénalités pour les problèmes de sécurité
        if security_issues.get('has_sql_injection_risk'):
            score -= 40

        if security_issues.get('has_hardcoded_credentials'):
            score -= 30

        if security_issues.get('has_dangerous_permissions'):
            score -= 20

        if security_issues.get('has_dynamic_sql_execution'):
            score -= 15

        if security_issues.get('has_weak_crypto'):
            score -= 10

        # Bonus pour les bonnes pratiques
        if len(analysis['constraints']) > 0:
            score += 5

        if analysis['analysis'].get('has_transactions'):
            score += 5

        return max(0, min(100, score))

    def _is_migration_file(self, analysis: Dict[str, Any]) -> bool:
        """Détermine si c'est un fichier de migration"""
        file_path = analysis.get('file_path', '')
        filename = file_path.lower() if file_path else ''

        # Vérifier le nom du fichier
        migration_patterns = ['migration', 'migrate', 'upgrade', 'downgrade', 'version']
        if any(pattern in filename for pattern in migration_patterns):
            return True

        # Vérifier le contenu
        operations = analysis['operations']
        if operations.get('CREATE', 0) > 0 or operations.get('ALTER', 0) > 0:
            return True

        return False

    def _is_seed_file(self, analysis: Dict[str, Any]) -> bool:
        """Détermine si c'est un fichier de données (seed)"""
        operations = analysis['operations']
        return operations.get('INSERT', 0) > operations.get('SELECT', 0)

    def _is_schema_file(self, analysis: Dict[str, Any]) -> bool:
        """Détermine si c'est un fichier de schéma"""
        operations = analysis['operations']
        return operations.get('CREATE', 0) > 0 and operations.get('INSERT', 0) == 0