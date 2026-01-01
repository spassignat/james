# sql_analyzer.py
import logging
import re
from typing import Dict, List, Any

from james_parsers.analyzer import Analyzer

logger = logging.getLogger(__name__)


class SQLAnalyzer(Analyzer):
    def analyze(self, file_path: str) -> Dict[str, Any]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return self.analyze_content(content, file_path)
        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {e}")
            return {
                'file_type': 'sql',
                'file_path': file_path,
                'is_valid': False,
                'error': str(e)
            }

    def analyze_content(self, content: str, file_path: str) -> Dict[str, Any]:
        return {
            'file_type': 'sql',
            'statements': self._extract_statements(content),
            'tables': self._extract_tables(content),
            'operations': self._analyze_operations(content),
            'constraints': self._extract_constraints(content),
            'analysis': self._analyze_sql_patterns(content)
        }

    def _extract_statements(self, content: str) -> List[Dict]:
        statements = []
        # Normaliser le contenu
        normalized = re.sub(r'--.*$', '', content, flags=re.MULTILINE)  # Remove comments
        normalized = re.sub(r'/\*.*?\*/', '', normalized, flags=re.DOTALL)  # Remove block comments

        # Détection des statements
        statement_patterns = {
            'CREATE_TABLE': r'CREATE\s+TABLE\s+(\w+)',
            'SELECT': r'SELECT\s+.*?FROM\s+(\w+)',
            'INSERT': r'INSERT\s+INTO\s+(\w+)',
            'UPDATE': r'UPDATE\s+(\w+)',
            'DELETE': r'DELETE\s+FROM\s+(\w+)',
            'ALTER': r'ALTER\s+TABLE\s+(\w+)'
        }

        for stmt_type, pattern in statement_patterns.items():
            for match in re.finditer(pattern, normalized, re.IGNORECASE):
                statements.append({
                    'type': stmt_type,
                    'target': match.group(1),
                    'content': match.group(0)[:200]  # Preview
                })

        return statements

    def _extract_tables(self, content: str) -> List[Dict]:
        tables = []
        # Tables créées
        create_matches = re.finditer(r'CREATE\s+TABLE\s+(\w+)', content, re.IGNORECASE)
        for match in create_matches:
            tables.append({
                'name': match.group(1),
                'operation': 'CREATE'
            })

        # Tables référencées
        ref_matches = re.finditer(r'FROM\s+(\w+)', content, re.IGNORECASE)
        for match in ref_matches:
            tables.append({
                'name': match.group(1),
                'operation': 'REFERENCE'
            })

        return tables

    def _analyze_operations(self, content: str) -> Dict[str, int]:
        operations = {
            'SELECT': len(re.findall(r'\bSELECT\b', content, re.IGNORECASE)),
            'INSERT': len(re.findall(r'\bINSERT\b', content, re.IGNORECASE)),
            'UPDATE': len(re.findall(r'\bUPDATE\b', content, re.IGNORECASE)),
            'DELETE': len(re.findall(r'\bDELETE\b', content, re.IGNORECASE)),
            'JOIN': len(re.findall(r'\bJOIN\b', content, re.IGNORECASE))
        }
        return operations

    def _extract_constraints(self, content: str) -> List[Dict]:
        constraints = []
        constraint_patterns = {
            'PRIMARY_KEY': r'PRIMARY\s+KEY',
            'FOREIGN_KEY': r'FOREIGN\s+KEY',
            'UNIQUE': r'UNIQUE',
            'NOT_NULL': r'NOT\s+NULL',
            'CHECK': r'CHECK\s*\('
        }

        for const_type, pattern in constraint_patterns.items():
            if re.search(pattern, content, re.IGNORECASE):
                constraints.append({'type': const_type})

        return constraints

    def _analyze_sql_patterns(self, content: str) -> Dict[str, Any]:
        return {
            'has_transactions': 'BEGIN' in content.upper() or 'COMMIT' in content.upper(),
            'has_subqueries': re.search(r'\(.*SELECT.*\)', content, re.IGNORECASE) is not None,
            'has_functions': len(re.findall(r'\b(COUNT|SUM|AVG|MAX|MIN)\b', content, re.IGNORECASE)) > 0,
            'complexity_level': self._calculate_complexity(content)
        }

    def _calculate_complexity(self, content: str) -> str:
        join_count = len(re.findall(r'\bJOIN\b', content, re.IGNORECASE))
        subquery_count = len(re.findall(r'\(.*SELECT.*\)', content, re.IGNORECASE))

        if join_count > 3 or subquery_count > 2:
            return 'HIGH'
        elif join_count > 1 or subquery_count > 0:
            return 'MEDIUM'
        else:
            return 'LOW'
