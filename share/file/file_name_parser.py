# src/utils/filename_parser.py
import re
from typing import List, Dict, Any, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class FilenameParser:
    """Parser pour découper et analyser les noms de fichiers selon différentes conventions"""

    # Patterns pour différentes conventions de nommage
    PATTERNS = {
        'snake_case': re.compile(r'^[a-z][a-z0-9_]*(_[a-z0-9]+)*$'),
        'camel_case': re.compile(r'^[a-z][a-zA-Z0-9]*$'),
        'pascal_case': re.compile(r'^[A-Z][a-zA-Z0-9]*$'),
        'kebab_case': re.compile(r'^[a-z][a-z0-9]*(-[a-z0-9]+)*$'),
        'upper_snake_case': re.compile(r'^[A-Z][A-Z0-9_]*(_[A-Z0-9]+)*$'),
        'mixed_case': re.compile(r'^[a-zA-Z0-9]+([A-Z][a-z0-9]+)*$')
    }

    # Mots clés courants dans les noms de fichiers
    KEYWORDS = {
        'types': ['model', 'view', 'controller', 'service', 'repository', 'component',
                  'module', 'util', 'helper', 'config', 'settings', 'constant', 'enum',
                  'interface', 'abstract', 'base', 'main', 'app', 'test', 'spec', 'mock'],

        'actions': ['get', 'set', 'create', 'update', 'delete', 'remove', 'add', 'find',
                    'search', 'fetch', 'load', 'save', 'import', 'export', 'generate',
                    'validate', 'check', 'verify', 'process', 'handle', 'manage'],

        'prefixes': ['is', 'has', 'can', 'should', 'will', 'did', 'was'],

        'suffixes': ['manager', 'handler', 'factory', 'builder', 'provider', 'adapter',
                     'decorator', 'strategy', 'observer', 'command', 'query', 'event']
    }

    @staticmethod
    def detect_convention(filename: str) -> str:
        """Détecte la convention de nommage utilisée"""
        name_without_ext = Path(filename).stem

        for convention, pattern in FilenameParser.PATTERNS.items():
            if pattern.match(name_without_ext):
                return convention

        # Si aucune convention standard ne correspond
        return 'unknown'

    @staticmethod
    def parse_filename(filename: str) -> Dict[str, Any]:
        """
        Analyse un nom de fichier et retourne une structure détaillée

        Args:
            filename: Nom du fichier (peut inclure le chemin)

        Returns:
            Dict avec l'analyse détaillée
        """
        path = Path(filename)
        stem = path.stem
        extension = path.suffix.lower()

        # Détecter la convention
        convention = FilenameParser.detect_convention(stem)

        # Découper en mots/tokens selon la convention
        tokens = FilenameParser._split_filename(stem, convention)

        # Analyser les tokens
        analysis = FilenameParser._analyze_tokens(tokens)

        return {
            'original_filename': filename,
            'stem': stem,
            'extension': extension,
            'convention': convention,
            'tokens': tokens,
            'analysis': analysis,
            'suggested_name': FilenameParser._suggest_improved_name(stem, tokens, convention),
            'metadata': {
                'has_keywords': analysis['has_keywords'],
                'keyword_types': analysis['keyword_types'],
                'is_test_file': FilenameParser._is_test_file(stem),
                'is_config_file': FilenameParser._is_config_file(stem),
                'is_main_file': FilenameParser._is_main_file(stem)
            }
        }

    @staticmethod
    def _split_filename(filename_stem: str, convention: str) -> List[str]:
        """Découpe un nom de fichier en tokens selon sa convention"""
        if convention == 'snake_case':
            return filename_stem.split('_')
        elif convention == 'camel_case' or convention == 'pascal_case':
            # Découper camelCase ou PascalCase
            tokens = re.findall(r'[A-Z]?[a-z0-9]+', filename_stem)
            return [token.lower() for token in tokens]
        elif convention == 'kebab_case':
            return filename_stem.split('-')
        elif convention == 'upper_snake_case':
            return [token.lower() for token in filename_stem.split('_')]
        else:
            # Pour les noms inconnus, essayer plusieurs méthodes
            # Essayer de découper par majuscules d'abord
            tokens = re.findall(r'[A-Z]?[a-z0-9]+', filename_stem)
            if len(tokens) > 1:
                return [token.lower() for token in tokens]

            # Essayer de découper par underscores
            if '_' in filename_stem:
                return [token.lower() for token in filename_stem.split('_')]

            # Essayer de découper par tirets
            if '-' in filename_stem:
                return [token.lower() for token in filename_stem.split('-')]

            # Sinon, retourner le nom entier
            return [filename_stem.lower()]

    @staticmethod
    def _analyze_tokens(tokens: List[str]) -> Dict[str, Any]:
        """Analyse les tokens pour extraire des informations sémantiques"""
        keyword_types = set()
        found_keywords = []

        for token in tokens:
            # Chercher dans chaque catégorie de mots clés
            for category, keywords in FilenameParser.KEYWORDS.items():
                if token in keywords:
                    keyword_types.add(category)
                    found_keywords.append({
                        'token': token,
                        'category': category
                    })

        # Détecter des patterns communs
        patterns = []
        if len(tokens) >= 2:
            # Pattern: Action + Type (ex: getUser, createService)
            if tokens[0] in FilenameParser.KEYWORDS['actions']:
                patterns.append('action_type')

            # Pattern: Type + Action (ex: userService, dataManager)
            if tokens[-1] in FilenameParser.KEYWORDS['suffixes']:
                patterns.append('type_suffix')

            # Pattern: Prefix + Type (ex: isActive, hasPermission)
            if tokens[0] in FilenameParser.KEYWORDS['prefixes']:
                patterns.append('prefix_type')

        return {
            'has_keywords': len(found_keywords) > 0,
            'keyword_types': list(keyword_types),
            'found_keywords': found_keywords,
            'patterns_detected': patterns,
            'token_count': len(tokens),
            'is_descriptive': len(tokens) >= 2 and len(found_keywords) > 0
        }

    @staticmethod
    def _suggest_improved_name(stem: str, tokens: List[str], convention: str) -> str:
        """Suggère un nom amélioré si nécessaire"""
        analysis = FilenameParser._analyze_tokens(tokens)

        # Si le nom est déjà descriptif, le garder
        if analysis['is_descriptive']:
            return stem

        # Suggestions basées sur les tokens existants
        if len(tokens) == 1:
            token = tokens[0]

            # Si c'est un type, suggérer un nom plus complet
            if token in FilenameParser.KEYWORDS['types']:
                if convention == 'snake_case':
                    return f"{token}_manager"
                elif convention == 'camel_case':
                    return f"{token}Manager"
                elif convention == 'pascal_case':
                    return f"{token.title()}Manager"

        return stem  # Pas de suggestion

    @staticmethod
    def _is_test_file(filename_stem: str) -> bool:
        """Vérifie si c'est un fichier de test"""
        test_patterns = [
            r'_test$', r'_spec$', r'test_', r'spec_',
            r'\.test$', r'\.spec$', r'Test$', r'Spec$'
        ]

        for pattern in test_patterns:
            if re.search(pattern, filename_stem, re.IGNORECASE):
                return True

        return False

    @staticmethod
    def _is_config_file(filename_stem: str) -> bool:
        """Vérifie si c'est un fichier de configuration"""
        config_patterns = [
            r'config', r'configuration', r'settings', r'options',
            r'env', r'\.env', r'properties', r'\.properties',
            r'yml$', r'\.yml', r'yaml$', r'\.yaml',
            r'json$', r'\.json', r'toml$', r'\.toml',
            r'ini$', r'\.ini'
        ]

        for pattern in config_patterns:
            if re.search(pattern, filename_stem, re.IGNORECASE):
                return True

        return False

    @staticmethod
    def _is_main_file(filename_stem: str) -> bool:
        """Vérifie si c'est un fichier principal"""
        main_patterns = [
            r'^main$', r'^app$', r'^index$', r'^application$',
            r'^server$', r'^client$', r'^start$', r'^run$'
        ]

        for pattern in main_patterns:
            if re.match(pattern, filename_stem, re.IGNORECASE):
                return True

        return False

    @staticmethod
    def normalize_to_snake_case(filename_stem: str) -> str:
        """Convertit n'importe quel nom en snake_case"""
        # D'abord, découper le nom
        tokens = FilenameParser._split_filename(filename_stem, 'unknown')
        # Rejoindre avec underscores
        return '_'.join(tokens)

    @staticmethod
    def normalize_to_camel_case(filename_stem: str) -> str:
        """Convertit n'importe quel nom en camelCase"""
        tokens = FilenameParser._split_filename(filename_stem, 'unknown')
        if not tokens:
            return filename_stem

        # Premier mot en minuscule, autres avec majuscule initiale
        result = tokens[0]
        for token in tokens[1:]:
            result += token.title()

        return result

    @staticmethod
    def normalize_to_pascal_case(filename_stem: str) -> str:
        """Convertit n'importe quel nom en PascalCase"""
        tokens = FilenameParser._split_filename(filename_stem, 'unknown')
        return ''.join(token.title() for token in tokens)

    @staticmethod
    def extract_semantic_info(filename: str) -> Dict[str, Any]:
        """Extrait des informations sémantiques d'un nom de fichier"""
        parsed = FilenameParser.parse_filename(filename)

        # Déduire le type de fichier basé sur le nom et l'extension
        file_type = FilenameParser._deduce_file_type(parsed['stem'], parsed['extension'])

        # Déduire le rôle/domaine
        domain = FilenameParser._deduce_domain(parsed['tokens'])

        return {
            'parsed': parsed,
            'deduced': {
                'file_type': file_type,
                'domain': domain,
                'purpose': FilenameParser._deduce_purpose(parsed['tokens']),
                'layer': FilenameParser._deduce_layer(parsed['tokens']),
                'is_core': FilenameParser._is_core_file(parsed['tokens'])
            }
        }

    @staticmethod
    def _deduce_file_type(stem: str, extension: str) -> str:
        """Déduit le type de fichier"""
        if FilenameParser._is_test_file(stem):
            return 'test'
        elif FilenameParser._is_config_file(stem):
            return 'configuration'
        elif FilenameParser._is_main_file(stem):
            return 'entry_point'
        elif extension in ['.py', '.js', '.ts', '.java', '.cpp', '.go', '.rs']:
            return 'source_code'
        elif extension in ['.md', '.txt', '.rst']:
            return 'documentation'
        elif extension in ['.yml', '.yaml', '.json', '.toml', '.ini']:
            return 'configuration'
        elif extension in ['.sql', '.db']:
            return 'database'
        else:
            return 'other'

    @staticmethod
    def _deduce_domain(tokens: List[str]) -> str:
        """Déduit le domaine du fichier"""
        # Chercher des mots clés de domaine communs
        domain_keywords = {
            'user': ['user', 'auth', 'login', 'register', 'profile', 'account'],
            'data': ['data', 'model', 'entity', 'database', 'db', 'schema'],
            'api': ['api', 'rest', 'graphql', 'endpoint', 'route', 'controller'],
            'ui': ['ui', 'view', 'component', 'page', 'screen', 'layout'],
            'business': ['service', 'business', 'logic', 'process', 'workflow'],
            'utils': ['util', 'helper', 'tool', 'common', 'shared', 'base']
        }

        for domain, keywords in domain_keywords.items():
            for token in tokens:
                if token in keywords:
                    return domain

        return 'general'

    @staticmethod
    def _deduce_purpose(tokens: List[str]) -> str:
        """Déduit le but du fichier"""
        # Analyser les premiers tokens pour le but
        if tokens and tokens[0] in FilenameParser.KEYWORDS['actions']:
            return 'action'
        elif any(token in FilenameParser.KEYWORDS['types'] for token in tokens):
            return 'definition'
        elif any(token in FilenameParser.KEYWORDS['suffixes'] for token in tokens):
            return 'implementation'
        else:
            return 'general'

    @staticmethod
    def _deduce_layer(tokens: List[str]) -> str:
        """Déduit la couche architecturale"""
        layer_keywords = {
            'presentation': ['view', 'controller', 'presenter', 'component', 'ui'],
            'business': ['service', 'usecase', 'interactor', 'business', 'logic'],
            'data': ['repository', 'dao', 'data', 'model', 'entity', 'schema'],
            'infrastructure': ['config', 'database', 'cache', 'queue', 'external']
        }

        for layer, keywords in layer_keywords.items():
            for token in tokens:
                if token in keywords:
                    return layer

        return 'unknown'

    @staticmethod
    def _is_core_file(tokens: List[str]) -> bool:
        """Détermine si c'est un fichier cœur de l'application"""
        core_keywords = ['main', 'app', 'application', 'core', 'base', 'common']
        return any(token in core_keywords for token in tokens)