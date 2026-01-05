# models/analysis_result.py
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional


class AnalysisStatus(Enum):
    """Statut de l'analyse"""
    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"
    SKIPPED = "skipped"
    PENDING = "pending"


class FileType(Enum):
    """Types de fichiers supportés"""
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    PYTHON = "python"
    JAVA = "java"
    VUE = "vue"
    HTML = "html"
    CSS = "css"
    JSON = "json"
    YAML = "yaml"
    XML = "xml"
    SQL = "sql"
    DOCKERFILE = "dockerfile"
    SHELL = "shell"
    MARKDOWN = "markdown"
    UNKNOWN = "unknown"


class FrameworkType(Enum):
    """Frameworks détectés"""
    VANILLA = "vanilla"
    REACT = "react"
    VUE = "vue"
    ANGULAR = "angular"
    EXPRESS = "express"
    DJANGO = "django"
    FLASK = "flask"
    SPRING = "spring"
    NEXTJS = "nextjs"
    NUXT = "nuxt"
    SVELTE = "svelte"


@dataclass
class CodeElement:
    """Élément de code (fonction, classe, etc.)"""
    name: str
    element_type: str  # 'function', 'class', 'method', 'variable', 'import', 'export'
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    parameters: List[str] = field(default_factory=list)
    return_type: Optional[str] = None
    modifiers: List[str] = field(default_factory=list)  # ['async', 'static', 'public', 'private']
    metadata: Dict[str, Any] = field(default_factory=dict)
    content: Optional[str] = None  # Contenu brut de l'élément


@dataclass
class FileMetrics:
    """Métriques d'analyse du fichier"""
    total_lines: int = 0
    code_lines: int = 0
    comment_lines: int = 0
    blank_lines: int = 0
    function_count: int = 0
    class_count: int = 0
    import_count: int = 0
    export_count: int = 0
    complexity_score: float = 0.0
    file_size_bytes: int = 0
    average_line_length: float = 0.0


@dataclass
class PatternDetection:
    """Détection de patterns dans le code"""
    patterns: List[str] = field(default_factory=list)  # ['mvc', 'repository', 'singleton']
    frameworks: List[FrameworkType] = field(default_factory=list)
    libraries: List[str] = field(default_factory=list)  # ['axios', 'lodash', 'moment']
    architecture_hints: List[str] = field(default_factory=list)  # ['microservice', 'monolith', 'spa']


@dataclass
class SecurityAnalysis:
    """Analyse de sécurité"""
    vulnerabilities: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    security_score: float = 0.0  # 0-100


@dataclass
class DependencyInfo:
    """Information sur les dépendances"""
    imports: List[Dict[str, Any]] = field(default_factory=list)
    exports: List[Dict[str, Any]] = field(default_factory=list)
    internal_deps: List[str] = field(default_factory=list)  # Dépendances internes
    external_deps: List[str] = field(default_factory=list)  # Dépendances externes
    package_manager: Optional[str] = None  # 'npm', 'pip', 'maven', etc.


@dataclass
class FileNamingAnalysis:
    """Analyse du nom de fichier"""
    original_name: str
    stem: str
    extension: str
    convention: str  # 'snake_case', 'camel_case', etc.
    tokens: List[str] = field(default_factory=list)
    suggested_name: Optional[str] = None
    is_test_file: bool = False
    is_config_file: bool = False
    is_main_file: bool = False
    file_type: Optional[FileType] = None
    domain: Optional[str] = None  # 'user', 'data', 'api', etc.
    layer: Optional[str] = None  # 'presentation', 'business', 'data', 'infrastructure'
    purpose: Optional[str] = None  # 'action', 'definition', 'implementation'


@dataclass
class SectionAnalysis:
    """Analyse d'une section (pour fichiers multi-sections comme Vue.js)"""
    section_type: str  # 'template', 'script', 'style', 'setup'
    content: Optional[str] = None
    elements: List[CodeElement] = field(default_factory=list)
    metrics: FileMetrics = field(default_factory=FileMetrics)
    language: Optional[str] = None
    analysis: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    """Résultat complet de l'analyse d'un fichier"""
    recommendations: str

    # Informations de base
    file_path: str
    file_type: FileType
    status: AnalysisStatus = AnalysisStatus.SUCCESS
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Métadonnées
    filename: str = ""
    file_size: int = 0
    last_modified: Optional[str] = None
    encoding: str = "utf-8"

    # Analyses
    elements: List[CodeElement] = field(default_factory=list)
    metrics: FileMetrics = field(default_factory=FileMetrics)
    dependencies: DependencyInfo = field(default_factory=DependencyInfo)
    patterns: PatternDetection = field(default_factory=PatternDetection)
    security: SecurityAnalysis = field(default_factory=SecurityAnalysis)
    naming_analysis: Optional[FileNamingAnalysis] = None

    # Pour fichiers multi-sections (Vue.js, etc.)
    sections: Dict[str, SectionAnalysis] = field(default_factory=dict)

    # Données spécifiques au langage/framework
    language_specific: Dict[str, Any] = field(default_factory=dict)

    # Diagnostics
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    # Métadonnées de l'analyse
    analyzer_version: str = "1.0.0"
    analyzer_name: str = "CodeAnalyzer"
    processing_time_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire pour sérialisation JSON"""
        result = {
            "file_path": self.file_path,
            "file_type": self.file_type.value,
            "status": self.status.value,
            "timestamp": self.timestamp,
            "filename": self.filename,
            "file_size": self.file_size,
            "metrics": {
                "total_lines": self.metrics.total_lines,
                "code_lines": self.metrics.code_lines,
                "comment_lines": self.metrics.comment_lines,
                "blank_lines": self.metrics.blank_lines,
                "function_count": self.metrics.function_count,
                "class_count": self.metrics.class_count,
                "import_count": self.metrics.import_count,
                "export_count": self.metrics.export_count,
                "complexity_score": self.metrics.complexity_score,
                "file_size_bytes": self.metrics.file_size_bytes,
                "average_line_length": self.metrics.average_line_length
            },
            "elements": [
                {
                    "name": elem.name,
                    "type": elem.element_type,
                    "line_start": elem.line_start,
                    "line_end": elem.line_end,
                    "parameters": elem.parameters,
                    "modifiers": elem.modifiers,
                    "metadata": elem.metadata
                }
                for elem in self.elements
            ],
            "dependencies": {
                "imports": self.dependencies.imports,
                "exports": self.dependencies.exports,
                "internal_deps": self.dependencies.internal_deps,
                "external_deps": self.dependencies.external_deps
            },
            "patterns": {
                "patterns": self.patterns.patterns,
                "frameworks": [f.value for f in self.patterns.frameworks],
                "libraries": self.patterns.libraries,
                "architecture_hints": self.patterns.architecture_hints
            },
            "security": {
                "vulnerabilities": self.security.vulnerabilities,
                "warnings": self.security.warnings,
                "recommendations": self.security.recommendations,
                "security_score": self.security.security_score
            },
            "diagnostics": {
                "errors": self.errors,
                "warnings": self.warnings,
                "notes": self.notes
            },
            "sections": {
                section_type: {
                    "section_type": section.section_type,
                    "language": section.language,
                    "metrics": {
                        "total_lines": section.metrics.total_lines,
                        "code_lines": section.metrics.code_lines
                    },
                    "analysis": section.analysis
                }
                for section_type, section in self.sections.items()
            },
            "language_specific": self.language_specific,
            "metadata": {
                "analyzer_version": self.analyzer_version,
                "analyzer_name": self.analyzer_name,
                "processing_time_ms": self.processing_time_ms
            }
        }

        if self.naming_analysis:
            result["naming_analysis"] = {
                "original_name": self.naming_analysis.original_name,
                "stem": self.naming_analysis.stem,
                "extension": self.naming_analysis.extension,
                "convention": self.naming_analysis.convention,
                "tokens": self.naming_analysis.tokens,
                "file_type": self.naming_analysis.file_type.value if self.naming_analysis.file_type else None,
                "domain": self.naming_analysis.domain,
                "layer": self.naming_analysis.layer,
                "purpose": self.naming_analysis.purpose,
                "is_test_file": self.naming_analysis.is_test_file,
                "is_config_file": self.naming_analysis.is_config_file,
                "is_main_file": self.naming_analysis.is_main_file
            }

        return result

    def to_json(self, indent: int = 2) -> str:
        """Convertit en JSON"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False, default=str)

    def is_valid(self) -> bool:
        """Vérifie si l'analyse est valide"""
        return self.status == AnalysisStatus.SUCCESS or self.status == AnalysisStatus.PARTIAL

    def has_errors(self) -> bool:
        """Vérifie s'il y a des erreurs"""
        return bool(self.errors) or self.status == AnalysisStatus.ERROR

    def get_summary(self) -> Dict[str, Any]:
        """Retourne un résumé de l'analyse"""
        return {
            "file_path": self.file_path,
            "file_type": self.file_type.value,
            "status": self.status.value,
            "elements_count": len(self.elements),
            "frameworks": [f.value for f in self.patterns.frameworks],
            "has_security_issues": self.security.security_score < 80,
            "complexity": self.metrics.complexity_score,
            "has_errors": self.has_errors()
        }

    # parsers/analysis_result.py


from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime


class ComplexityLevel(Enum):
    """Niveaux de complexité du code"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"
    CRITICAL = "critical"


@dataclass
class CodeMetrics:
    """Métriques d'analyse de code"""

    # Métriques de base
    lines_of_code: int = 0
    comment_lines: int = 0
    empty_lines: int = 0
    total_lines: int = 0
    comment_ratio: float = 0.0
    average_line_length: float = 0.0
    max_line_length: int = 0

    # Métriques de complexité
    function_count: int = 0
    class_count: int = 0
    method_count: int = 0
    cyclomatic_complexity: int = 0
    cognitive_complexity: int = 0
    nesting_depth: int = 0
    complexity_score: int = 0
    complexity_level: ComplexityLevel = ComplexityLevel.LOW

    # Métriques de structure
    import_count: int = 0
    export_count: int = 0
    variable_count: int = 0
    constant_count: int = 0
    interface_count: int = 0
    type_alias_count: int = 0

    # Métriques de qualité
    duplicated_lines: int = 0
    duplicated_blocks: int = 0
    long_methods: int = 0  # Méthodes > 20 lignes
    long_files: int = 0  # Fichiers > 500 lignes
    high_complexity_functions: int = 0  # Fonctions avec complexité > 10

    # Métriques de test (si applicable)
    test_count: int = 0
    test_coverage: float = 0.0
    assertion_count: int = 0

    # Métriques de sécurité
    security_issues: int = 0
    vulnerability_count: int = 0

    # Métriques de performance
    performance_hints: int = 0
    memory_issues: int = 0

    # Métriques temporelles
    analysis_time_ms: int = 0
    parsing_time_ms: int = 0

    # Métriques spécifiques au langage
    language_specific_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire"""
        result = asdict(self)
        result['complexity_level'] = self.complexity_level.value
        return result

    def calculate_comment_ratio(self) -> None:
        """Calcule le ratio commentaires/code"""
        if self.total_lines > 0:
            self.comment_ratio = round(self.comment_lines / self.total_lines, 3)

    def update_total_lines(self) -> None:
        """Met à jour le nombre total de lignes"""
        self.total_lines = self.lines_of_code + self.comment_lines + self.empty_lines

    def calculate_complexity_level(self) -> None:
        """Détermine le niveau de complexité basé sur le score"""
        if self.complexity_score >= 50:
            self.complexity_level = ComplexityLevel.CRITICAL
        elif self.complexity_score >= 30:
            self.complexity_level = ComplexityLevel.VERY_HIGH
        elif self.complexity_score >= 20:
            self.complexity_level = ComplexityLevel.HIGH
        elif self.complexity_score >= 10:
            self.complexity_level = ComplexityLevel.MEDIUM
        else:
            self.complexity_level = ComplexityLevel.LOW

    def get_summary(self) -> Dict[str, Any]:
        """Retourne un résumé des métriques"""
        return {
            'lines': {
                'total': self.total_lines,
                'code': self.lines_of_code,
                'comments': self.comment_lines,
                'empty': self.empty_lines,
                'comment_ratio': self.comment_ratio
            },
            'structure': {
                'functions': self.function_count,
                'classes': self.class_count,
                'methods': self.method_count,
                'imports': self.import_count,
                'exports': self.export_count
            },
            'complexity': {
                'score': self.complexity_score,
                'level': self.complexity_level.value,
                'cyclomatic': self.cyclomatic_complexity,
                'cognitive': self.cognitive_complexity,
                'nesting': self.nesting_depth
            },
            'quality': {
                'duplicated_lines': self.duplicated_lines,
                'long_methods': self.long_methods,
                'long_files': self.long_files,
                'high_complexity_functions': self.high_complexity_functions
            }
        }

    def is_high_complexity(self) -> bool:
        """Vérifie si le code a une complexité élevée"""
        return self.complexity_level in [
            ComplexityLevel.HIGH,
            ComplexityLevel.VERY_HIGH,
            ComplexityLevel.CRITICAL
        ]

    def is_large_file(self) -> bool:
        """Vérifie si le fichier est considéré comme gros"""
        return self.total_lines > 1000

    def has_quality_issues(self) -> bool:
        """Vérifie s'il y a des problèmes de qualité"""
        return (
                self.duplicated_lines > 10 or
                self.long_methods > 5 or
                self.high_complexity_functions > 3 or
                self.complexity_level == ComplexityLevel.CRITICAL
        )

    def has_security_issues(self) -> bool:
        """Vérifie s'il y a des problèmes de sécurité"""
        return self.security_issues > 0 or self.vulnerability_count > 0

    def get_quality_score(self) -> float:
        """Calcule un score de qualité (0-100)"""
        score = 100

        # Pénalités pour la complexité
        if self.complexity_level == ComplexityLevel.CRITICAL:
            score -= 40
        elif self.complexity_level == ComplexityLevel.VERY_HIGH:
            score -= 25
        elif self.complexity_level == ComplexityLevel.HIGH:
            score -= 15
        elif self.complexity_level == ComplexityLevel.MEDIUM:
            score -= 5

        # Pénalités pour les lignes dupliquées
        if self.duplicated_lines > 50:
            score -= 20
        elif self.duplicated_lines > 20:
            score -= 10
        elif self.duplicated_lines > 10:
            score -= 5

        # Pénalités pour les méthodes longues
        score -= min(self.long_methods * 2, 20)

        # Pénalités pour les fichiers longs
        if self.is_large_file():
            score -= 10

        # Bonus pour la documentation
        if self.comment_ratio > 0.2:  # Plus de 20% de commentaires
            score += 10
        elif self.comment_ratio > 0.1:  # Plus de 10% de commentaires
            score += 5

        return max(0, min(100, round(score, 1)))


@dataclass
class AnalysisResult:
    """Résultat d'analyse d'un fichier"""

    # Informations de base
    file_path: str
    filename: str
    file_type: FileType
    file_size: int
    status: AnalysisStatus
    last_modified: float
    naming_analysis: FileNamingAnalysis
    processing_time_ms: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Résultats d'analyse
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    # Métriques de code
    metrics: CodeMetrics = field(default_factory=CodeMetrics)

    # Éléments de code
    elements: List[CodeElement] = field(default_factory=list)

    # Dépendances
    dependencies: DependencyInfo = field(default_factory=DependencyInfo)

    # Patterns et frameworks
    patterns: PatternDetection = field(default_factory=PatternDetection)

    # Analyse de sécurité
    security: SecurityAnalysis = field(default_factory=SecurityAnalysis)

    # Données spécifiques au langage
    language_specific: Dict[str, Any] = field(default_factory=dict)

    # Tags et catégories
    tags: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    # Informations supplémentaires
    module_name: str = None
    package_name: str = None
    version: str = None
    rule_count: int = None
    imports:List[str] = field(default_factory=list)

    def __init__(self,
                 file_path: str,
                 filename: str,
                 file_type: FileType,
                 file_size: int,
                 last_modified: float,
                 status: AnalysisStatus,
                 naming_analysis: FileNamingAnalysis,
                 processing_time_ms: Optional[int],
                 language_specific: Optional[Dict[str, Any]] = {}
                 ) -> None:
        self.file_path = file_path
        self.filename = filename
        self.file_type = file_type
        self.file_size = file_size
        self.last_modified = last_modified
        self.naming_analysis = naming_analysis
        self.errors = []
        self.categories = []
        self.imports = []
        self.patterns = PatternDetection()
        self.dependencies = DependencyInfo()
        self.status = status
        self.warnings = []
        self.notes = []
        self.recommendations = []
        self.tags = []
        self.elements = []
        self.metrics = CodeMetrics()
        self.processing_time_ms = processing_time_ms
        self.language_specific = language_specific
        self.security=SecurityAnalysis()


def to_dict(self) -> Dict[str, Any]:
    """Convertit en dictionnaire"""
    return {
        'file_path': self.file_path,
        'file_type': self.file_type.value,
        'status': self.status.value,
        'processing_time_ms': self.processing_time_ms,
        'timestamp': self.timestamp,
        'warnings': self.warnings,
        'errors': self.errors,
        'metrics': self.metrics.to_dict(),
        'elements': [elem.to_dict() for elem in self.elements],
        'dependencies': self.dependencies.to_dict(),
        'patterns': self.patterns.to_dict(),
        'security': self.security.to_dict(),
        'language_specific': self.language_specific,
        'tags': self.tags,
        'categories': self.categories,
        'module_name': self.module_name,
        'package_name': self.package_name,
        'rule_count': self.rule_count,
        'version': self.version
    }


def is_successful(self) -> bool:
    """Vérifie si l'analyse a réussi"""
    return self.status == AnalysisStatus.SUCCESS


def has_errors(self) -> bool:
    """Vérifie s'il y a des erreurs"""
    return len(self.errors) > 0 or self.status == AnalysisStatus.ERROR


def has_warnings(self) -> bool:
    """Vérifie s'il y a des avertissements"""
    return len(self.warnings) > 0


def add_warning(self, warning: str) -> None:
    """Ajoute un avertissement"""
    self.warnings.append(warning)


def add_error(self, error: str) -> None:
    """Ajoute une erreur"""
    self.errors.append(error)
    self.status = AnalysisStatus.ERROR


def add_tag(self, tag: str) -> None:
    """Ajoute un tag"""
    if tag not in self.tags:
        self.tags.append(tag)


def add_category(self, category: str) -> None:
    """Ajoute une catégorie"""
    if category not in self.categories:
        self.categories.append(category)


def get_summary(self) -> Dict[str, Any]:
    """Retourne un résumé de l'analyse"""
    return {
        'file': Path(self.file_path).name,
        'status': self.status.value,
        'processing_time': f"{self.processing_time_ms}ms",
        'metrics_summary': self.metrics.get_summary(),
        'elements_count': len(self.elements),
        'warnings_count': len(self.warnings),
        'errors_count': len(self.errors),
        'dependencies_count': len(self.dependencies.imports) + len(self.dependencies.exports),
        'security_score': self.security.security_score,
        'quality_score': self.metrics.get_quality_score()
    }


def validate(self) -> bool:
    """Valide le résultat d'analyse"""
    if not self.file_path:
        self.add_error("Chemin de fichier manquant")
        return False

    if not isinstance(self.metrics, CodeMetrics):
        self.add_error("Métriques invalides")
        return False

    if self.status == AnalysisStatus.SUCCESS and not self.elements:
        self.add_warning("Aucun élément de code détecté")

    return True


# parsers/analysis_result.py (suite)

class AnalysisStatus(Enum):
    """Statut de l'analyse"""
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCESS = "success"
    ERROR = "error"
    SKIPPED = "skipped"
    WARNING = "warning"
    PARTIAL = "partial"


class FileType(Enum):
    """Types de fichiers supportés"""
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    PYTHON = "python"
    JAVA = "java"
    CSHARP = "csharp"
    CPP = "cpp"
    GO = "go"
    RUST = "rust"
    RUBY = "ruby"
    PHP = "php"
    SQL = "sql"
    HTML = "html"
    CSS = "css"
    JSON = "json"
    YAML = "yaml"
    VUE = "vue"
    XML = "xml"
    MARKDOWN = "markdown"
    DOCKERFILE = "dockerfile"
    SHELL = "shell"
    UNKNOWN = "unknown"


# parsers/analysis_result.py (suite)

@dataclass
class CodeElement:
    """Élément de code (fonction, classe, méthode, etc.)"""

    name: str
    element_type: str  # 'function', 'class', 'method', 'variable', 'interface', etc.

    # Informations de localisation
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    column_start: Optional[int] = None
    column_end: Optional[int] = None

    # Propriétés de l'élément
    parameters: List[str] = field(default_factory=list)
    return_type: Optional[str] = None
    modifiers: List[str] = field(default_factory=list)  # 'public', 'private', 'static', 'async', etc.
    access_level: str = "public"  # 'public', 'private', 'protected', 'internal'

    # Métriques spécifiques
    complexity: int = 0
    lines_of_code: int = 0
    parameter_count: int = 0

    # Métadonnées supplémentaires
    metadata: Dict[str, Any] = field(default_factory=dict)
    documentation: Optional[str] = None
    annotations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire"""
        return {
            'name': self.name,
            'element_type': self.element_type,
            'line_start': self.line_start,
            'line_end': self.line_end,
            'column_start': self.column_start,
            'column_end': self.column_end,
            'parameters': self.parameters,
            'return_type': self.return_type,
            'modifiers': self.modifiers,
            'access_level': self.access_level,
            'complexity': self.complexity,
            'lines_of_code': self.lines_of_code,
            'parameter_count': self.parameter_count,
            'metadata': self.metadata,
            'documentation': self.documentation,
            'annotations': self.annotations
        }

    def get_signature(self) -> str:
        """Retourne la signature de l'élément"""
        modifiers = " ".join(self.modifiers) + " " if self.modifiers else ""
        params = ", ".join(self.parameters)
        return_type = f": {self.return_type}" if self.return_type else ""

        return f"{modifiers}{self.name}({params}){return_type}"

    def is_public(self) -> bool:
        """Vérifie si l'élément est public"""
        return self.access_level == "public" or "public" in self.modifiers

    def is_private(self) -> bool:
        """Vérifie si l'élément est privé"""
        return self.access_level == "private" or "private" in self.modifiers

    def is_static(self) -> bool:
        """Vérifie si l'élément est statique"""
        return "static" in self.modifiers

    def is_async(self) -> bool:
        """Vérifie si l'élément est asynchrone"""
        return "async" in self.modifiers


@dataclass
class DependencyInfo:
    """Informations sur les dépendances"""

    imports: List[Dict[str, Any]] = field(default_factory=list)
    exports: List[Dict[str, Any]] = field(default_factory=list)

    # Catégorisation des dépendances
    internal_deps: List[str] = field(default_factory=list)  # Chemins relatifs
    external_deps: List[str] = field(default_factory=list)  # Paquets externes
    package_deps: List[str] = field(default_factory=list)  # Noms de paquets
    peer_deps: List[str] = field(default_factory=list)  # Dépendances peer

    # Métadonnées
    package_manager: Optional[str] = None  # 'npm', 'pip', 'maven', etc.
    package_file: Optional[str] = None  # 'package.json', 'requirements.txt', etc.
    version_constraints: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire"""
        return {
            'imports': self.imports,
            'exports': self.exports,
            'internal_deps': self.internal_deps,
            'external_deps': self.external_deps,
            'package_deps': self.package_deps,
            'peer_deps': self.peer_deps,
            'package_manager': self.package_manager,
            'package_file': self.package_file,
            'version_constraints': self.version_constraints
        }

    def get_all_deps(self) -> List[str]:
        """Retourne toutes les dépendances"""
        return self.internal_deps + self.external_deps + self.package_deps

    def has_dependencies(self) -> bool:
        """Vérifie s'il y a des dépendances"""
        return bool(self.get_all_deps())


class FrameworkType(Enum):
    """Types de frameworks détectés"""
    REACT = "react"
    VUE = "vue"
    ANGULAR = "angular"
    SPRING = "spring"
    DJANGO = "django"
    FLASK = "flask"
    EXPRESS = "express"
    NESTJS = "nestjs"
    LARAVEL = "laravel"
    RAILS = "rails"
    DOTNET = "dotnet"
    FLUTTER = "flutter"
    REACT_NATIVE = "react_native"
    VUE_NATIVE = "vue_native"


@dataclass
class PatternDetection:
    """Détection de patterns et frameworks"""

    patterns: List[str] = field(default_factory=list)  # Patterns de code
    frameworks: List[FrameworkType] = field(default_factory=list)  # Frameworks détectés
    libraries: List[str] = field(default_factory=list)  # Bibliothèques populaires
    architecture_hints: List[str] = field(default_factory=list)  # Indices architecturaux

    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire"""
        return {
            'patterns': self.patterns,
            'frameworks': [fw.value for fw in self.frameworks],
            'libraries': self.libraries,
            'architecture_hints': self.architecture_hints
        }

    def has_pattern(self, pattern: str) -> bool:
        """Vérifie si un pattern est présent"""
        return pattern in self.patterns

    def has_framework(self, framework: FrameworkType) -> bool:
        """Vérifie si un framework est présent"""
        return framework in self.frameworks


@dataclass
class SecurityAnalysis:
    """Analyse de sécurité"""

    warnings: List[str] = field(default_factory=list)
    vulnerabilities: List[Dict[str, Any]] = field(default_factory=list)
    security_score: int = 100  # 0-100, 100 = parfait
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire"""
        return {
            'warnings': self.warnings,
            'vulnerabilities': self.vulnerabilities,
            'security_score': self.security_score,
            'recommendations': self.recommendations
        }

    def add_vulnerability(self,
                          vuln_type: str,
                          severity: str,
                          description: str,
                          location: Optional[str] = None) -> None:
        """Ajoute une vulnérabilité"""
        self.vulnerabilities.append({
            'type': vuln_type,
            'severity': severity,
            'description': description,
            'location': location
        })

    def update_security_score(self) -> None:
        """Met à jour le score de sécurité"""
        score = 100

        for vuln in self.vulnerabilities:
            if vuln['severity'] == 'critical':
                score -= 30
            elif vuln['severity'] == 'high':
                score -= 20
            elif vuln['severity'] == 'medium':
                score -= 10
            elif vuln['severity'] == 'low':
                score -= 5

        self.security_score = max(0, min(100, score))

    def is_secure(self) -> bool:
        """Vérifie si le code est sécurisé"""
        return self.security_score >= 70 and not any(
            v['severity'] in ['critical', 'high'] for v in self.vulnerabilities
        )
