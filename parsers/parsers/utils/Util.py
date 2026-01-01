import re


def infer_language_from_path(file_path: str) -> str:
    """Déduit le langage depuis le chemin du fichier"""
    if file_path.endswith(".java"):
        return "java"
    elif file_path.endswith(".js"):
        return "javascript"
    elif file_path.endswith(".ts"):
        return "typescript"
    elif file_path.endswith(".vue"):
        return "vue"
    elif file_path.endswith(".py"):
        return "python"
    else:
        return "unknown"


def infer_category_from_type(chunk_type: str, file_path: str) -> str:
    """Déduit la catégorie du chunk"""
    path_lower = file_path.lower()
    if "controller" in chunk_type or "controller" in path_lower:
        return "controller"
    elif "service" in chunk_type or "service" in path_lower:
        return "service"
    elif "repository" in chunk_type or "repository" in path_lower:
        return "repository"
    elif "component" in chunk_type or "component" in path_lower:
        return "component"
    elif "config" in chunk_type or "config" in path_lower:
        return "configuration"
    else:
        return "business_logic"


def _detect_language(content: str, extension: str) -> str:
    """Détecte le langage de programmation"""
    language_map = {
        '.js': 'javascript',
        '.jsx': 'javascript',
        '.ts': 'typescript',
        '.tsx': 'typescript',
        '.py': 'python',
        '.java': 'java',
        '.cpp': 'cpp',
        '.c': 'c',
        '.cs': 'csharp',
        '.rb': 'ruby',
        '.php': 'php',
        '.go': 'go',
        '.rs': 'rust',
        '.swift': 'swift',
        '.kt': 'kotlin',
        '.scala': 'scala',
        '.html': 'html',
        '.css': 'css',
        '.json': 'json',
        '.yml': 'yaml',
        '.yaml': 'yaml',
        '.md': 'markdown',
        '.sql': 'sql',
        '.xml': 'xml'
    }

    # D'abord par extension
    lang = language_map.get(extension.lower(), 'unknown')

    # Validation par contenu pour certains cas ambigus
    if lang == 'unknown' or lang == 'text':
        content_lower = content.lower()
        if 'def ' in content and ':' in content and not content.strip().startswith('//'):
            return 'python'
        elif 'function ' in content_lower or 'const ' in content_lower or 'let ' in content_lower:
            return 'javascript'
        elif 'public class' in content_lower or 'private ' in content_lower or 'system.out.println' in content_lower:
            return 'java'
        elif '#include <' in content_lower or 'std::' in content_lower:
            return 'cpp'
        elif '<?php' in content_lower or '$_' in content_lower:
            return 'php'
        elif '<!doctype html>' in content_lower or '<html>' in content_lower:
            return 'html'
        elif 'package ' in content_lower and 'import ' in content_lower:
            return 'java'

    return lang


def _detect_frameworks(content: str, extension: str) -> List[str]:
    """Détecte les frameworks et bibliothèques"""
    frameworks = []
    content_lower = content.lower()

    framework_checks = [
        ('react', ['react', 'usestate', 'useeffect', 'react.createclass']),
        ('vue', ['vue', 'vue.component', 'v-model', 'v-for', 'v-if']),
        ('angular', ['angular', '@component', '@injectable', 'ngoninit']),
        ('express', ['express', 'app.get', 'app.post', 'express()']),
        ('django', ['django', 'from django', 'django.views']),
        ('flask', ['flask', '@app.route', 'flask(', 'from flask']),
        ('spring', ['@restcontroller', '@service', '@autowired', 'springbootapplication']),
        ('laravel', ['laravel', 'route::', 'eloquent', 'artisan']),
        ('nextjs', ['next/', 'next/router', 'getserversideprops']),
        ('nuxtjs', ['nuxt/', 'nuxt.config', 'asyncdata']),
        ('nestjs', ['@nestjs', 'nestjs', '@controller', '@module']),
        ('fastapi', ['fastapi', '@app.get', 'fastapi()']),
        ('jquery', ['jquery', '$(']),
        ('bootstrap', ['bootstrap', 'data-bs-', 'btn-primary']),
        ('tailwind', ['tailwind', 'class=".*-\\[.*\\]']),
    ]

    for framework_name, indicators in framework_checks:
        for indicator in indicators:
            if indicator in content_lower:
                frameworks.append(framework_name)
                break

    return list(set(frameworks))


def _detect_patterns(content: str) -> List[str]:
    """Détecte les patterns de conception"""
    patterns = []
    content_lower = content.lower()

    pattern_checks = [
        ('singleton', [r'getinstance\(\)', r'instance\s*=\s*null', r'private\s+constructor']),
        ('factory', [r'factory\s+method', r'create[a-z][a-za-z]*\(', r'factory[a-z][a-za-z]*\s+class']),
        ('observer', [r'addobserver', r'notifyobservers', r'\.notify\(', r'implements.*observer']),
        ('decorator', [r'decorator', r'@[a-z][a-za-z]*\s*\(', r'extends.*decorator']),
        ('adapter', [r'adapter\s+class', r'implements.*adapter', r'adapts.*to']),
        ('strategy', [r'strategy\s+pattern', r'implements.*strategy', r'strategy[a-z][a-za-z]*\s+interface']),
        ('dependency_injection', [r'@inject', r'@autowired', r'injectable', r'dependencyinjection']),
        ('repository', [r'repository\s+pattern', r'extends.*repository', r'implements.*repository']),
        ('service', [r'service\s+layer', r'@service', r'service[a-z][a-za-z]*\s+class']),
        ('mvc', [r'controller', r'@controller', r'@restcontroller', r'model.*view.*controller'])
    ]

    for pattern_name, indicators in pattern_checks:
        for indicator in indicators:
            if re.search(indicator, content_lower):
                patterns.append(pattern_name)
                break

    return list(set(patterns))


def _auto_detect_features(content: str, extension: str) -> Dict[str, Any]:
    """Détecte automatiquement des features dans le contenu"""
    features = {
        'language': _detect_language(content, extension),
        'has_functions': bool(re.search(r'function\s+\w+|def\s+\w+|\w+\s*=\s*\([^)]*\)\s*=>', content)),
        'has_classes': bool(re.search(r'class\s+\w+', content)),
        'has_comments': bool(re.search(r'//|#|/\*|\*/|<!--|-->', content)),
        'line_count': len(content.split('\n')),
        'word_count': len(content.split()),
    }

    # Détection de patterns
    patterns = _detect_patterns(content)
    if patterns:
        features['patterns'] = patterns

    # Détection de frameworks
    frameworks = _detect_frameworks(content, extension)
    if frameworks:
        features['frameworks'] = frameworks

    return features