# src/agents/enhanced_architecture_agent.py
import logging
import re
from typing import Dict, Any, List, Optional

from agents.base_agent import BaseAgent
from vector.vector_store import VectorStore

logger = logging.getLogger(__name__)


class EnhancedArchitectureAgent(BaseAgent):
    """Agent d'architecture amélioré avec analyse détaillée du code"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, 'enhanced_architecture_agent')
        self.analysis_depth = config.get('analysis', {}).get('depth', 'detailed')

    def analyze(self, context: Dict[str, Any], vector_store: VectorStore) -> Dict[str, Any]:
        """Analyse approfondie de l'architecture avec extraction de descriptions"""
        logger.info(f"🔍 Début analyse approfondie par {self.agent_name}")

        project_structure = context.get('project_structure', {})
        chunks = context.get('chunks', [])
        file_patterns = context.get('file_patterns', {})

        # Analyse préliminaire des chunks
        code_analysis = self._pre_analyze_chunks(chunks)

        prompt = self._build_detailed_architecture_prompt(
            project_structure,
            chunks,
            file_patterns,
            code_analysis
        )

        system_prompt = """Vous êtes un architecte logiciel expert spécialisé en rétro-ingénierie.
Votre tâche est d'analyser le code source pour extraire:
1. L'architecture et les patterns
2. Les descriptions qui auraient permis de générer ce code
3. Les intentions de conception
4. Les spécifications implicites

Fournissez une analyse technique détaillée avec des spécifications précises."""

        response = self._call_llm(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.1  # Température basse pour plus de précision
        )

        # Analyse supplémentaire pour les descriptions détaillées
        detailed_descriptions = self._extract_detailed_descriptions(response, chunks)

        return {
            'type': 'detailed_architecture_analysis',
            'agent': self.agent_name,
            'timestamp': self._get_timestamp(),
            'content': response,
            'summary': self._extract_detailed_summary(response),
            'recommendations': self._extract_recommendations(response),
            'patterns_identified': self._extract_patterns(response),
            'code_descriptions': detailed_descriptions,
            'architectural_elements': self._extract_architectural_elements(response),
            'data_models': self._extract_data_models(response),
            'api_specifications': self._extract_api_specs(chunks),
            'business_logic': self._extract_business_logic(response, chunks)
        }

    def _pre_analyze_chunks(self, chunks: List[Dict]) -> Dict[str, Any]:
        """Analyse préliminaire des chunks pour extraire des informations structurelles"""
        analysis = {
            'classes': [],
            'functions': [],
            'imports': set(),
            'dependencies': [],
            'entry_points': [],
            'data_structures': [],
            'configurations': []
        }

        for chunk in chunks:
            if isinstance(chunk, dict):
                content = chunk.get('content', '')
                metadata = chunk.get('metadata', {})

                # Extraire les classes
                classes = self._extract_classes(content)
                if classes:
                    analysis['classes'].extend(classes)

                # Extraire les fonctions
                functions = self._extract_functions(content)
                if functions:
                    analysis['functions'].extend(functions)

                # Extraire les imports
                imports = self._extract_imports(content)
                analysis['imports'].update(imports)

                # Identifier les points d'entrée
                if self._is_entry_point(content, metadata):
                    analysis['entry_points'].append({
                        'filename': metadata.get('filename', 'unknown'),
                        'type': self._identify_entry_point_type(content)
                    })

                # Identifier les structures de données
                data_structures = self._identify_data_structures(content)
                if data_structures:
                    analysis['data_structures'].extend(data_structures)

        return analysis

    def _build_detailed_architecture_prompt(self,
                                            structure: Dict,
                                            chunks: List,
                                            file_patterns: Dict,
                                            code_analysis: Dict) -> str:
        """Construit un prompt détaillé pour l'analyse d'architecture"""

        # Échantillonner des chunks représentatifs avec analyse
        sampled_chunks = self._sample_and_analyze_chunks(chunks, max_samples=15)

        prompt = f"""
# ANALYSE DÉTAILLÉE D'ARCHITECTURE ET DE CONCEPTION

## CONTEXTE DU PROJET
Analyse rétrospective d'un codebase pour extraire les spécifications de conception.

## 1. STRUCTURE DU PROJET
{self._format_detailed_structure(structure)}

## 2. ANALYSE PRÉLIMINAIRE DU CODE
{self._format_code_analysis(code_analysis)}

## 3. PATTERNS DE FICHIERS
{self._format_file_patterns(file_patterns)}

## 4. EXTRAITS DE CODE AVEC ANALYSE
{sampled_chunks}

## TÂCHE D'ANALYSE

En tant qu'architecte en rétro-ingénierie, vous devez fournir:

### SECTION A: ARCHITECTURE DÉTAILLÉE
1. **Architecture Technique**
   - Style architectural (MVC, Clean, Hexagonal, etc.)
   - Organisation des couches (présentation, métier, données)
   - Patterns architecturaux utilisés
   - Communication entre composants

2. **Composants et Modules**
   - Description détaillée de chaque module principal
   - Responsabilités et contrats
   - Dépendances internes/externes
   - Points d'extension

### SECTION B: SPÉCIFICATIONS DE CONCEPTION
Pour chaque élément important du code, fournissez:

1. **Description de Conception**
   - Qu'est-ce que cet élément fait ?
   - Pourquoi a-t-il été conçu ainsi ?
   - Quels problèmes résout-il ?

2. **Spécifications qui auraient permis de générer ce code**:
   Pour les classes importantes:
        Classe: [Nom]
        Objectif: [Description de l'objectif]
        Responsabilités: [Liste des responsabilités]
        Contrats: [Interfaces, méthodes publiques]
        Contraintes: [Limitations, pré-conditions]
        
3. **Modèles de Données**
- Entités principales
- Relations entre entités
- Validations et règles métier
- Schémas de persistance

### SECTION C: LOGIQUE MÉTIER
1. **Règles Métier Identifiées**
- Règles de validation
- Règles de calcul
- Workflows et processus

2. **API et Interfaces**
- Points d'entrée externes
- Contrats d'API
- Formats de données

### SECTION D: DIAGRAMMES (Mermaid)
1. Diagramme de classes
2. Diagramme de séquence pour les flux principaux
3. Diagramme d'architecture

### SECTION E: DOCUMENTATION GÉNÉRATIVE
Pour permettre la régénération ou l'extension du code, fournissez:
- Spécifications techniques détaillées
- Templates de code
- Guidelines de développement
- Patterns à réutiliser

**Format de réponse**: Structurez votre réponse avec les sections ci-dessus.
**Détail**: Soyez aussi précis et technique que possible.
"""
        return prompt

    def _sample_and_analyze_chunks(self, chunks: List, max_samples: int = 15) -> str:
        """Échantillonne et analyse des chunks représentatifs"""
        if not chunks:
            return "Aucun extrait de code disponible."

        # Sélectionner les chunks les plus informatifs
        informative_chunks = []
        for chunk in chunks:
            if isinstance(chunk, dict):
                content = chunk.get('content', '')
                score = self._calculate_informativeness_score(content)
                informative_chunks.append((score, chunk))

        # Trier par score et prendre les meilleurs
        informative_chunks.sort(key=lambda x: x[0], reverse=True)
        selected_chunks = [chunk for _, chunk in informative_chunks[:max_samples]]

        # Formater avec analyse
        formatted = []
        for i, chunk in enumerate(selected_chunks):
            if isinstance(chunk, dict):
                content = chunk.get('content', '')
                metadata = chunk.get('metadata', {})
                filename = metadata.get('filename', 'Inconnu')
                language = metadata.get('language', 'code')

                # Analyse du chunk
                analysis = self._analyze_single_chunk(content)

                formatted.append(f"### 📄 Fichier {i + 1}: {filename} ({language})")
                formatted.append(f"**Localisation**: {metadata.get('file_path', 'N/A')}")
                formatted.append(f"**Type**: {metadata.get('chunk_type', 'code')}")

                if analysis['classes']:
                    formatted.append("**Classes identifiées**:")
                    for cls in analysis['classes'][:3]:
                        formatted.append(f"- {cls}")

                if analysis['functions']:
                    formatted.append("**Fonctions identifiées**:")
                    for func in analysis['functions'][:3]:
                        formatted.append(f"- {func}")

                formatted.append("**Contenu (extrait)**:")
                formatted.append(f"```{language}")
                formatted.append(content[:400] + ("..." if len(content) > 400 else ""))
                formatted.append("```")

                formatted.append("**Analyse**:")
                formatted.append(analysis['summary'])

                formatted.append("")  # Ligne vide entre les chunks

        return '\n'.join(formatted)

    def _calculate_informativeness_score(self, content: str) -> int:
        """Calcule un score d'informativité pour un chunk de code"""
        score = 0

        # Points pour les éléments structurels
        if re.search(r'class\s+\w+', content):
            score += 10
        if re.search(r'interface\s+\w+', content):
            score += 8
        if re.search(r'function\s+\w+', content):
            score += 5
        if re.search(r'def\s+\w+', content):
            score += 5
        if re.search(r'export\s+', content):
            score += 3

        # Points pour les patterns
        if re.search(r'implements|extends', content):
            score += 7
        if re.search(r'@\w+', content):  # Annotations/decorators
            score += 4

        # Points pour la complexité
        lines = content.split('\n')
        if len(lines) > 20:
            score += 3  # Code plus long = potentiellement plus informatif

        return score

    def _analyze_single_chunk(self, content: str) -> Dict[str, Any]:
        """Analyse un chunk de code individuel"""
        analysis = {
            'classes': self._extract_classes(content),
            'functions': self._extract_functions(content),
            'imports': list(self._extract_imports(content)),
            'patterns': self._identify_patterns(content),
            'complexity': self._estimate_complexity(content),
            'summary': ''
        }

        # Générer un résumé
        summary_parts = []
        if analysis['classes']:
            summary_parts.append(f"Contient {len(analysis['classes'])} classe(s)")
        if analysis['functions']:
            summary_parts.append(f"{len(analysis['functions'])} fonction(s)")
        if analysis['patterns']:
            summary_parts.append(f"Patterns: {', '.join(analysis['patterns'][:3])}")

        analysis['summary'] = ' | '.join(summary_parts) if summary_parts else "Code basique"

        return analysis

    def _extract_classes(self, content: str) -> List[str]:
        """Extrait les noms de classes du code"""
        classes = []

        # Java/C#/TypeScript
        patterns = [
            r'class\s+(\w+)',
            r'interface\s+(\w+)',
            r'struct\s+(\w+)',
            r'enum\s+(\w+)',
            r'@Entity\s*\n\s*class\s+(\w+)',
            r'@Service\s*\n\s*class\s+(\w+)'
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, content, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                classes.append(match.group(1))

        return list(set(classes))

    def _extract_functions(self, content: str) -> List[str]:
        """Extrait les noms de fonctions/méthodes"""
        functions = []

        # Multi-langage
        patterns = [
            r'function\s+(\w+)',  # JavaScript
            r'def\s+(\w+)',  # Python
            r'(\w+)\s*\([^)]*\)\s*\{',  # Méthodes
            r'public\s+\w+\s+(\w+)\s*\(',  # Java/C#
            r'private\s+\w+\s+(\w+)\s*\(',
            r'protected\s+\w+\s+(\w+)\s*\('
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, content, re.MULTILINE)
            for match in matches:
                functions.append(match.group(1))

        return list(set(functions))

    def _extract_imports(self, content: str) -> set:
        """Extrait les imports du code"""
        imports = set()

        patterns = [
            r'import\s+[\'"]([^\'"]+)[\'"]',  # ES6/TypeScript
            r'require\s*\([\'"]([^\'"]+)[\'"]\)',  # CommonJS
            r'from\s+[\'"]([^\'"]+)[\'"]',  # Python
            r'#include\s+[<"]([^>"]+)[>"]',  # C/C++
            r'using\s+([\w.]+);'  # C#
        ]

        for pattern in patterns:
            matches = re.findall(pattern, content)
            imports.update(matches)

        return imports

    def _identify_patterns(self, content: str) -> List[str]:
        """Identifie les patterns de conception dans le code"""
        patterns = []
        content_lower = content.lower()

        pattern_indicators = {
            'singleton': [r'getinstance\(\)', r'instance\s*=\s*null'],
            'factory': [r'factory\s+method', r'create\w+\(\)'],
            'observer': [r'addobserver', r'notify', r'\.subscribe\(\)'],
            'repository': [r'repository', r'findby', r'crudrepository'],
            'decorator': [r'@\w+', r'decorator'],
            'adapter': [r'adapter', r'adapt\('],
            'strategy': [r'strategy', r'algorithm'],
            'template_method': [r'template', r'abstract\s+class'],
            'dependency_injection': [r'@inject', r'@autowired', r'injectable']
        }

        for pattern_name, indicators in pattern_indicators.items():
            for indicator in indicators:
                if re.search(indicator, content_lower):
                    patterns.append(pattern_name)
                    break

        return patterns

    def _estimate_complexity(self, content: str) -> str:
        """Estime la complexité du code"""
        lines = len(content.split('\n'))

        if lines < 20:
            return "faible"
        elif lines < 100:
            return "moyenne"
        else:
            return "élevée"

    def _is_entry_point(self, content: str, metadata: Dict) -> bool:
        """Identifie si un chunk est un point d'entrée"""
        filename = metadata.get('filename', '').lower()

        # Fichiers typiquement points d'entrée
        entry_point_files = [
            'main', 'app', 'index', 'server', 'startup',
            'application', 'program', 'bootstrap'
        ]

        for entry in entry_point_files:
            if entry in filename:
                return True

        # Contenu typique des points d'entrée
        entry_point_patterns = [
            r'public\s+static\s+void\s+main',
            r'if\s+__name__\s*==\s*[\'"]__main__[\'"]',
            r'app\.listen\(|app\.run\(',
            r'express\(\)|flask\(\)|koa\(\)',
            r'@springbootapplication',
            r'@nestjsapplication'
        ]

        for pattern in entry_point_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return True

        return False

    def _identify_entry_point_type(self, content: str) -> str:
        """Identifie le type de point d'entrée"""
        if re.search(r'public\s+static\s+void\s+main', content):
            return "Java Application"
        elif re.search(r'if\s+__name__\s*==', content):
            return "Python Script"
        elif re.search(r'express\(\)', content):
            return "Node.js/Express API"
        elif re.search(r'@springbootapplication', content):
            return "Spring Boot Application"
        elif re.search(r'@nestjsapplication', content):
            return "NestJS Application"
        elif re.search(r'app\.listen\(|app\.run\(', content):
            return "Web Server"
        else:
            return "Unknown"

    def _identify_data_structures(self, content: str) -> List[str]:
        """Identifie les structures de données dans le code"""
        structures = []

        # Patterns pour les structures de données
        data_patterns = [
            r'class\s+(\w+)\s*{[\s\S]*?(?:List|Set|Map|Collection)<',  # Collections génériques
            r'@entity\s*\n\s*class\s+(\w+)',  # Entités JPA
            r'@document\s*\n\s*class\s+(\w+)',  # Documents MongoDB
            r'interface\s+(\w+)\s*{[\s\S]*?get\w*\(|set\w*\('  # Interfaces avec getters/setters
        ]

        for pattern in data_patterns:
            matches = re.finditer(pattern, content, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                structures.append(match.group(1))

        return structures

    def _format_detailed_structure(self, structure: Dict) -> str:
        """Formate la structure du projet de manière détaillée"""
        if not structure:
            return "Aucune structure de projet fournie."

        formatted = ["### Organisation du Projet"]

        if 'total_files' in structure:
            formatted.append(f"- **Fichiers totaux**: {structure['total_files']}")

        if 'extensions_found' in structure:
            formatted.append(f"- **Extensions**: {', '.join(structure['extensions_found'])}")

        if 'file_count_by_extension' in structure:
            formatted.append("- **Répartition par extension**:")
            for ext, count in sorted(structure['file_count_by_extension'].items(),
                                     key=lambda x: x[1], reverse=True)[:5]:
                formatted.append(f"  - {ext}: {count} fichiers")

        if 'directory_structure' in structure:
            formatted.append("- **Hiérarchie des répertoires** (principaux):")
            dirs = list(structure['directory_structure'].items())[:10]
            for path, info in dirs:
                if info.get('files_count', 0) > 0:
                    formatted.append(f"  - {path}/ ({info['files_count']} fichiers)")

        return '\n'.join(formatted)

    def _format_code_analysis(self, analysis: Dict) -> str:
        """Formate l'analyse préliminaire du code"""
        formatted = ["### Analyse Préliminaire du Codebase"]

        if analysis['classes']:
            formatted.append(f"- **Classes identifiées**: {len(analysis['classes'])}")
            formatted.append("  Principales classes:")
            for cls in analysis['classes'][:5]:
                formatted.append(f"    - {cls}")

        if analysis['functions']:
            formatted.append(f"- **Fonctions/méthodes**: {len(analysis['functions'])}")

        if analysis['imports']:
            formatted.append(f"- **Dépendances externes**: {len(analysis['imports'])}")
            formatted.append("  Principales dépendances:")
            for imp in list(analysis['imports'])[:10]:
                formatted.append(f"    - {imp}")

        if analysis['entry_points']:
            formatted.append("- **Points d'entrée identifiés**:")
            for entry in analysis['entry_points']:
                formatted.append(f"  - {entry['filename']} ({entry['type']})")

        if analysis['data_structures']:
            formatted.append(f"- **Structures de données**: {len(analysis['data_structures'])}")
            for ds in analysis['data_structures'][:5]:
                formatted.append(f"    - {ds}")

        return '\n'.join(formatted)

    def _extract_detailed_descriptions(self, response: str, chunks: List[Dict]) -> Dict[str, Any]:
        """Extrait les descriptions détaillées de la réponse LLM"""
        descriptions = {
            'classes': [],
            'functions': [],
            'modules': [],
            'patterns': [],
            'data_models': [],
            'apis': []
        }

        lines = response.split('\n')
        current_section = None

        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue

            line_lower = line_stripped.lower()

            # Détecter les sections principales (h1, h2, h3)
            if line_stripped.startswith('## '):  # Niveau 2
                section_title = line_stripped[3:].strip().lower()

                # Mapper les sections
                if 'classe' in section_title or 'class' in section_title:
                    current_section = 'classes'
                elif 'fonction' in section_title or 'function' in section_title or 'méthode' in section_title:
                    current_section = 'functions'
                elif 'module' in section_title or 'composant' in section_title:
                    current_section = 'modules'
                elif 'pattern' in section_title or 'patron' in section_title:
                    current_section = 'patterns'
                elif 'modèle' in section_title and ('donné' in section_title or 'data' in section_title):
                    current_section = 'data_models'
                elif 'api' in section_title or 'endpoint' in section_title or 'interface' in section_title:
                    current_section = 'apis'
                else:
                    current_section = None

            elif line_stripped.startswith('### '):  # Niveau 3 - éléments spécifiques
                element_title = line_stripped[4:].strip()

                # Vérifier si c'est un nom d'élément (pas un titre de section)
                if len(element_title) < 50 and not element_title.startswith('**'):  # Probablement un nom d'élément
                    if current_section == 'classes' and self._looks_like_class_name(element_title):
                        descriptions['classes'].append({
                            'name': element_title,
                            'description': '',
                            'methods': [],
                            'attributes': []
                        })

            # Extraire le contenu pour l'élément courant
            elif current_section and line_stripped and not line_stripped.startswith('#'):
                # Pour les listes d'attributs/méthodes
                if line_stripped.startswith('- **'):
                    # Format: - **nom**: description
                    line_content = line_stripped[2:].strip()  # Enlever "- "

                    if ':**' in line_content:
                        parts = line_content.split(':**', 1)
                        if len(parts) == 2:
                            name = parts[0].replace('**', '').strip()
                            value = parts[1].strip()

                            # Ajouter aux descriptions appropriées
                            if current_section == 'classes' and descriptions['classes']:
                                current_class = descriptions['classes'][-1]

                                if 'méthode' in name.lower() or 'method' in name.lower() or '()' in name:
                                    current_class['methods'].append({
                                        'name': name.split(':')[0] if ':' in name else name,
                                        'description': value
                                    })
                                elif 'attribut' in name.lower() or 'attribute' in name.lower() or 'propriété' in name.lower():
                                    current_class['attributes'].append({
                                        'name': name.split(':')[0] if ':' in name else name,
                                        'description': value
                                    })
                                else:
                                    # Description générale de la classe
                                    if not current_class['description']:
                                        current_class['description'] = f"{name}: {value}"
                                    else:
                                        current_class['description'] += f" | {name}: {value}"

                # Pour les descriptions en texte simple
                elif len(line_stripped) > 20 and not line_stripped.startswith('```'):
                    if current_section == 'classes' and descriptions['classes']:
                        current_class = descriptions['classes'][-1]
                        if not current_class['description']:
                            current_class['description'] = line_stripped
                        elif len(current_class['description']) < 500:  # Limiter la longueur
                            current_class['description'] += f" {line_stripped}"

                    elif current_section == 'functions':
                        # Extraire les descriptions de fonctions
                        if ':' in line_stripped and len(line_stripped) < 150:
                            parts = line_stripped.split(':', 1)
                            if len(parts) == 2 and self._looks_like_function_name(parts[0].strip()):
                                descriptions['functions'].append({
                                    'name': parts[0].strip(),
                                    'description': parts[1].strip()
                                })

        # Si peu de descriptions extraites, essayer de les générer depuis les chunks
        if self._are_descriptions_empty(descriptions):
            logger.info("Peu de descriptions extraites de la réponse LLM, génération depuis les chunks...")
            descriptions = self._generate_descriptions_from_chunks(chunks)

        return descriptions

    def _looks_like_class_name(self, text: str) -> bool:
        """Vérifie si un texte ressemble à un nom de classe"""
        # Un nom de classe est généralement en PascalCase, pas trop long
        if len(text) > 50:
            return False

        # Vérifier le format (PascalCase ou avec des points pour les namespaces)
        if re.match(r'^[A-Z][a-zA-Z0-9]*(\.[A-Z][a-zA-Z0-9]*)*$', text):
            return True

        # Vérifier les mots-clés communs dans les noms de classe
        class_indicators = ['Controller', 'Service', 'Repository', 'Model', 'Entity',
                            'DTO', 'VO', 'BO', 'DAO', 'Config', 'Helper', 'Util', 'Manager']

        return any(indicator in text for indicator in class_indicators)

    def _looks_like_function_name(self, text: str) -> bool:
        """Vérifie si un texte ressemble à un nom de fonction/méthode"""
        if len(text) > 40:
            return False

        # Les noms de fonction sont généralement en camelCase ou avec des underscores
        patterns = [
            r'^[a-z][a-zA-Z0-9]*$',  # camelCase
            r'^[a-z][a-z0-9_]*$',  # snake_case
            r'^[A-Z][a-zA-Z0-9]*$',  # PascalCase (pour les méthodes statiques)
            r'^[a-z][a-zA-Z0-9]*\(',  # Avec parenthèses
        ]

        return any(re.match(pattern, text) for pattern in patterns)

    def _are_descriptions_empty(self, descriptions: Dict[str, Any]) -> bool:
        """Vérifie si les descriptions sont vides"""
        total_items = 0
        for key, items in descriptions.items():
            if isinstance(items, list):
                total_items += len(items)

        return total_items < 5  # Si moins de 5 éléments, considérer comme vide

    def _generate_descriptions_from_chunks(self, chunks: List[Dict]) -> Dict[str, Any]:
        """Génère des descriptions à partir des chunks de code"""
        descriptions = {
            'classes': [],
            'functions': [],
            'modules': [],
            'patterns': [],
            'data_models': [],
            'apis': []
        }

        for chunk in chunks:
            if isinstance(chunk, dict):
                content = chunk.get('content', '')
                metadata = chunk.get('metadata', {})
                filename = metadata.get('filename', 'unknown')

                # Extraire les classes et générer des descriptions
                classes = self._extract_classes(content)
                for class_name in classes:
                    class_type = self._infer_class_type(class_name, content)
                    description = self._infer_class_description(class_name, content, metadata)

                    # Extraire les méthodes de la classe
                    methods = self._extract_class_methods_details(content, class_name)

                    descriptions['classes'].append({
                        'name': class_name,
                        'description': description,
                        'file': filename,
                        'type': class_type,
                        'methods': methods[:5],  # Limiter à 5 méthodes
                        'imports': list(self._extract_imports(content))[:5]
                    })

                # Extraire les fonctions (hors classes)
                functions = self._extract_functions(content)
                # Filtrer les fonctions qui ne sont pas dans des classes
                for func_name in functions:
                    if not self._is_function_in_class(func_name, content, classes):
                        descriptions['functions'].append({
                            'name': func_name,
                            'description': self._infer_function_description(func_name, content),
                            'file': filename,
                            'scope': 'global'
                        })

                # Extraire les APIs
                apis = self._extract_detailed_apis(content, metadata)
                descriptions['apis'].extend(apis)

        return descriptions

    def _extract_class_methods_details(self, content: str, class_name: str) -> List[Dict[str, str]]:
        """Extrait les détails des méthodes d'une classe spécifique"""
        methods = []

        # Trouver la section de la classe
        class_pattern = rf'class\s+{re.escape(class_name)}[^{{]*{{([^}}]+)}}'
        match = re.search(class_pattern, content, re.DOTALL)

        if match:
            class_body = match.group(1)

            # Extraire les méthodes
            method_patterns = [
                r'(?:public|private|protected)?\s+(?:static\s+)?(?:\w+\s+)?(\w+)\s*\([^)]*\)',
                r'def\s+(\w+)\s*\([^)]*\):',
                r'(\w+)\s*\([^)]*\)\s*\{',
            ]

            for pattern in method_patterns:
                method_matches = re.finditer(pattern, class_body)
                for method_match in method_matches:
                    method_name = method_match.group(1)
                    if method_name not in ['class', 'self', 'this', 'super']:  # Filtrer les mots-clés
                        methods.append({
                            'name': method_name,
                            'description': self._infer_method_description(method_name, class_body)
                        })

        return methods

    def _is_function_in_class(self, function_name: str, content: str, classes: List[str]) -> bool:
        """Vérifie si une fonction est dans une classe"""
        # Chercher la fonction dans le contenu
        lines = content.split('\n')
        inside_class = False
        current_class = None

        for line in lines:
            # Vérifier si on entre dans une classe
            for class_name in classes:
                if f'class {class_name}' in line or f'class {class_name}(' in line:
                    inside_class = True
                    current_class = class_name
                    break

            # Vérifier si on sort de la classe
            if inside_class and line.strip() == '}' or line.strip() == '})' or line.strip().endswith('}:'):
                inside_class = False
                current_class = None

            # Si on est dans une classe et qu'on trouve la fonction
            if inside_class and function_name in line:
                # Vérifier que c'est bien une déclaration de méthode
                method_patterns = [
                    rf'\s+{re.escape(function_name)}\s*\(',
                    rf'def\s+{re.escape(function_name)}\s*\(',
                    rf'(?:public|private|protected)?\s+(?:static\s+)?(?:\w+\s+)?{re.escape(function_name)}\s*\('
                ]

                for pattern in method_patterns:
                    if re.search(pattern, line):
                        return True

        return False

    def _infer_method_description(self, method_name: str, class_body: str) -> str:
        """Infère une description pour une méthode"""
        # Chercher la ligne de la méthode
        lines = class_body.split('\n')
        for i, line in enumerate(lines):
            if method_name in line and '(' in line:
                # Chercher des commentaires avant la méthode
                if i > 0:
                    for j in range(1, 4):  # Regarder les 3 lignes précédentes
                        if i - j >= 0:
                            comment_line = lines[i - j].strip()
                            if comment_line.startswith('//') or comment_line.startswith('#'):
                                return comment_line[2:].strip()
                            elif comment_line.startswith('/*'):
                                # Extraire le commentaire multi-ligne
                                comment = comment_line[2:].strip()
                                k = i - j + 1
                                while k < i and '*/' not in lines[k]:
                                    comment += ' ' + lines[k].strip()
                                    k += 1
                                return comment

                # Inférer basé sur le nom de la méthode
                method_lower = method_name.lower()
                if any(word in method_lower for word in ['get', 'find', 'search', 'fetch', 'retrieve']):
                    return f"Récupère {method_name.replace('get', '').replace('find', '').replace('search', '').replace('fetch', '').replace('retrieve', '')}"
                elif any(word in method_lower for word in ['create', 'add', 'insert', 'save']):
                    return f"Crée un nouvel élément {method_name.replace('create', '').replace('add', '').replace('insert', '').replace('save', '')}"
                elif any(word in method_lower for word in ['update', 'modify', 'change']):
                    return f"Met à jour {method_name.replace('update', '').replace('modify', '').replace('change', '')}"
                elif any(word in method_lower for word in ['delete', 'remove']):
                    return f"Supprime {method_name.replace('delete', '').replace('remove', '')}"
                elif any(word in method_lower for word in ['validate', 'check', 'verify']):
                    return f"Valide {method_name.replace('validate', '').replace('check', '').replace('verify', '')}"
                elif any(word in method_lower for word in ['calculate', 'compute']):
                    return f"Calcule {method_name.replace('calculate', '').replace('compute', '')}"
                else:
                    return f"Implémente la logique pour {method_name}"

        return f"Méthode {method_name}"

    def _extract_detailed_apis(self, content: str, metadata: Dict) -> List[Dict[str, Any]]:
        """Extrait les détails des APIs du code"""
        apis = []

        # Patterns pour différentes technologies
        api_patterns = [
            # Spring Boot
            (r'@(GetMapping|PostMapping|PutMapping|DeleteMapping|RequestMapping|PatchMapping)\s*\(\s*[\'"]([^\'"]+)[\'"]\s*\)',
             lambda match: {'method': match.group(1).replace('Mapping', '').upper(), 'endpoint': match.group(2)}),

            # Express.js
            (r'app\.(get|post|put|delete|patch|all)\s*\(\s*[\'"]([^\'"]+)[\'"]',
             lambda match: {'method': match.group(1).upper(), 'endpoint': match.group(2)}),

            # Flask
            (r'@app\.route\s*\(\s*[\'"]([^\'"]+)[\'"]\s*,\s*methods\s*=\s*\[[\'"](\w+)[\'"]\]',
             lambda match: {'method': match.group(2).upper(), 'endpoint': match.group(1)}),

            # FastAPI
            (r'@app\.(get|post|put|delete|patch)\s*\(\s*[\'"]([^\'"]+)[\'"]',
             lambda match: {'method': match.group(1).upper(), 'endpoint': match.group(2)}),
        ]

        for pattern, processor in api_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                try:
                    api_info = processor(match)
                    api_info.update({
                        'file': metadata.get('filename', 'unknown'),
                        'class': self._find_containing_class_at_position(content, match.start()),
                        'line': self._get_line_number(content, match.start())
                    })
                    apis.append(api_info)
                except Exception as e:
                    logger.debug(f"Erreur traitement API pattern: {e}")

        return apis

    def _find_containing_class_at_position(self, content: str, position: int) -> Optional[str]:
        """Trouve la classe contenant une position spécifique"""
        # Extraire le contenu avant la position
        content_before = content[:position]

        # Chercher la dernière classe avant cette position
        matches = list(re.finditer(r'class\s+(\w+)', content_before))
        if matches:
            return matches[-1].group(1)

        return None

    def _get_line_number(self, content: str, position: int) -> int:
        """Retourne le numéro de ligne d'une position dans le contenu"""
        return content[:position].count('\n') + 1

    def _generate_descriptions_from_chunks(self, chunks: List[Dict]) -> Dict[str, Any]:
        """Génère des descriptions à partir des chunks de code"""
        descriptions = {
            'classes': [],
            'functions': [],
            'modules': [],
            'patterns': [],
            'data_models': [],
            'apis': []
        }

        for chunk in chunks:
            if isinstance(chunk, dict):
                content = chunk.get('content', '')
                metadata = chunk.get('metadata', {})

                # Extraire les classes et générer des descriptions
                classes = self._extract_classes(content)
                for class_name in classes:
                    description = self._infer_class_description(class_name, content, metadata)
                    descriptions['classes'].append({
                        'name': class_name,
                        'description': description,
                        'file': metadata.get('filename', 'unknown'),
                        'type': self._infer_class_type(class_name, content)
                    })

                # Extraire les fonctions
                functions = self._extract_functions(content)
                for func_name in functions:
                    descriptions['functions'].append({
                        'name': func_name,
                        'description': self._infer_function_description(func_name, content),
                        'class': self._find_containing_class(func_name, content)
                    })

        return descriptions

    def _infer_class_description(self, class_name: str, content: str, metadata: Dict) -> str:
        """Infère une description pour une classe basée sur son contexte"""
        filename = metadata.get('filename', '').lower()

        # Détecter le type de classe par le nom et le contexte
        if 'controller' in class_name.lower() or 'controller' in filename:
            return f"Contrôleur gérant les requêtes HTTP pour {class_name.replace('Controller', '')}"
        elif 'service' in class_name.lower() or 'service' in filename:
            return f"Service métier implémentant la logique pour {class_name.replace('Service', '')}"
        elif 'repository' in class_name.lower() or 'dao' in class_name.lower():
            return f"Repository pour l'accès aux données de {class_name.replace('Repository', '').replace('DAO', '')}"
        elif 'model' in class_name.lower() or 'entity' in class_name.lower():
            return f"Modèle de données représentant {class_name}"
        elif 'config' in class_name.lower() or 'configuration' in filename:
            return f"Classe de configuration pour {class_name}"
        elif 'util' in class_name.lower() or 'helper' in class_name.lower():
            return f"Classe utilitaire fournissant des fonctions auxiliaires"
        else:
            # Analyser le contenu pour inférer
            if re.search(r'extends|implements', content):
                return f"Implémentation de {class_name} avec héritage/interface"
            elif re.search(r'@\w+', content):
                return f"Classe annotée {class_name} probablement liée à un framework"
            else:
                return f"Classe {class_name} avec responsabilités à déterminer"

    def _extract_detailed_summary(self, response: str) -> str:
        """Extrait un résumé détaillé de la réponse"""
        # Chercher le premier paragraphe significatif
        lines = response.split('\n')
        summary_lines = []

        for line in lines:
            line_stripped = line.strip()
            if line_stripped and not line_stripped.startswith('#'):
                if len(line_stripped) > 50:  # Ignorer les lignes trop courtes
                    summary_lines.append(line_stripped)
                    if len(summary_lines) >= 5:
                        break

        return ' '.join(summary_lines) if summary_lines else "Résumé non disponible"

    def _extract_architectural_elements(self, response: str) -> List[Dict[str, str]]:
        """Extrait les éléments architecturaux identifiés"""
        elements = []
        lines = response.split('\n')

        current_element = None
        for line in lines:
            line_stripped = line.strip()

            # Détecter un nouvel élément architectural
            if line_stripped.startswith('**') and ':' in line_stripped:
                if current_element:
                    elements.append(current_element)

                parts = line_stripped.strip('*').split(':', 1)
                if len(parts) == 2:
                    current_element = {
                        'name': parts[0].strip(),
                        'description': parts[1].strip(),
                        'responsibilities': [],
                        'dependencies': []
                    }

            # Ajouter des détails à l'élément courant
            elif current_element and line_stripped.startswith('-'):
                detail = line_stripped[2:].strip()
                if 'responsabilité' in detail.lower() or 'responsibility' in detail.lower():
                    current_element['responsibilities'].append(detail)
                elif 'dépend' in detail.lower() or 'depend' in detail.lower():
                    current_element['dependencies'].append(detail)

        if current_element:
            elements.append(current_element)

        return elements

    def _extract_data_models(self, response: str) -> List[Dict[str, Any]]:
        """Extrait les modèles de données identifiés"""
        data_models = []
        lines = response.split('\n')

        in_data_models_section = False
        current_model = None

        for line in lines:
            line_stripped = line.strip()
            line_lower = line_stripped.lower()

            # Détecter la section des modèles de données
            if 'modèle de données' in line_lower or 'data model' in line_lower:
                in_data_models_section = True
                continue

            if in_data_models_section and line_stripped.startswith('###'):
                if current_model:
                    data_models.append(current_model)
                current_model = {
                    'name': line_stripped.strip('#').strip(),
                    'fields': [],
                    'relationships': []
                }

            elif current_model and line_stripped.startswith('-'):
                if ':' in line_stripped:
                    field_parts = line_stripped[2:].split(':', 1)
                    if len(field_parts) == 2:
                        current_model['fields'].append({
                            'name': field_parts[0].strip(),
                            'type': field_parts[1].strip()
                        })

        if current_model:
            data_models.append(current_model)

        return data_models

    def _extract_api_specs(self, chunks: List[Dict]) -> List[Dict[str, Any]]:
        """Extrait les spécifications d'API des chunks"""
        api_specs = []

        for chunk in chunks:
            if isinstance(chunk, dict):
                content = chunk.get('content', '')
                metadata = chunk.get('metadata', {})

                # Chercher des endpoints API
                api_patterns = [
                    (r'@GetMapping\s*\(\s*[\'"]([^\'"]+)[\'"]\s*\)', 'GET'),
                    (r'@PostMapping\s*\(\s*[\'"]([^\'"]+)[\'"]\s*\)', 'POST'),
                    (r'@PutMapping\s*\(\s*[\'"]([^\'"]+)[\'"]\s*\)', 'PUT'),
                    (r'@DeleteMapping\s*\(\s*[\'"]([^\'"]+)[\'"]\s*\)', 'DELETE'),
                    (r'@RequestMapping\s*\(\s*[\'"]([^\'"]+)[\'"]\s*\)', 'REQUEST'),
                    (r'app\.get\s*\(\s*[\'"]([^\'"]+)[\'"]', 'GET'),
                    (r'app\.post\s*\(\s*[\'"]([^\'"]+)[\'"]', 'POST'),
                    (r'app\.put\s*\(\s*[\'"]([^\'"]+)[\'"]', 'PUT'),
                    (r'app\.delete\s*\(\s*[\'"]([^\'"]+)[\'"]', 'DELETE'),
                ]

                for pattern, method in api_patterns:
                    matches = re.finditer(pattern, content)
                    for match in matches:
                        api_specs.append({
                            'endpoint': match.group(1),
                            'method': method,
                            'file': metadata.get('filename', 'unknown'),
                            'class': self._find_containing_class_at_position(content, match.start())
                        })

        return api_specs

    def _extract_business_logic(self, response: str, chunks: List[Dict]) -> List[Dict[str, Any]]:
        """Extrait la logique métier identifiée"""
        business_logic = []
        lines = response.split('\n')

        current_rule = None
        for line in lines:
            line_stripped = line.strip()
            line_lower = line_stripped.lower()

            # Détecter les règles métier
            if any(keyword in line_lower for keyword in
                   ['règle', 'rule', 'validation', 'calcul', 'business logic', 'logique métier']):
                if current_rule:
                    business_logic.append(current_rule)

                current_rule = {
                    'description': line_stripped,
                    'type': self._classify_business_rule(line_stripped),
                    'implementation_hints': []
                }

            elif current_rule and line_stripped.startswith('-'):
                current_rule['implementation_hints'].append(line_stripped[2:])

        if current_rule:
            business_logic.append(current_rule)

        return business_logic

    def _classify_business_rule(self, rule_text: str) -> str:
        """Classe une règle métier par type"""
        rule_lower = rule_text.lower()

        if any(word in rule_lower for word in ['validation', 'validate', 'must', 'required']):
            return "validation"
        elif any(word in rule_lower for word in ['calcul', 'compute', 'calculate', 'formula']):
            return "calculation"
        elif any(word in rule_lower for word in ['workflow', 'process', 'flow']):
            return "workflow"
        elif any(word in rule_lower for word in ['permission', 'authorization', 'access']):
            return "security"
        else:
            return "general"

    def _find_containing_class(self, function_name: str, content: str) -> Optional[str]:
        """Trouve la classe contenant une fonction"""
        # Chercher la classe la plus proche avant la fonction
        lines = content.split('\n')
        last_class = None

        for line in lines:
            if function_name in line:
                return last_class
            match = re.search(r'class\s+(\w+)', line)
            if match:
                last_class = match.group(1)

        return None

    def _find_containing_class_at_position(self, content: str, position: int) -> Optional[str]:
        """Trouve la classe contenant une position spécifique"""
        # Extraire le contenu avant la position
        content_before = content[:position]

        # Chercher la dernière classe avant cette position
        matches = list(re.finditer(r'class\s+(\w+)', content_before))
        if matches:
            return matches[-1].group(1)

        return None

    def _infer_class_type(self, class_name: str, content: str) -> str:
        """Infère le type d'une classe basé sur son nom et son contenu"""
        class_lower = class_name.lower()
        content_lower = content.lower()

        if 'controller' in class_lower or '@controller' in content_lower:
            return "Controller"
        elif 'service' in class_lower or '@service' in content_lower:
            return "Service"
        elif 'repository' in class_lower or 'dao' in class_lower or '@repository' in content_lower:
            return "Repository"
        elif 'model' in class_lower or 'entity' in class_lower or '@entity' in content_lower:
            return "Model/Entity"
        elif 'config' in class_lower or '@configuration' in content_lower:
            return "Configuration"
        elif 'dto' in class_lower:
            return "DTO"
        elif 'exception' in class_lower:
            return "Exception"
        elif 'util' in class_lower or 'helper' in class_lower:
            return "Utility"
        else:
            return "General"

    def _infer_function_description(self, function_name: str, content: str) -> str:
        """Infère une description pour une fonction"""
        # Chercher la ligne de la fonction
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if function_name in line:
                # Analyser la ligne suivante pour les commentaires
                if i > 0 and '//' in lines[i - 1]:
                    return lines[i - 1].split('//', 1)[1].strip()
                elif i > 0 and '/*' in lines[i - 1]:
                    return lines[i - 1].split('/*', 1)[1].strip()

                # Inférer basé sur le nom
                func_lower = function_name.lower()
                if any(word in func_lower for word in ['get', 'find', 'search', 'fetch']):
                    return f"Récupère {function_name.replace('get', '').replace('find', '')}"
                elif any(word in func_lower for word in ['create', 'add', 'insert']):
                    return f"Crée un nouvel élément {function_name.replace('create', '')}"
                elif any(word in func_lower for word in ['update', 'modify']):
                    return f"Met à jour {function_name.replace('update', '')}"
                elif any(word in func_lower for word in ['delete', 'remove']):
                    return f"Supprime {function_name.replace('delete', '')}"
                elif any(word in func_lower for word in ['validate', 'check']):
                    return f"Valide {function_name.replace('validate', '')}"
                elif any(word in func_lower for word in ['calculate', 'compute']):
                    return f"Calcule {function_name.replace('calculate', '')}"
                else:
                    return f"Implémente la logique pour {function_name}"

        return f"Fonction {function_name}"
