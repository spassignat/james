# src/documentation/template_generator.py
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from models.project_structure import ProjectStructure


class TemplateDocumentationGenerator:
    """Générateur de documentation basé sur des templates Markdown avec ``` pour les blocs de code"""

    def __init__(self, template_dir: str = "templates", output_dir: str = "docs"):
        self.template_dir = Path(template_dir)
        self.output_dir = Path(output_dir)
        self.templates: Dict[str, str] = {}
        self.variables: Dict[str, Any] = {}

        # Créer les répertoires si nécessaire
        self.template_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)

    def load_templates(self) -> Dict[str, str]:
        """Charge tous les templates depuis le répertoire"""
        templates = {}

        for template_file in self.template_dir.glob("*.md"):
            template_name = template_file.stem.upper()
            with open(template_file, 'r', encoding='utf-8') as f:
                templates[template_name] = f.read()

        self.templates = templates
        return templates

    def load_template(self, template_name: str) -> str:
        """Charge un template spécifique"""
        template_path = self.template_dir / f"{template_name.lower()}.md"

        if not template_path.exists():
            raise FileNotFoundError(f"Template {template_name}.md non trouvé dans {self.template_dir}")

        with open(template_path, 'r', encoding='utf-8') as f:
            return f.read()

    def set_project_structure(self, project: ProjectStructure):
        """Configure les variables à partir d'une ProjectStructure"""
        self.variables.update(self._extract_variables_from_project(project))

    def set_custom_variables(self, variables: Dict[str, Any]):
        """Définit des variables personnalisées"""
        self.variables.update(variables)

    def _extract_variables_from_project(self, project: ProjectStructure) -> Dict[str, Any]:
        """Extrait les variables d'une ProjectStructure"""
        now = datetime.now()

        return {
            # Variables générales
            "PROJECT_NAME": project.project_name,
            "PROJECT_DESCRIPTION": project.architecture_overview or f"Projet {project.project_name}",
            "VERSION": "1.0.0",
            "STATUS": "Développement",
            "LAST_UPDATE": now.strftime("%Y-%m-%d"),
            "LICENSE": "MIT",
            "LICENSE_DETAILS": "Voir le fichier LICENSE pour plus de détails.",
            "MAINTAINER": "Équipe de développement",
            "EMAIL": "dev@example.com",
            "ISSUES_URL": f"https://github.com/username/{project.project_name}/issues",
            "GENERATION_DATE": now.strftime("%d/%m/%Y %H:%M"),
            "LAST_ANALYSIS_DATE": project.last_analysis_at,

            # Structure du projet
            "PROJECT_TREE": self._generate_project_tree(project),
            "ARCHITECTURE_OVERVIEW": project.architecture_overview,
            "MAIN_COMPONENTS": self._format_components(project.components),
            "PATTERNS_USED": self._format_patterns(project.patterns_identified),

            # Installation
            "PREREQUISITES": self._generate_prerequisites(project),
            "INSTALL_COMMANDS": self._generate_install_commands(project),
            "CONFIG_COMMANDS": self._generate_config_commands(),
            "START_COMMANDS": self._generate_start_commands(project),
            "DEV_INSTALL_COMMANDS": self._generate_dev_install_commands(project),
            "TEST_COMMANDS": self._generate_test_commands(project),
            "FORMAT_COMMANDS": self._generate_format_commands(project),

            # API
            "API_BASE_URL": "http://localhost:8000",
            "AUTHENTICATION_DETAILS": self._generate_auth_details(),
            "RESPONSE_FORMAT": self._generate_response_format(),
            "ERROR_CODES": self._generate_error_codes(),

            # Architecture
            "ARCHITECTURE_DIAGRAM": self._generate_mermaid_diagram(project),
            "COMPONENTS_DETAILS": self._format_components_details(project),
            "MAIN_DATA_FLOW": self._generate_main_data_flow(project),

            # Variables pour déploiement
            "DEPLOYMENT_PREREQUISITES": self._generate_deployment_prerequisites(),
            "DEPLOYMENT_CONFIG": self._generate_deployment_config(),
            "BUILD_COMMANDS": self._generate_build_commands(project),
            "DOCKER_COMMANDS": self._generate_docker_commands(project),

            # Variables supplémentaires
            "MODULES_LIST": self._format_modules(project.modules),
            "DEPENDENCIES_LIST": self._format_dependencies(project.dependencies),
            "FILES_COUNT": len(project.files),
            "MODULES_COUNT": len(project.modules),
            "PATTERNS_COUNT": len(project.patterns_identified),
            "DEPENDENCIES_COUNT": len(project.dependencies),
        }

    def render_template(self, template_content: str) -> str:
        """Rend un template avec les variables actuelles"""
        # Remplacer toutes les variables {{VARIABLE}}
        pattern = r'\{\{([^}]+)\}\}'

        def replace_match(match):
            var_name = match.group(1).strip()
            # Chercher la variable dans différentes formes
            value = self.variables.get(var_name)

            if value is None:
                # Essayer avec différents formats
                alt_names = [
                    var_name.lower(),
                    var_name.upper(),
                    var_name.capitalize(),
                ]
                for alt_name in alt_names:
                    if alt_name in self.variables:
                        value = self.variables[alt_name]
                        break

            return str(value) if value is not None else f"{{{{{var_name}}}}}"

        rendered = re.sub(pattern, replace_match, template_content)

        # Convertir les ``` en ``` pour les blocs de code finaux
        rendered = self._convert_to_code_blocks(rendered)

        return rendered

    def _convert_to_code_blocks(self, content: str) -> str:
        """Convertit les ``` en ``` pour les blocs de code finaux"""
        # Remplace ```lang\ncontent\n``` par ```lang\ncontent\n```
        lines = content.split('\n')
        result = []
        in_code_block = False
        language = ""

        for i, line in enumerate(lines):
            stripped_line = line.strip()

            if stripped_line.startswith('```'):
                if not in_code_block:
                    # Début d'un bloc de code
                    in_code_block = True
                    language = stripped_line[3:].strip()
                    if language:
                        result.append(f"```{language}")
                    else:
                        result.append("```")
                else:
                    # Fin d'un bloc de code
                    in_code_block = False
                    result.append("```")
            else:
                result.append(line)

        return '\n'.join(result)

    def generate_all(self) -> Dict[str, str]:
        """Génère tous les documents à partir des templates"""
        if not self.templates:
            self.load_templates()

        generated_docs = {}

        for template_name, template_content in self.templates.items():
            output_filename = f"{template_name.lower()}.md"
            rendered_content = self.render_template(template_content)

            # Sauvegarder le fichier
            output_path = self.output_dir / output_filename
            output_path.write_text(rendered_content, encoding='utf-8')

            generated_docs[template_name] = {
                "filename": output_filename,
                "path": str(output_path),
                "content": rendered_content
            }

        # Générer un fichier de variables pour référence
        self._generate_variables_reference()

        return generated_docs

    def generate_specific(self, template_names: List[str]) -> Dict[str, str]:
        """Génère des documents spécifiques"""
        generated_docs = {}

        for template_name in template_names:
            template_content = self.load_template(template_name)
            output_filename = f"{template_name.lower()}.md"
            rendered_content = self.render_template(template_content)

            # Sauvegarder le fichier
            output_path = self.output_dir / output_filename
            output_path.write_text(rendered_content, encoding='utf-8')

            generated_docs[template_name] = {
                "filename": output_filename,
                "path": str(output_path),
                "content": rendered_content
            }

        return generated_docs

    def _generate_variables_reference(self):
        """Génère un fichier de référence des variables"""
        ref_content = "# Variables disponibles\n\n"

        # Trier les variables par nom
        sorted_vars = sorted(self.variables.items())

        for var_name, var_value in sorted_vars:
            # Tronquer les valeurs longues
            if isinstance(var_value, str) and len(var_value) > 100:
                display_value = var_value[:100] + "..."
            else:
                display_value = str(var_value)

            ref_content += f"- `{var_name}`: {display_value}\n"

        ref_path = self.output_dir / "VARIABLES.md"
        ref_path.write_text(ref_content, encoding='utf-8')

    # Méthodes de génération de contenu avec ``` pour les templates
    def _generate_project_tree(self, project: ProjectStructure, max_depth: int = 3) -> str:
        """Génère un arbre du projet"""
        dirs = {}
        for file_path in project.files:
            parts = file_path.split('/')
            for i in range(len(parts)):
                dir_path = '/'.join(parts[:i])
                if dir_path not in dirs:
                    dirs[dir_path] = []
                if i == len(parts) - 1:
                    dirs[dir_path].append(parts[-1])

        def build_tree(current_dir: str = "", depth: int = 0) -> List[str]:
            if depth > max_depth:
                return []

            lines = []
            indent = "    " * depth

            items = dirs.get(current_dir, [])
            subdirs = []
            files = []

            for item in items:
                full_path = f"{current_dir}/{item}" if current_dir else item
                if full_path in dirs and full_path != current_dir:
                    subdirs.append(item)
                else:
                    files.append(item)

            subdirs.sort()
            files.sort()

            if current_dir:
                dir_name = current_dir.split('/')[-1]
                lines.append(f"{indent}{dir_name}/")
                indent += "    "

            for i, subdir in enumerate(subdirs):
                is_last_dir = i == len(subdirs) - 1 and not files
                prefix = "└── " if is_last_dir else "├── "
                full_path = f"{current_dir}/{subdir}" if current_dir else subdir
                lines.append(f"{indent}{prefix}{subdir}/")
                lines.extend(build_tree(full_path, depth + 1))

            for i, file in enumerate(files):
                is_last = i == len(files) - 1
                prefix = "└── " if is_last else "├── "
                lines.append(f"{indent}{prefix}{file}")

            return lines

        tree_lines = build_tree()
        tree_content = '\n'.join(tree_lines) if tree_lines else "."

        # Retourner avec ``` pour template
        return tree_content

    def _format_components(self, components: Dict[str, Any]) -> str:
        """Formate les composants"""
        if not components:
            return "Aucun composant spécifique identifié."

        md = ""
        for comp_type, comp_names in components.items():
            md += f"- **{comp_type}**: {', '.join(comp_names[:3])}"
            if len(comp_names) > 3:
                md += f"... (+{len(comp_names) - 3} autres)"
            md += "\n"

        return md

    def _format_patterns(self, patterns: List[str]) -> str:
        """Formate les patterns"""
        if not patterns:
            return "Aucun pattern spécifique identifié."

        pattern_descriptions = {
            'MVC': 'Modèle-Vue-Contrôleur',
            'MVVM': 'Modèle-Vue-VueModèle',
            'Repository': 'Pattern repository',
            'Factory': 'Pattern factory',
            'Singleton': 'Pattern singleton',
            'Observer': 'Pattern observer',
            'Strategy': 'Pattern strategy',
            'Decorator': 'Pattern decorator',
        }

        md = ""
        for pattern in patterns[:5]:
            desc = pattern_descriptions.get(pattern, pattern)
            md += f"- **{pattern}**: {desc}\n"

        if len(patterns) > 5:
            md += f"\n*... et {len(patterns) - 5} autres patterns*"

        return md

    def _generate_prerequisites(self, project: ProjectStructure) -> str:
        """Génère les prérequis"""
        prereqs = [
            f"Python 3.8+ (fichiers .py: {len([f for f in project.files if f.endswith('.py')])})",
            "Git",
            f"Dépendances: {len(project.dependencies)} packages"
        ]

        if any(f.endswith('.js') for f in project.files):
            prereqs.append("Node.js 14+")
        if any(f.endswith('.java') for f in project.files):
            prereqs.append("Java 11+")
        if any(f.endswith('.go') for f in project.files):
            prereqs.append("Go 1.19+")

        return '\n'.join(f"- {p}" for p in prereqs)

    def _generate_install_commands(self, project: ProjectStructure) -> str:
        """Génère les commandes d'installation"""
        py_files = len([f for f in project.files if f.endswith('.py')])
        js_files = len([f for f in project.files if f.endswith('.js')])
        java_files = len([f for f in project.files if f.endswith('.java')])

        if py_files > 0:
            return "pip install -r requirements.txt"
        elif js_files > 0:
            return "npm install"
        elif java_files > 0:
            return "mvn install"
        else:
            return "# Installation à configurer"

    def _generate_config_commands(self) -> str:
        """Génère les commandes de configuration"""
        return "cp .env.example .env\n# Éditer .env avec vos configurations"

    def _generate_start_commands(self, project: ProjectStructure) -> str:
        """Génère les commandes de démarrage"""
        if any('manage.py' in f for f in project.files):
            return "python manage.py runserver"
        elif any('app.py' in f for f in project.files):
            return "python app.py"
        elif any('main.py' in f for f in project.files):
            return "python main.py"
        elif any('index.js' in f for f in project.files):
            return "node index.js"
        else:
            return "# Commande de démarrage à configurer"

    def _generate_dev_install_commands(self, project: ProjectStructure) -> str:
        """Génère les commandes d'installation pour développement"""
        base_cmd = self._generate_install_commands(project)
        if "pip" in base_cmd:
            return base_cmd + "\npip install -r requirements-dev.txt"
        elif "npm" in base_cmd:
            return base_cmd + "\nnpm install --dev"
        return base_cmd

    def _generate_test_commands(self, project: ProjectStructure) -> str:
        """Génère les commandes de test"""
        py_files = len([f for f in project.files if f.endswith('.py')])
        js_files = len([f for f in project.files if f.endswith('.js')])

        if py_files > 0:
            return "pytest"
        elif js_files > 0:
            return "npm test"
        else:
            return "# Tests à configurer"

    def _generate_format_commands(self, project: ProjectStructure) -> str:
        """Génère les commandes de formatage"""
        py_files = len([f for f in project.files if f.endswith('.py')])
        js_files = len([f for f in project.files if f.endswith('.js')])

        if py_files > 0:
            return "black .\nisort ."
        elif js_files > 0:
            return "npx prettier --write ."
        else:
            return "# Formatage à configurer"

    def _generate_mermaid_diagram(self, project: ProjectStructure) -> str:
        """Génère un diagramme Mermaid"""
        components_count = sum(len(comp) for comp in project.components.values())

        diagram = f"""graph TD
    Client["Client\n(utilisateur)"] --> API["API Gateway\n{len(project.modules)} modules"]
    
    subgraph "Services"
        direction LR
        S1["Service 1\n{components_count//3} composants"]
        S2["Service 2\n{components_count//3} composants"]
        S3["Service 3\n{components_count//3} composants"]
    end
    
    API --> S1
    API --> S2
    API --> S3
    
    S1 --> DB["Base de données\n{len([f for f in project.files if 'db' in f.lower() or 'sql' in f.lower()])} fichiers"]
    S2 --> DB
    S3 --> DB
    
    style Client fill:#f9f,stroke:#333
    style API fill:#ccf,stroke:#333
    style S1 fill:#cfc,stroke:#333
    style S2 fill:#fcc,stroke:#333
    style S3 fill:#ffc,stroke:#333
    style DB fill:#ddf,stroke:#333
"""
        return diagram

    def _format_components_details(self, project: ProjectStructure) -> str:
        """Formate les détails des composants"""
        if not project.components:
            return "Aucun composant spécifique identifié."

        md = ""
        for comp_type, comp_names in project.components.items():
            md += f"### {comp_type.capitalize()}\n\n"
            for name in comp_names[:5]:
                md += f"- **{name}**\n"
            if len(comp_names) > 5:
                md += f"\n*... et {len(comp_names) - 5} autres*\n"
            md += "\n"

        return md

    def _generate_main_data_flow(self, project: ProjectStructure) -> str:
        """Génère la description du flux principal"""
        return f"""1. Requête entrante via l'API
2. Validation et authentification
3. Traitement par les services appropriés ({len(project.components.get('services', []))} services)
4. Accès aux données via les repositories ({len(project.components.get('repositories', []))} repositories)
5. Réponse formatée au client"""

    def _generate_auth_details(self) -> str:
        """Génère les détails d'authentification"""
        return "Authentification par token JWT. Incluez le token dans les headers."

    def _generate_response_format(self) -> str:
        """Génère le format des réponses"""
        return "Toutes les réponses sont au format JSON avec les champs: success, data, message."

    def _generate_error_codes(self) -> str:
        """Génère les codes d'erreur"""
        return """| Code | Description |
|------|-------------|
| 400 | Requête invalide |
| 401 | Non authentifié |
| 403 | Non autorisé |
| 404 | Ressource non trouvée |
| 500 | Erreur serveur |"""

    def _generate_deployment_prerequisites(self) -> str:
        """Génère les prérequis de déploiement"""
        return "- Serveur avec Linux\n- Docker et Docker Compose\n- Accès SSH au serveur\n- Certificat SSL (pour production)"

    def _generate_deployment_config(self) -> str:
        """Génère la configuration de déploiement"""
        return "Configurez les variables d'environnement:\n- DATABASE_URL\n- SECRET_KEY\n- DEBUG (False en production)"

    def _generate_build_commands(self, project: ProjectStructure) -> str:
        """Génère les commandes de build"""
        if any('Dockerfile' in f for f in project.files):
            return "docker build -t monapp ."
        elif any('package.json' in f for f in project.files):
            return "npm run build"
        else:
            return "# Build à configurer"

    def _generate_docker_commands(self, project: ProjectStructure) -> str:
        """Génère les commandes Docker"""
        if any('docker-compose.yml' in f for f in project.files):
            return "docker-compose up -d"
        elif any('Dockerfile' in f for f in project.files):
            return "docker build -t monapp .\ndocker run -p 8000:8000 monapp"
        else:
            return "# Configuration Docker à ajouter"

    def _format_modules(self, modules: List[str]) -> str:
        """Formate la liste des modules"""
        if not modules:
            return "Aucun module spécifique."

        md = ""
        for module in modules[:10]:
            md += f"- `{module}`\n"

        if len(modules) > 10:
            md += f"\n*... et {len(modules) - 10} autres*"

        return md

    def _format_dependencies(self, dependencies: List[str]) -> str:
        """Formate la liste des dépendances"""
        if not dependencies:
            return "Aucune dépendance externe."

        md = ""
        for dep in dependencies[:15]:
            md += f"- `{dep}`\n"

        if len(dependencies) > 15:
            md += f"\n*... et {len(dependencies) - 15} autres*"

        return md