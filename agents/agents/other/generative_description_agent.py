# src/agents/generative_description_agent.py
import logging
import re
from typing import Dict, Any, List

from agents.base_agent import BaseAgent
from vector.vector_store import VectorStore

# Import optionnel de Jinja2 pour le rendu de templates
try:
    from jinja2 import Environment, FileSystemLoader
except Exception:  # si la dépendance n'est pas installée
    Environment = None
    FileSystemLoader = None

logger = logging.getLogger(__name__)


class GenerativeDescriptionAgent(BaseAgent):
    """Agent spécialisé dans l'extraction de descriptions pour la génération de code"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, 'generative_description_agent')
        # Initialise l'environnement Jinja2 si disponible
        self.jinja_env = None
        try:
            if Environment is not None and FileSystemLoader is not None:
                self.jinja_env = Environment(
                    loader=FileSystemLoader('templates/jinja'),
                    autoescape=False,
                    trim_blocks=True,
                    lstrip_blocks=True,
                )
        except Exception as e:
            logger.warning(f"Échec d'initialisation Jinja2: {e}")

    def analyze(self, context: Dict[str, Any], vector_store: VectorStore) -> Dict[str, Any]:
        """Extrait les descriptions qui auraient permis de générer le code"""
        logger.info(f"📝 Début extraction descriptions génératives")

        chunks = context.get('chunks', [])
        architecture_analysis = context.get('architecture_analysis', {})

        # Analyser les chunks pour extraire les éléments
        code_elements = self._extract_code_elements(chunks)

        # Générer les descriptions
        descriptions = self._generate_generative_descriptions(code_elements, architecture_analysis)

        prompt = self._build_generative_prompt(code_elements, architecture_analysis)

        system_prompt = """Vous êtes un expert en spécifications logicielles.
Votre tâche est de créer des descriptions précises qui auraient permis
à un développeur de générer le code que vous voyez.
Fournissez des spécifications techniques détaillées, pas des analyses."""

        response = self._call_llm(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.1
        )

        return {
            'type': 'generative_descriptions',
            'agent': self.agent_name,
            'timestamp': self._get_timestamp(),
            'content': response,
            'specifications': self._extract_specifications(response),
            'code_elements': code_elements,
            'descriptions': descriptions,
            'templates': self._generate_code_templates(code_elements)
        }

    def _extract_code_elements(self, chunks: List[Dict]) -> Dict[str, List]:
        """Extrait les éléments de code des chunks"""
        elements = {
            'classes': [],
            'functions': [],
            'interfaces': [],
            'data_structures': [],
            'apis': [],
            'configurations': [],
            'tests': []
        }

        for chunk in chunks:
            if isinstance(chunk, dict):
                content = chunk.get('content', '')
                metadata = chunk.get('metadata', {})

                # Classes
                classes = self._extract_detailed_classes(content, metadata)
                elements['classes'].extend(classes)

                # Fonctions
                functions = self._extract_detailed_functions(content, metadata)
                elements['functions'].extend(functions)

                # APIs
                apis = self._extract_detailed_apis(content, metadata)
                elements['apis'].extend(apis)

                # Tests
                if self._is_test_file(metadata):
                    elements['tests'].append({
                        'file': metadata.get('filename'),
                        'type': self._identify_test_type(content)
                    })

        return elements

    def _extract_detailed_classes(self, content: str, metadata: Dict) -> List[Dict]:
        """Extrait des informations détaillées sur les classes"""
        classes = []

        # Chercher les classes avec leurs attributs et méthodes
        class_pattern = r'class\s+(\w+)(?:\s+extends\s+(\w+))?(?:\s+implements\s+([\w\s,]+))?'
        matches = re.finditer(class_pattern, content, re.MULTILINE)

        for match in matches:
            class_name = match.group(1)
            extends = match.group(2)
            implements = match.group(3)

            # Extraire les attributs
            attributes = self._extract_class_attributes(content, class_name)

            # Extraire les méthodes
            methods = self._extract_class_methods(content, class_name)

            classes.append({
                'name': class_name,
                'extends': extends,
                'implements': implements.split(',') if implements else [],
                'attributes': attributes,
                'methods': methods,
                'file': metadata.get('filename'),
                'package': self._extract_package(content)
            })

        return classes

    def _generate_generative_descriptions(self, elements: Dict, architecture: Dict) -> Dict[str, Any]:
        """Génère des descriptions qui permettraient de recréer le code"""
        descriptions = {
            'project_spec': self._generate_project_specification(elements, architecture),
            'class_specs': self._generate_class_specifications(elements['classes']),
            'api_specs': self._generate_api_specifications(elements['apis']),
            'data_model_specs': self._generate_data_model_specifications(elements['classes']),
            'architecture_decisions': self._extract_architecture_decisions(architecture)
        }

        return descriptions

    def _extract_detailed_functions(self, content: str, metadata: Dict) -> List[Dict]:
        """Extrait des informations détaillées sur les fonctions"""
        functions = []

        # Patterns pour différents langages
        patterns = [
            r'def\s+(\w+)\s*\((.*?)\)(?:\s*->\s*(\w+))?',  # Python
            r'function\s+(\w+)\s*\((.*?)\)',  # JavaScript
            r'(\w+)\s+(\w+)\s*\((.*?)\)(?:\s*:\s*(\w+))?',  # TypeScript, Java, C#
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, content, re.MULTILINE)
            for match in matches:
                if pattern.startswith('def'):
                    name = match.group(1)
                    params = match.group(2)
                    return_type = match.group(3)
                elif pattern.startswith('function'):
                    name = match.group(1)
                    params = match.group(2)
                    return_type = None
                else:
                    return_type = match.group(1)
                    name = match.group(2)
                    params = match.group(3)

                functions.append({
                    'name': name,
                    'parameters': self._parse_parameters(params),
                    'return_type': return_type,
                    'file': metadata.get('filename'),
                    'scope': self._determine_function_scope(content, name, match.start())
                })

        return functions

    def _extract_detailed_apis(self, content: str, metadata: Dict) -> List[Dict]:
        """Extrait des informations détaillées sur les APIs"""
        apis = []

        # Détection de routes HTTP
        route_patterns = [
            r'@(GET|POST|PUT|DELETE|PATCH|RequestMapping)\s*["\']([^"\']+)["\']',
            r'\.(get|post|put|delete|patch)\s*\(\s*["\']([^"\']+)["\']',
            r'router\.(get|post|put|delete)\s*\(\s*["\']([^"\']+)["\']'
        ]

        for pattern in route_patterns:
            matches = re.finditer(pattern, content, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                method = match.group(1).upper()
                path = match.group(2)

                # Extraire les paramètres de la fonction associée
                handler_info = self._extract_api_handler(content, match.end())

                apis.append({
                    'method': method,
                    'path': path,
                    'handler': handler_info.get('name'),
                    'parameters': handler_info.get('parameters', []),
                    'file': metadata.get('filename')
                })

        return apis

    def _extract_class_attributes(self, content: str, class_name: str) -> List[Dict]:
        """Extrait les attributs d'une classe"""
        attributes = []

        # Trouver le bloc de la classe
        class_block = self._extract_class_block(content, class_name)
        if not class_block:
            return attributes

        # Chercher les attributs avec différents patterns
        attribute_patterns = [
            r'(?:private|protected|public|)\s+(?:static\s+)?(\w+)\s+(\w+)(?:\s*=\s*[^;]+)?;',  # Java, C#
            r'self\.(\w+)\s*=\s*',  # Python __init__
            r'this\.(\w+)\s*=\s*',  # JavaScript
            r'(\w+)\s*:\s*(\w+)',  # TypeScript
        ]

        for pattern in attribute_patterns:
            matches = re.finditer(pattern, class_block, re.MULTILINE)
            for match in matches:
                if len(match.groups()) >= 2:
                    type_name = match.group(1)
                    attr_name = match.group(2)
                else:
                    type_name = None
                    attr_name = match.group(1)

                attributes.append({
                    'name': attr_name,
                    'type': type_name,
                    'visibility': self._determine_visibility(class_block, attr_name),
                    'default_value': self._extract_default_value(class_block, attr_name)
                })

        return attributes

    def _extract_class_methods(self, content: str, class_name: str) -> List[Dict]:
        """Extrait les méthodes d'une classe"""
        methods = []

        class_block = self._extract_class_block(content, class_name)
        if not class_block:
            return methods

        # Patterns pour les méthodes
        method_patterns = [
            r'(?:public|private|protected)\s+(?:static\s+)?(\w+)\s+(\w+)\s*\((.*?)\)',
            r'def\s+(\w+)\s*\((.*?)\)(?:\s*->\s*(\w+))?',  # Python
            r'(\w+)\s*\((.*?)\)\s*{',  # JavaScript
        ]

        for pattern in method_patterns:
            matches = re.finditer(pattern, class_block, re.MULTILINE)
            for match in matches:
                if pattern.startswith('def'):
                    name = match.group(1)
                    params = match.group(2)
                    return_type = match.group(3)
                elif '(' in pattern:
                    return_type = match.group(1) if len(match.groups()) > 2 else None
                    name = match.group(2) if len(match.groups()) > 2 else match.group(1)
                    params = match.group(3) if len(match.groups()) > 2 else match.group(2)
                else:
                    name = match.group(1)
                    params = match.group(2)
                    return_type = None

                methods.append({
                    'name': name,
                    'parameters': self._parse_parameters(params),
                    'return_type': return_type,
                    'visibility': self._determine_method_visibility(class_block, name),
                    'is_static': 'static' in match.group(0)
                })

        return methods

    def _is_test_file(self, metadata: Dict) -> bool:
        """Détermine si un fichier est un fichier de test"""
        filename = metadata.get('filename', '').lower()
        return any(test_indicator in filename for test_indicator in ['test', 'spec', 'specs'])

    def _identify_test_type(self, content: str) -> str:
        """Identifie le type de test"""
        if '@Test' in content or 'junit' in content.lower():
            return 'unit'
        elif 'cypress' in content.lower() or 'selenium' in content.lower():
            return 'e2e'
        elif 'mock' in content.lower() or 'spy' in content.lower():
            return 'integration'
        else:
            return 'unknown'

    def _extract_package(self, content: str) -> str:
        """Extrait le package/module du fichier"""
        package_patterns = [
            r'package\s+([\w.]+);',
            r'from\s+([\w.]+)\s+import',
            r'import\s+([\w.]+)',
            r'namespace\s+([\w.]+)'
        ]

        for pattern in package_patterns:
            match = re.search(pattern, content)
            if match:
                return match.group(1)
        return ''

    def _parse_parameters(self, param_string: str) -> List[Dict]:
        """Parse une chaîne de paramètres en objets structurés"""
        if not param_string.strip():
            return []

        params = []
        param_parts = param_string.split(',')

        for part in param_parts:
            part = part.strip()
            if not part:
                continue

            # Patterns pour différents formats de paramètres
            patterns = [
                r'(\w+)\s+(\w+)',  # Type nom
                r'(\w+)\s*:\s*(\w+)',  # nom: Type
                r'(\w+)\s*=\s*([^,]+)',  # nom=valeur
            ]

            for pattern in patterns:
                match = re.match(pattern, part)
                if match:
                    if '=' in pattern:
                        params.append({
                            'name': match.group(1),
                            'default_value': match.group(2)
                        })
                    else:
                        params.append({
                            'type': match.group(1),
                            'name': match.group(2)
                        })
                    break
            else:
                params.append({'name': part})

        return params

    def _determine_function_scope(self, content: str, func_name: str, position: int) -> str:
        """Détermine la portée d'une fonction"""
        # Vérifier si c'est une méthode de classe
        before = content[:position]
        if 'class' in before and 'def' in before[before.rfind('class'):]:
            return 'instance_method'

        # Vérifier si c'est statique
        func_start = content.rfind('def ', 0, position)
        if func_start != -1:
            line_start = content.rfind('\n', 0, func_start) + 1
            line = content[line_start:func_start]
            if '@staticmethod' in line or 'static' in line:
                return 'static_method'

        return 'global_function'

    def _extract_api_handler(self, content: str, start_pos: int) -> Dict:
        """Extrait les informations du gestionnaire d'API"""
        # Trouver la fonction suivante après la route
        rest = content[start_pos:]

        # Chercher la prochaine fonction/méthode
        func_patterns = [
            r'def\s+(\w+)\s*\((.*?)\)',
            r'function\s+(\w+)\s*\((.*?)\)',
            r'(\w+)\s*\((.*?)\)\s*{'
        ]

        for pattern in func_patterns:
            match = re.search(pattern, rest)
            if match:
                return {
                    'name': match.group(1),
                    'parameters': self._parse_parameters(match.group(2))
                }

        return {'name': 'anonymous', 'parameters': []}

    def _extract_class_block(self, content: str, class_name: str) -> str:
        """Extrait le bloc de code d'une classe"""
        pattern = rf'class\s+{class_name}.*?(?=\nclass\s+\w+|\Z)'
        match = re.search(pattern, content, re.DOTALL | re.MULTILINE)
        return match.group(0) if match else ''

    def _determine_visibility(self, class_block: str, attr_name: str) -> str:
        """Détermine la visibilité d'un attribut"""
        # Chercher la déclaration de l'attribut
        pattern = rf'(private|protected|public)\s+.*?\b{attr_name}\b'
        match = re.search(pattern, class_block, re.MULTILINE)

        if match:
            return match.group(1)

        # Par défaut
        lines = class_block.split('\n')
        for line in lines:
            if attr_name in line:
                if line.strip().startswith('_'):
                    return 'protected' if not line.strip().startswith('__') else 'private'

        return 'public'

    def _extract_default_value(self, class_block: str, attr_name: str) -> str:
        """Extrait la valeur par défaut d'un attribut"""
        pattern = rf'\b{attr_name}\s*=\s*([^;\n]+)'
        match = re.search(pattern, class_block)
        return match.group(1).strip() if match else ''

    def _determine_method_visibility(self, class_block: str, method_name: str) -> str:
        """Détermine la visibilité d'une méthode"""
        pattern = rf'(private|protected|public)\s+.*?\b{method_name}\s*\('
        match = re.search(pattern, class_block, re.MULTILINE)

        if match:
            return match.group(1)

        # Pour Python
        lines = class_block.split('\n')
        for line in lines:
            if f'def {method_name}' in line:
                if line.strip().startswith('_'):
                    return 'protected' if not line.strip().startswith('__') else 'private'

        return 'public'

    def _build_generative_prompt(self, elements: Dict, architecture: Dict) -> str:
        """Construit le prompt pour générer les descriptions"""
        prompt = f"""Générez des spécifications techniques qui auraient permis de créer ce code.

ARCHITECTURE DÉTECTÉE:
{architecture.get('summary', 'Non disponible')}

ÉLÉMENTS DE CODE DÉTECTÉS:
"""

        if elements['classes']:
            prompt += "\nCLASSES:\n"
            for cls in elements['classes'][:10]:  # Limiter pour éviter des prompts trop longs
                prompt += f"- {cls['name']}"
                if cls.get('extends'):
                    prompt += f" extends {cls['extends']}"
                if cls.get('implements'):
                    prompt += f" implements {', '.join(cls['implements'])}"
                prompt += "\n"

        if elements['functions']:
            prompt += f"\nFONCTIONS: {len(elements['functions'])} fonctions détectées\n"

        if elements['apis']:
            prompt += "\nAPIs:\n"
            for api in elements['apis'][:5]:
                prompt += f"- {api['method']} {api['path']}\n"

        prompt += """
EXIGENCES:
1. Fournissez des spécifications techniques précises
2. Décrivez les signatures de fonctions/méthodes attendues
3. Spécifiez les structures de données
4. Incluez les contrats d'interface
5. Décrivez les flux de données
6. Précisez les dépendances et intégrations

FORMAT ATTENDU:
- Objectif du système
- Spécifications techniques détaillées
- Contrats d'API
- Modèles de données
- Architecture logicielle
"""

        return prompt

    def _extract_specifications(self, response: str) -> Dict[str, Any]:
        """Extrait les spécifications structurées de la réponse LLM"""
        specs = {
            'system_objectives': [],
            'technical_specs': [],
            'api_contracts': [],
            'data_models': [],
            'architecture': {},
            'dependencies': []
        }

        # Sections de base à extraire
        sections = {
            'Objectif': 'system_objectives',
            'Spécification': 'technical_specs',
            'API': 'api_contracts',
            'Modèle': 'data_models',
            'Architecture': 'architecture',
            'Dépendance': 'dependencies'
        }

        lines = response.split('\n')
        current_section = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Détecter les nouvelles sections
            for section_key, section_name in sections.items():
                if line.lower().startswith(section_key.lower()):
                    current_section = section_name
                    break

            # Ajouter le contenu à la section courante
            if current_section and line and not any(line.lower().startswith(k.lower()) for k in sections.keys()):
                if current_section == 'architecture' and isinstance(specs[current_section], dict):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        specs[current_section][key.strip()] = value.strip()
                else:
                    if isinstance(specs[current_section], list):
                        specs[current_section].append(line)

        return specs

    def _generate_code_templates(self, elements: Dict) -> Dict[str, List[str]]:
        """Génère des templates de code à partir des éléments via des fichiers Jinja2.
        Si Jinja2 ou les fichiers ne sont pas disponibles, revient à la génération précédente.
        """
        templates: Dict[str, List[str]] = {
            'class_templates': [],
            'function_templates': [],
            'api_templates': []
        }

        if self.jinja_env is not None:
            try:
                class_tmpl = self.jinja_env.get_template('class_template.jinja2')
                func_tmpl = self.jinja_env.get_template('function_template.jinja2')
                api_tmpl = self.jinja_env.get_template('api_template.jinja2')

                for cls in elements.get('classes', [])[:5]:
                    templates['class_templates'].append(class_tmpl.render(cls=cls))

                for func in elements.get('functions', [])[:5]:
                    templates['function_templates'].append(func_tmpl.render(func=func))

                for api in elements.get('apis', [])[:5]:
                    templates['api_templates'].append(api_tmpl.render(api=api))

                return templates
            except Exception as e:
                logger.warning(f"Rendu Jinja2 indisponible, fallback utilisé: {e}")

        # Fallback: génération simple via assemblage de chaînes
        for cls in elements['classes'][:5]:  # Limiter à 5 pour éviter la surcharge
            template = f"class {cls['name']}:\n"
            if cls.get('attributes'):
                template += "    def __init__(self"
                for attr in cls['attributes']:
                    template += f", {attr['name']}"
                template += "):\n"
                for attr in cls['attributes']:
                    template += f"        self.{attr['name']} = {attr['name']}\n"
            else:
                template += "    pass\n"
            templates['class_templates'].append(template)

        for func in elements['functions'][:5]:
            template = f"def {func['name']}("
            if func.get('parameters'):
                template += ', '.join([p.get('name', 'param') for p in func['parameters']])
            template += "):\n    # TODO: Implémenter la fonction\n    pass\n"
            templates['function_templates'].append(template)

        for api in elements['apis'][:5]:
            method = api.get('method', 'GET')
            path = api.get('path', '/')
            handler = api.get('handler') or 'handle_request'
            template = f"@app.route('{path}', methods=['{method}'])\n"
            template += f"def {handler}():\n"
            template += "    # TODO: Implémenter la logique de l'API\n    return jsonify({{}})\n"
            templates['api_templates'].append(template)

        return templates

    def _generate_project_specification(self, elements: Dict, architecture: Dict) -> Dict[str, Any]:
        """Génère la spécification du projet"""
        return {
            'project_name': architecture.get('name', 'Unknown Project'),
            'description': architecture.get('description', ''),
            'tech_stack': architecture.get('technologies', []),
            'key_features': self._extract_key_features(elements),
            'project_structure': architecture.get('structure', {}),
            'code_quality_metrics': {
                'total_classes': len(elements['classes']),
                'total_functions': len(elements['functions']),
                'total_apis': len(elements['apis']),
                'test_coverage': len(elements['tests']) / max(len(elements['classes']), 1)
            }
        }

    def _generate_class_specifications(self, classes: List[Dict]) -> List[Dict]:
        """Génère des spécifications détaillées pour chaque classe"""
        specs = []

        for cls in classes:
            spec = {
                'class_name': cls['name'],
                'inheritance': {
                    'extends': cls.get('extends'),
                    'implements': cls.get('implements', [])
                },
                'attributes': cls.get('attributes', []),
                'methods': cls.get('methods', []),
                'responsibilities': self._infer_class_responsibilities(cls),
                'design_patterns': self._identify_design_patterns(cls)
            }
            specs.append(spec)

        return specs

    def _generate_api_specifications(self, apis: List[Dict]) -> List[Dict]:
        """Génère des spécifications d'API"""
        specs = []

        for api in apis:
            spec = {
                'endpoint': api['path'],
                'method': api['method'],
                'handler': api['handler'],
                'parameters': api.get('parameters', []),
                'expected_response': {
                    'status_codes': self._infer_status_codes(api['method']),
                    'content_type': 'application/json',
                    'response_schema': self._infer_response_schema(api)
                },
                'authentication': self._infer_authentication(api['path']),
                'rate_limiting': self._infer_rate_limiting(api['method'])
            }
            specs.append(spec)

        return specs

    def _generate_data_model_specifications(self, classes: List[Dict]) -> List[Dict]:
        """Génère des spécifications de modèles de données"""
        data_models = []

        for cls in classes:
            # Identifier les classes qui sont probablement des modèles de données
            if self._is_data_model_class(cls):
                model = {
                    'model_name': cls['name'],
                    'fields': [
                        {
                            'name': attr['name'],
                            'type': attr.get('type', 'unknown'),
                            'required': not attr.get('default_value'),
                            'constraints': self._infer_field_constraints(attr)
                        }
                        for attr in cls.get('attributes', [])
                    ],
                    'relationships': self._infer_relationships(cls),
                    'validation_rules': self._infer_validation_rules(cls)
                }
                data_models.append(model)

        return data_models

    def _extract_architecture_decisions(self, architecture: Dict) -> List[str]:
        """Extrait les décisions d'architecture"""
        decisions = []

        patterns = architecture.get('patterns', [])
        if patterns:
            decisions.append(f"Patterns architecturaux utilisés: {', '.join(patterns)}")

        tech_stack = architecture.get('technologies', [])
        if tech_stack:
            decisions.append(f"Stack technologique: {', '.join(tech_stack)}")

        structure = architecture.get('structure', {})
        if 'layers' in structure:
            decisions.append(f"Architecture en couches: {structure['layers']}")

        return decisions

    def _extract_key_features(self, elements: Dict) -> List[str]:
        """Extrait les fonctionnalités clés du code"""
        features = []

        # Basé sur les classes
        for cls in elements['classes'][:10]:
            if any(keyword in cls['name'].lower() for keyword in ['service', 'manager', 'controller', 'handler']):
                features.append(f"{cls['name']} - Gestion des opérations métier")
            elif any(keyword in cls['name'].lower() for keyword in ['repository', 'dao', 'persistence']):
                features.append(f"{cls['name']} - Gestion de la persistance des données")
            elif any(keyword in cls['name'].lower() for keyword in ['model', 'entity', 'dto']):
                features.append(f"{cls['name']} - Représentation des données")

        # Basé sur les APIs
        api_methods = {}
        for api in elements['apis']:
            method = api['method']
            api_methods[method] = api_methods.get(method, 0) + 1

        for method, count in api_methods.items():
            features.append(f"{count} endpoints {method} - API REST")

        return list(set(features))[:10]  # Limiter à 10 features uniques

    def _infer_class_responsibilities(self, cls: Dict) -> List[str]:
        """Infère les responsabilités d'une classe"""
        responsibilities = []
        name_lower = cls['name'].lower()

        if 'service' in name_lower:
            responsibilities.append('Gestion de la logique métier')
        if 'controller' in name_lower:
            responsibilities.append('Gestion des requêtes HTTP')
            responsibilities.append('Orchestration des services')
        if 'repository' in name_lower or 'dao' in name_lower:
            responsibilities.append('Accès aux données')
            responsibilities.append('Persistance')
        if 'model' in name_lower or 'entity' in name_lower:
            responsibilities.append('Représentation des données')
        if 'factory' in name_lower:
            responsibilities.append('Création d\'objets')
        if 'adapter' in name_lower:
            responsibilities.append('Adaptation d\'interfaces')
        if 'decorator' in name_lower:
            responsibilities.append('Ajout de fonctionnalités')

        if not responsibilities:
            responsibilities.append('Logique applicative')

        return responsibilities

    def _identify_design_patterns(self, cls: Dict) -> List[str]:
        """Identifie les design patterns utilisés"""
        patterns = []
        name_lower = cls['name'].lower()

        if 'factory' in name_lower:
            patterns.append('Factory Pattern')
        if 'singleton' in name_lower:
            patterns.append('Singleton Pattern')
        if 'builder' in name_lower:
            patterns.append('Builder Pattern')
        if 'adapter' in name_lower:
            patterns.append('Adapter Pattern')
        if 'decorator' in name_lower:
            patterns.append('Decorator Pattern')
        if 'observer' in name_lower:
            patterns.append('Observer Pattern')
        if 'strategy' in name_lower:
            patterns.append('Strategy Pattern')

        return patterns

    def _infer_status_codes(self, method: str) -> str:
        """Infère les codes de statut HTTP attendus"""
        status_codes = {
            'GET': ['200 OK', '404 Not Found'],
            'POST': ['201 Created', '400 Bad Request'],
            'PUT': ['200 OK', '204 No Content'],
            'DELETE': ['204 No Content', '404 Not Found'],
            'PATCH': ['200 OK', '400 Bad Request']
        }

        return status_codes.get(method, '200 OK')

    def _infer_response_schema(self, api: Dict) -> Dict[str, Any]:
        """Infère le schéma de réponse"""
        return {
            'type': 'object',
            'properties': {
                'data': {'type': 'object'},
                'status': {'type': 'string'},
                'message': {'type': 'string'}
            }
        }

    def _infer_authentication(self, path: str) -> str:
        """Infère le type d'authentification requis"""
        if any(secured in path.lower() for secured in ['admin', 'secure', 'private', 'user']):
            return 'JWT Token Required'
        return 'Public Access'

    def _infer_rate_limiting(self, method: str) -> Dict[str, Any]:
        """Infère les limites de taux"""
        limits = {
            'GET': {'requests_per_minute': 60},
            'POST': {'requests_per_minute': 30},
            'PUT': {'requests_per_minute': 30},
            'DELETE': {'requests_per_minute': 20},
            'PATCH': {'requests_per_minute': 30}
        }

        return limits.get(method, {'requests_per_minute': 60})

    def _is_data_model_class(self, cls: Dict) -> bool:
        """Détermine si une classe est un modèle de données"""
        name_lower = cls['name'].lower()
        return any(keyword in name_lower for keyword in ['model', 'entity', 'dto', 'vo', 'bo', 'pojo'])

    def _infer_field_constraints(self, field: Dict) -> Dict[str, Any]:
        """Infère les contraintes d'un champ"""
        constraints = {}

        if field.get('type'):
            constraints['type'] = field['type']

        if not field.get('default_value'):
            constraints['required'] = True

        return constraints

    def _infer_relationships(self, cls: Dict) -> List[Dict]:
        """Infère les relations entre modèles"""
        relationships = []

        for attr in cls.get('attributes', []):
            attr_type = attr.get('type', '').lower()
            if any(model_type in attr_type for model_type in ['list', 'array', 'collection']):
                relationships.append({
                    'type': 'OneToMany',
                    'target': attr_type.replace('list', '').replace('array', '').strip('<>[]'),
                    'field': attr['name']
                })
            elif any(keyword in attr_type for keyword in ['model', 'entity', 'dto']):
                relationships.append({
                    'type': 'ManyToOne',
                    'target': attr_type,
                    'field': attr['name']
                })

        return relationships

    def _infer_validation_rules(self, cls: Dict) -> Dict[str, List[str]]:
        """Infère les règles de validation"""
        rules = {}

        for attr in cls.get('attributes', []):
            attr_rules = []

            if attr.get('type') == 'String':
                attr_rules.append('NotBlank')
            elif attr.get('type') in ['Integer', 'Long', 'Float', 'Double']:
                attr_rules.append('NotNull')
                if 'id' in attr['name'].lower():
                    attr_rules.append('Positive')
            elif 'email' in attr['name'].lower():
                attr_rules.append('Email')

            if attr_rules:
                rules[attr['name']] = attr_rules

        return rules
