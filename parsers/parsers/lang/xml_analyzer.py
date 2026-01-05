# xml_analyzer.py
import logging
import re
from typing import Dict, List, Any, Optional
from datetime import datetime
import xml.etree.ElementTree as ET

from parsers.analyzer import Analyzer
from parsers.analysis_result import (
    AnalysisResult, AnalysisStatus, FileType, FrameworkType,
    CodeElement, FileMetrics, PatternDetection, DependencyInfo,
    SecurityAnalysis, SectionAnalysis
)

logger = logging.getLogger(__name__)


class XMLAnalyzer(Analyzer):
    """Analyseur de fichiers XML retournant des AnalysisResult"""

    def __init__(self):
        super().__init__()
        self.file_type = FileType.XML

    def analyze_file(self, file_path: str) -> AnalysisResult:
        """Analyse un fichier XML et retourne un AnalysisResult"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Mesurer le temps de traitement
            start_time = datetime.now()

            # Créer le résultat de base
            result = self._create_base_result(file_path)
            result.analyzer_name = "XMLAnalyzer"

            # Analyser le contenu
            analysis = self.analyze_content(content, file_path)

            # Mettre à jour les résultats
            self._update_result_from_analysis(result, analysis)
            result.metrics = self._calculate_xml_metrics(content, analysis)

            # Calculer le temps de traitement
            result.processing_time_ms = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )

            return result

        except Exception as e:
            logger.error(f"Error analyzing XML file {file_path}: {e}")
            return self._create_error_result(file_path, str(e))

    def analyze_content(self, content: str, file_path: str) -> Dict[str, Any]:
        """Analyse du contenu XML (méthode interne)"""
        try:
            # Essayer d'utiliser ElementTree pour une analyse plus précise
            return self._analyze_with_elementtree(content, file_path)
        except ET.ParseError as e:
            # Fallback vers l'analyse regex si le XML n'est pas bien formé
            logger.warning(f"XML parsing error for {file_path}: {e}, using regex fallback")
            return self._analyze_with_regex(content, file_path)
        except Exception as e:
            logger.error(f"Error in XML analysis for {file_path}: {e}")
            return self._analyze_with_regex(content, file_path)

    def _analyze_with_elementtree(self, content: str, file_path: str) -> Dict[str, Any]:
        """Analyse XML avec ElementTree (plus précis)"""
        try:
            root = ET.fromstring(content)

            return {
                'is_valid': True,
                'parsing_method': 'elementtree',
                'root_element': self._extract_root_element_et(root),
                'elements': self._extract_elements_et(root),
                'attributes': self._extract_attributes_et(root),
                'namespaces': self._extract_namespaces_et(root),
                'text_content': self._extract_text_content_et(root),
                'processing_instructions': self._extract_processing_instructions(content),
                'comments': self._extract_comments(content),
                'cdata_sections': self._extract_cdata_sections(content),
                'analysis': self._analyze_xml_structure_et(root, content),
                'schema_info': self._extract_schema_info(content)
            }
        except ET.ParseError as e:
            raise e

    def _analyze_with_regex(self, content: str, file_path: str) -> Dict[str, Any]:
        """Analyse XML avec regex (fallback)"""
        return {
            'is_valid': False,
            'parsing_method': 'regex',
            'root_element': self._extract_root_element(content),
            'elements': self._extract_elements(content),
            'attributes': self._extract_attributes(content),
            'namespaces': self._extract_namespaces(content),
            'analysis': self._analyze_xml_structure(content),
            'schema_info': self._extract_schema_info(content),
            'raw_content_preview': content[:1000]
        }

    def _update_result_from_analysis(self, result: AnalysisResult, analysis: Dict[str, Any]) -> None:
        """Met à jour l'AnalysisResult avec les données de l'analyse XML"""

        if not analysis.get('is_valid', False):
            result.status = AnalysisStatus.PARTIAL
            result.warnings.append("XML may not be well-formed, analysis limited")

        # Ajouter les éléments de code (éléments XML)
        self._add_xml_elements(result, analysis)

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

    def _add_xml_elements(self, result: AnalysisResult, analysis: Dict[str, Any]) -> None:
        """Convertit les éléments XML en CodeElement"""

        # Élément racine
        root_element = analysis.get('root_element', {})
        if root_element:
            root_elem = CodeElement(
                name=root_element.get('name', 'unknown'),
                element_type='xml_root',
                metadata={
                    'attributes': root_element.get('attributes', {}),
                    'namespaces': root_element.get('namespaces', []),
                    'type': 'xml_root_element',
                    'is_root': True
                }
            )
            result.elements.append(root_elem)
            result.metrics.class_count += 1

        # Éléments réguliers
        for elem in analysis.get('elements', []):
            element = CodeElement(
                name=elem['name'],
                element_type='xml_element',
                metadata={
                    'attributes': elem.get('attributes', {}),
                    'is_self_closing': elem.get('is_self_closing', False),
                    'has_children': elem.get('has_children', False),
                    'child_count': elem.get('child_count', 0),
                    'depth': elem.get('depth', 0),
                    'namespace': elem.get('namespace'),
                    'type': 'xml_element'
                }
            )
            result.elements.append(element)
            result.metrics.import_count += 1  # Utiliser import_count pour compter les éléments

        # Attributs (comme éléments séparés)
        for attr in analysis.get('attributes', []):
            attribute = CodeElement(
                name=attr['name'],
                element_type='xml_attribute',
                metadata={
                    'value': attr.get('value', ''),
                    'parent_element': attr.get('parent_element'),
                    'type': 'xml_attribute',
                    'is_namespaced': ':' in attr.get('name', '')
                }
            )
            result.elements.append(attribute)

    def _update_dependencies(self, result: AnalysisResult, analysis: Dict[str, Any]) -> None:
        """Met à jour les informations de dépendances"""
        deps = result.dependencies

        # Extraire les schémas XSD
        schema_info = analysis.get('schema_info', {})
        if schema_info.get('xsd_location'):
            deps.external_deps.append(schema_info['xsd_location'])

        # Extraire les namespaces comme dépendances
        namespaces = analysis.get('namespaces', [])
        for ns in namespaces:
            uri = ns.get('uri', '')
            if uri.startswith('http://') or uri.startswith('https://'):
                deps.external_deps.append(uri)

        # Détecter les références dans les attributs
        for attr in analysis.get('attributes', []):
            value = attr.get('value', '')
            if value.startswith('classpath:') or value.startswith('file:'):
                deps.internal_deps.append(value)
            elif value.startswith('http://') or value.startswith('https://'):
                deps.external_deps.append(value)

        # Déterminer le package manager basé sur le type de schéma
        schema_type = analysis.get('analysis', {}).get('schema_type', '')
        if schema_type == 'maven_pom':
            deps.package_manager = 'maven'
        elif schema_type == 'web_xml':
            deps.package_manager = 'maven'  # Souvent avec Maven

    def _update_patterns(self, result: AnalysisResult, analysis: Dict[str, Any]) -> None:
        """Met à jour la détection de patterns"""
        patterns = result.patterns
        xml_analysis = analysis.get('analysis', {})
        schema_info = analysis.get('schema_info', {})

        # Détecter le type de schéma
        schema_type = xml_analysis.get('schema_type', '')
        if schema_type:
            patterns.patterns.append(f'{schema_type}')

            # Associer les frameworks
            if schema_type == 'spring_config':
                patterns.frameworks.append(FrameworkType.SPRING)
            elif schema_type == 'web_xml':
                patterns.frameworks.append(FrameworkType.SPRING)  # Pour les applications web Java
            elif schema_type == 'maven_pom':
                patterns.frameworks.append(FrameworkType.SPRING)  # Maven souvent avec Spring

        # Détecter les namespaces spécifiques
        namespaces = analysis.get('namespaces', [])
        for ns in namespaces:
            uri = ns.get('uri', '')
            if 'springframework' in uri:
                patterns.frameworks.append(FrameworkType.SPRING)
                patterns.libraries.append('spring-framework')
            elif 'javax' in uri or 'jakarta' in uri:
                patterns.libraries.append('java-ee')
            elif 'maven' in uri:
                patterns.libraries.append('maven')

        # Architecture hints basées sur la structure
        if xml_analysis.get('element_count', 0) > 50:
            patterns.architecture_hints.append('complex_xml_structure')

        if xml_analysis.get('max_depth', 0) > 5:
            patterns.architecture_hints.append('deeply_nested_xml')

        if analysis.get('cdata_sections'):
            patterns.architecture_hints.append('contains_cdata')

        if schema_info.get('has_schema_reference'):
            patterns.architecture_hints.append('schema_validated')

    def _update_security(self, result: AnalysisResult, analysis: Dict[str, Any]) -> None:
        """Met à jour l'analyse de sécurité"""
        security = result.security

        # Détecter les entités externes (XXE vulnerability)
        if self._has_external_entities(analysis):
            security.vulnerabilities.append({
                'type': 'xxe_vulnerability',
                'severity': 'high',
                'description': 'XML file may be vulnerable to XXE attacks',
                'recommendation': 'Disable external entity processing in XML parser'
            })
            security.warnings.append("Potential XXE vulnerability detected")

        # Vérifier les DTDs externes
        if self._has_external_dtd(analysis):
            security.warnings.append("External DTD reference found")
            security.recommendations.append("Consider using local DTDs or disable DTD processing")

        # Vérifier les schémas non sécurisés
        schema_info = analysis.get('schema_info', {})
        if schema_info.get('xsd_location', '').startswith('http://'):
            security.warnings.append("HTTP schema reference (not HTTPS)")
            security.recommendations.append("Use HTTPS for schema references")

        # Détecter les injections potentielles dans les attributs
        for attr in analysis.get('attributes', []):
            value = attr.get('value', '')
            if any(injection_pattern in value for injection_pattern in ['${', '#{', '<?', '<script']):
                security.warnings.append(f"Potential injection in attribute: {attr.get('name')}")

        # Calculer le score de sécurité
        security.security_score = self._calculate_security_score(analysis)

    def _update_language_specific(self, result: AnalysisResult, analysis: Dict[str, Any]) -> None:
        """Met à jour les données spécifiques au XML"""
        xml_analysis = analysis.get('analysis', {})
        schema_info = analysis.get('schema_info', {})

        result.language_specific = {
            'xml': {
                'is_valid': analysis.get('is_valid', False),
                'parsing_method': analysis.get('parsing_method', 'unknown'),
                'schema_type': xml_analysis.get('schema_type', 'generic'),
                'encoding': self._detect_encoding(analysis),
                'has_prolog': self._has_xml_prolog(analysis),
                'element_count': xml_analysis.get('element_count', 0),
                'attribute_count': xml_analysis.get('attribute_count', 0),
                'namespace_count': len(analysis.get('namespaces', [])),
                'max_depth': xml_analysis.get('max_depth', 0),
                'has_cdata': xml_analysis.get('has_cdata', False),
                'has_comments': xml_analysis.get('has_comments', False),
                'has_processing_instructions': bool(analysis.get('processing_instructions')),
                'schema_location': schema_info.get('xsd_location'),
                'target_namespace': schema_info.get('target_namespace'),
                'is_config_file': self._is_configuration_file(analysis),
                'is_data_file': self._is_data_file(analysis),
                'is_web_descriptor': xml_analysis.get('schema_type') == 'web_xml'
            }
        }

    def _add_diagnostics(self, result: AnalysisResult, analysis: Dict[str, Any]) -> None:
        """Ajoute des diagnostics et recommandations"""

        xml_analysis = analysis.get('analysis', {})

        # Vérifier la validité
        if not analysis.get('is_valid', False):
            result.warnings.append("XML may not be well-formed")
            result.recommendations.append("Validate XML syntax")

        # Vérifier la complexité
        if xml_analysis.get('max_depth', 0) > 10:
            result.warnings.append(f"Deeply nested XML (depth: {xml_analysis['max_depth']})")
            result.recommendations.append("Consider flattening XML structure")

        # Vérifier les namespaces
        namespaces = analysis.get('namespaces', [])
        if len(namespaces) > 10:
            result.notes.append(f"Many namespaces: {len(namespaces)}")

        # Vérifier les commentaires
        if xml_analysis.get('has_comments'):
            result.notes.append("XML contains comments")

        # Recommandations pour les fichiers de configuration
        if self._is_configuration_file(analysis):
            if not analysis.get('schema_info', {}).get('has_schema_reference'):
                result.recommendations.append("Consider adding schema validation")

    def _extract_root_element_et(self, root: ET.Element) -> Dict[str, Any]:
        """Extrait l'élément racine avec ElementTree"""
        return {
            'name': root.tag.split('}')[-1] if '}' in root.tag else root.tag,
            'attributes': dict(root.attrib),
            'namespaces': self._extract_namespaces_from_element(root),
            'text': (root.text or '').strip() if root.text else None,
            'child_count': len(root)
        }

    def _extract_elements_et(self, root: ET.Element, depth: int = 0) -> List[Dict]:
        """Extrait récursivement les éléments avec ElementTree"""
        elements = []

        def traverse(element, current_depth):
            # Nom de l'élément (sans namespace)
            tag_name = element.tag.split('}')[-1] if '}' in element.tag else element.tag

            element_info = {
                'name': tag_name,
                'attributes': dict(element.attrib),
                'depth': current_depth,
                'has_children': len(element) > 0,
                'child_count': len(element),
                'is_self_closing': len(element) == 0 and not element.text,
                'namespace': element.tag.split('}')[0][1:] if '}' in element.tag else None,
                'text': (element.text or '').strip() if element.text else None
            }
            elements.append(element_info)

            # Traverser les enfants
            for child in element:
                traverse(child, current_depth + 1)

        traverse(root, 0)
        return elements

    def _extract_attributes_et(self, root: ET.Element) -> List[Dict]:
        """Extrait tous les attributs avec ElementTree"""
        attributes = []

        def extract_from_element(element):
            for name, value in element.attrib.items():
                # Nom d'attribut sans namespace
                attr_name = name.split('}')[-1] if '}' in name else name

                attributes.append({
                    'name': attr_name,
                    'value': value,
                    'parent_element': element.tag.split('}')[-1] if '}' in element.tag else element.tag,
                    'namespace': name.split('}')[0][1:] if '}' in name else None
                })

            for child in element:
                extract_from_element(child)

        extract_from_element(root)
        return attributes

    def _extract_namespaces_et(self, root: ET.Element) -> List[Dict]:
        """Extrait les namespaces avec ElementTree"""
        namespaces = []

        # Extraire les namespaces de l'élément racine et ses enfants
        def extract_namespaces(element):
            for name, value in element.attrib.items():
                if name.startswith('xmlns'):
                    prefix = name.split(':')[1] if ':' in name else 'default'
                    namespaces.append({
                        'prefix': prefix,
                        'uri': value
                    })

            for child in element:
                extract_namespaces(child)

        extract_namespaces(root)

        # Dédupliquer
        unique_namespaces = []
        seen = set()
        for ns in namespaces:
            key = (ns['prefix'], ns['uri'])
            if key not in seen:
                seen.add(key)
                unique_namespaces.append(ns)

        return unique_namespaces

    def _extract_namespaces_from_element(self, element: ET.Element) -> Dict[str, str]:
        """Extrait les namespaces d'un élément spécifique"""
        namespaces = {}
        for name, value in element.attrib.items():
            if name.startswith('xmlns'):
                prefix = name.split(':')[1] if ':' in name else 'default'
                namespaces[prefix] = value
        return namespaces

    def _extract_text_content_et(self, root: ET.Element) -> Dict[str, Any]:
        """Extrait le contenu textuel"""
        text_parts = []

        def collect_text(element):
            if element.text and element.text.strip():
                text_parts.append(element.text.strip())
            for child in element:
                collect_text(child)
            if element.tail and element.tail.strip():
                text_parts.append(element.tail.strip())

        collect_text(root)

        return {
            'total_text_length': sum(len(t) for t in text_parts),
            'text_parts_count': len(text_parts),
            'sample': ' '.join(text_parts[:3]) if text_parts else None
        }

    def _extract_processing_instructions(self, content: str) -> List[Dict]:
        """Extrait les instructions de traitement"""
        instructions = []
        pi_pattern = r'<\?(\w+)(.*?)\?>'

        for match in re.finditer(pi_pattern, content, re.DOTALL):
            instructions.append({
                'target': match.group(1),
                'content': match.group(2).strip(),
                'is_xml_declaration': match.group(1).lower() == 'xml'
            })

        return instructions

    def _extract_comments(self, content: str) -> List[Dict]:
        """Extrait les commentaires"""
        comments = []
        comment_pattern = r'<!--(.*?)-->'

        for match in re.finditer(comment_pattern, content, re.DOTALL):
            comments.append({
                'content': match.group(1).strip(),
                'length': len(match.group(1))
            })

        return comments

    def _extract_cdata_sections(self, content: str) -> List[Dict]:
        """Extrait les sections CDATA"""
        cdata_sections = []
        cdata_pattern = r'<!\[CDATA\[(.*?)\]\]>'

        for match in re.finditer(cdata_pattern, content, re.DOTALL):
            cdata_sections.append({
                'content': match.group(1),
                'length': len(match.group(1))
            })

        return cdata_sections

    def _analyze_xml_structure_et(self, root: ET.Element, content: str) -> Dict[str, Any]:
        """Analyse la structure XML avec ElementTree"""
        elements = self._extract_elements_et(root)

        return {
            'element_count': len(elements),
            'attribute_count': len(self._extract_attributes_et(root)),
            'max_depth': max((e['depth'] for e in elements), default=0),
            'avg_attributes_per_element': self._calculate_avg_attributes(elements),
            'has_cdata': 'CDATA' in content,
            'has_comments': '<!--' in content,
            'schema_type': self._detect_schema_type_et(root, content),
            'is_balanced': self._check_xml_balance(content)
        }

    def _extract_root_element(self, content: str) -> Dict[str, Any]:
        """Extrait l'élément racine avec regex (fallback)"""
        match = re.search(r'<(\w+:)?(\w+)[^>]*>', content)
        if match:
            return {
                'name': match.group(2),  # Nom sans namespace
                'full_tag': match.group(0),
                'attributes': self._extract_element_attributes(match.group(0)),
                'has_namespace': bool(match.group(1))
            }
        return {}

    def _extract_elements(self, content: str) -> List[Dict]:
        """Extrait les éléments avec regex (fallback)"""
        elements = []
        element_pattern = r'<(\w+:)?(\w+)([^>]*)>'

        for match in re.finditer(element_pattern, content):
            full_match = match.group(0)
            namespace_prefix = match.group(1)
            element_name = match.group(2)
            attributes_str = match.group(3)

            elements.append({
                'name': element_name,
                'namespace_prefix': namespace_prefix[:-1] if namespace_prefix else None,
                'attributes': self._extract_element_attributes(attributes_str),
                'is_self_closing': full_match.endswith('/>'),
                'is_closing': full_match.startswith('</'),
                'full_tag': full_match
            })

        return elements

    def _extract_element_attributes(self, attributes_str: str) -> Dict[str, str]:
        """Extrait les attributs d'un élément avec regex"""
        attributes = {}
        attr_pattern = r'(\w+:)?(\w+)\s*=\s*["\']([^"\']*)["\']'

        for match in re.finditer(attr_pattern, attributes_str):
            namespace_prefix = match.group(1)
            attr_name = match.group(2)
            attr_value = match.group(3)

            full_name = f"{namespace_prefix}{attr_name}" if namespace_prefix else attr_name
            attributes[full_name] = attr_value

        return attributes

    def _extract_attributes(self, content: str) -> List[Dict]:
        """Extrait tous les attributs avec regex"""
        attributes = []
        attr_pattern = r'(\w+:)?(\w+)\s*=\s*["\']([^"\']*)["\']'

        for match in re.finditer(attr_pattern, content):
            namespace_prefix = match.group(1)
            attr_name = match.group(2)
            attr_value = match.group(3)

            attributes.append({
                'name': attr_name,
                'full_name': f"{namespace_prefix}{attr_name}" if namespace_prefix else attr_name,
                'value': attr_value,
                'has_namespace': bool(namespace_prefix)
            })

        return attributes

    def _extract_namespaces(self, content: str) -> List[Dict]:
        """Extrait les namespaces avec regex"""
        namespaces = []
        ns_pattern = r'xmlns:?(\w*)\s*=\s*["\']([^"\']*)["\']'

        for match in re.finditer(ns_pattern, content):
            namespaces.append({
                'prefix': match.group(1) or 'default',
                'uri': match.group(2)
            })

        return namespaces

    def _analyze_xml_structure(self, content: str) -> Dict[str, Any]:
        """Analyse la structure XML avec regex (fallback)"""
        elements = self._extract_elements(content)

        return {
            'element_count': len([e for e in elements if not e['is_closing']]),
            'attribute_count': len(self._extract_attributes(content)),
            'max_depth': self._calculate_max_depth(content),
            'has_cdata': '<![CDATA[' in content,
            'has_comments': '<!--' in content,
            'schema_type': self._detect_schema_type(content),
            'is_balanced': self._check_xml_balance(content)
        }

    def _calculate_max_depth(self, content: str) -> int:
        """Calcule la profondeur maximale avec regex"""
        depth = 0
        max_depth = 0

        # Pattern pour les balises
        tag_pattern = r'</?(\w+:)?(\w+)[^>]*>'

        for match in re.finditer(tag_pattern, content):
            tag = match.group(0)
            if tag.startswith('</'):
                # Balise fermante
                depth = max(0, depth - 1)
            elif not tag.endswith('/>'):
                # Balise ouvrante (pas auto-fermante)
                depth += 1
                max_depth = max(max_depth, depth)

        return max_depth

    def _detect_schema_type(self, content: str) -> str:
        """Détecte le type de schéma avec regex"""
        content_lower = content.lower()

        if any(tag in content_lower for tag in ['web-app', 'servlet', 'filter', 'listener']):
            return 'web_xml'
        elif any(tag in content_lower for tag in ['beans', 'bean', 'context:', 'aop:', 'tx:']):
            return 'spring_config'
        elif any(tag in content_lower for tag in ['project', 'modelversion', 'groupid', 'artifactid']):
            return 'maven_pom'
        elif any(tag in content_lower for tag in ['configuration', 'appsettings', 'connectionstrings']):
            return 'app_config'
        elif any(tag in content_lower for tag in ['widget', 'window', 'button', 'label']):
            return 'ui_descriptor'
        elif any(tag in content_lower for tag in ['rss', 'channel', 'item']):
            return 'rss_feed'
        elif any(tag in content_lower for tag in ['soap:', 'envelope', 'body', 'header']):
            return 'soap_message'
        elif 'xs:schema' in content_lower or 'xsd:schema' in content_lower:
            return 'xml_schema'

        return 'generic'

    def _detect_schema_type_et(self, root: ET.Element, content: str) -> str:
        """Détecte le type de schéma avec ElementTree"""
        root_tag = root.tag.split('}')[-1] if '}' in root.tag else root.tag

        if root_tag in ['web-app', 'web-app-ext']:
            return 'web_xml'
        elif root_tag == 'beans':
            return 'spring_config'
        elif root_tag == 'project':
            return 'maven_pom'
        elif root_tag == 'configuration':
            return 'app_config'

        # Fallback vers la détection regex
        return self._detect_schema_type(content)

    def _extract_schema_info(self, content: str) -> Dict[str, Any]:
        """Extrait les informations de schéma"""
        schema_info = {
            'has_schema_reference': False,
            'xsd_location': None,
            'target_namespace': None,
            'schema_version': None
        }

        # Chercher schemaLocation
        schema_pattern = r'schemaLocation\s*=\s*["\']([^"\']*)["\']'
        match = re.search(schema_pattern, content, re.IGNORECASE)
        if match:
            schema_info['has_schema_reference'] = True
            schema_info['xsd_location'] = match.group(1).split()[1] if ' ' in match.group(1) else match.group(1)

        # Chercher targetNamespace
        ns_pattern = r'targetNamespace\s*=\s*["\']([^"\']*)["\']'
        match = re.search(ns_pattern, content, re.IGNORECASE)
        if match:
            schema_info['target_namespace'] = match.group(1)

        # Chercher version
        version_pattern = r'version\s*=\s*["\']([^"\']*)["\']'
        match = re.search(version_pattern, content)
        if match:
            schema_info['schema_version'] = match.group(1)

        return schema_info

    def _calculate_avg_attributes(self, elements: List[Dict]) -> float:
        """Calcule la moyenne d'attributs par élément"""
        if not elements:
            return 0.0

        total_attributes = sum(len(e.get('attributes', {})) for e in elements)
        return total_attributes / len(elements)

    def _check_xml_balance(self, content: str) -> bool:
        """Vérifie si le XML est bien équilibré"""
        opening_tags = []
        tag_pattern = r'</?(\w+:)?(\w+)[^>]*>'

        for match in re.finditer(tag_pattern, content):
            tag = match.group(0)
            element_name = match.group(2)

            if tag.startswith('</'):
                # Balise fermante
                if not opening_tags or opening_tags[-1] != element_name:
                    return False
                opening_tags.pop()
            elif not tag.endswith('/>'):
                # Balise ouvrante (pas auto-fermante)
                opening_tags.append(element_name)

        return len(opening_tags) == 0

    def _has_external_entities(self, analysis: Dict[str, Any]) -> bool:
        """Vérifie la présence d'entités externes"""
        content = analysis.get('raw_content_preview', '')
        if not content:
            return False

        # Vérifier les déclarations DTD externes
        external_patterns = [
            r'<!ENTITY.*SYSTEM.*',
            r'<!DOCTYPE.*SYSTEM.*',
            r'<!DOCTYPE.*PUBLIC.*'
        ]

        return any(re.search(pattern, content, re.IGNORECASE) for pattern in external_patterns)

    def _has_external_dtd(self, analysis: Dict[str, Any]) -> bool:
        """Vérifie la présence de DTDs externes"""
        content = analysis.get('raw_content_preview', '')
        if not content:
            return False

        return bool(re.search(r'<!DOCTYPE.*(SYSTEM|PUBLIC).*', content, re.IGNORECASE))

    def _has_xml_prolog(self, analysis: Dict[str, Any]) -> bool:
        """Vérifie la présence d'un prologue XML"""
        content = analysis.get('raw_content_preview', '')
        if not content:
            return False

        return content.strip().startswith('<?xml')

    def _detect_encoding(self, analysis: Dict[str, Any]) -> str:
        """Détecte l'encodage du XML"""
        content = analysis.get('raw_content_preview', '')
        if not content:
            return 'unknown'

        encoding_match = re.search(r'encoding\s*=\s*["\']([^"\']*)["\']', content, re.IGNORECASE)
        return encoding_match.group(1) if encoding_match else 'utf-8'

    def _is_configuration_file(self, analysis: Dict[str, Any]) -> bool:
        """Détermine si c'est un fichier de configuration"""
        schema_type = analysis.get('analysis', {}).get('schema_type', '')
        config_types = ['web_xml', 'spring_config', 'app_config', 'maven_pom']
        return schema_type in config_types

    def _is_data_file(self, analysis: Dict[str, Any]) -> bool:
        """Détermine si c'est un fichier de données"""
        schema_type = analysis.get('analysis', {}).get('schema_type', '')
        return schema_type in ['generic', 'rss_feed', 'soap_message']

    def _calculate_xml_metrics(self, content: str, analysis: Dict[str, Any]) -> FileMetrics:
        """Calcule les métriques spécifiques au XML"""
        metrics = super()._calculate_metrics(content)

        xml_analysis = analysis.get('analysis', {})

        # Mettre à jour avec les métriques spécifiques au XML
        metrics.total_elements = xml_analysis.get('element_count', 0)
        metrics.attribute_count = xml_analysis.get('attribute_count', 0)

        # Utiliser import_count pour les éléments et class_count pour la profondeur
        metrics.import_count = xml_analysis.get('element_count', 0)
        metrics.class_count = xml_analysis.get('max_depth', 0)

        # Complexité basée sur la profondeur et le nombre d'attributs
        complexity = (
                xml_analysis.get('max_depth', 0) * 2 +
                xml_analysis.get('attribute_count', 0) * 0.5 +
                len(analysis.get('namespaces', [])) * 0.3
        )
        metrics.complexity_score = complexity

        # Compter les lignes de XML valides
        lines = content.split('\n')
        xml_lines = len([l for l in lines if re.search(r'<[^>]+>', l)])
        metrics.code_lines = xml_lines

        return metrics

    def _calculate_security_score(self, analysis: Dict[str, Any]) -> float:
        """Calcule un score de sécurité pour XML"""
        score = 100.0

        # Pénalités pour les problèmes de sécurité
        if self._has_external_entities(analysis):
            score -= 40  # XXE vulnerability

        if self._has_external_dtd(analysis):
            score -= 30  # External DTD

        # Vérifier les schémas non sécurisés
        schema_info = analysis.get('schema_info', {})
        if schema_info.get('xsd_location', '').startswith('http://'):
            score -= 20  # HTTP instead of HTTPS

        # Bonus pour les bonnes pratiques
        if analysis.get('is_valid', False):
            score += 10

        if schema_info.get('has_schema_reference'):
            score += 5  # Schema validation

        return max(0, min(100, score))