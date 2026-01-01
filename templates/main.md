# {{ metadata.project_name }} - Règles de Codage

*Généré le {{ generation_date }}*

## 📋 Table des Matières
{{ formatters.list(results.get('agents_executed', [])) }}

## 🏗️ Architecture
{{ results.architecture.content if results.architecture else 'Non disponible' }}

## 🔍 Patterns
{{ results.patterns.content if results.patterns else 'Non disponible' }}

## 📝 Règles
{{ results.rules.content if results.rules else 'Non disponible' }}

## 📊 Statistiques
- Fichiers analysés: {{ metadata.total_files }}
- Règles générées: {{ metadata.total_rules }}
- Patterns identifiés: {{ metadata.total_patterns }}
