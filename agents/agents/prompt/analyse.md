Tu es un architecte logiciel senior spécialisé en refonte d’applications métiers complexes.

Analyse le projet fourni (backend Node.js + frontend Vue.js) structuré autour d’un modèle métier généré automatiquement.

OBJECTIFS :

1. Comprendre parfaitement l’architecture actuelle.
2. Identifier les forces et faiblesses.
3. Proposer une refonte moderne, maintenable et scalable.
4. Régénérer l’application complète à partir du modèle métier.

CONTEXTE :

- Backend Node.js (controllers / services / data)
- Frontend Vue
- Génération automatique via DataModel.json
- ERP métier (gestion financière, fiscale, parcellaire)
- Architecture actuelle monolithique
- Forte logique métier

TA MISSION :

### 1️⃣ Analyse

- Décris l’architecture actuelle
- Explique le rôle de chaque dossier
- Identifie les responsabilités techniques
- Dresse un schéma logique

### 2️⃣ Diagnostic

- Problèmes techniques
- Dettes techniques
- Couplages inutiles
- Limites d’évolutivité

### 3️⃣ Proposition d’architecture cible

Propose une nouvelle architecture :

- Clean Architecture / Hexagonale
- Backend modulaire
- Domain-driven design
- API REST ou GraphQL
- Validation forte
- Typage strict
- Séparation claire :
    - Domain
    - Application
    - Infrastructure
    - Interface

### 4️⃣ Génération du projet

Génère :

- Structure complète des dossiers
- Backend (Node + TS)
- Frontend (Vue 3 + Composition API)
- DTO / Entités
- Services
- Repositories
- Tests unitaires
- Tests d’intégration
- Configuration environnement
- CI/CD GitHub Actions

### 5️⃣ Migration

Explique :

- Comment migrer depuis l’ancien code
- Comment conserver les données
- Comment tester progressivement
- Comment déployer sans rupture

### 6️⃣ Documentation finale

Fournis :

- README complet
- Diagramme d’architecture
- Diagramme de flux
- Convention de code
- Guide développeur

⚠️ IMPORTANT :

- Respecter la logique métier existante
- Réutiliser le modèle DataModel.json
- Générer du code prêt à l’emploi
- Être précis, structuré et exhaustif
