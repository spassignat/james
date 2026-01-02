# 🧠 JAMES — AI-Assisted Code Analysis & Generation System

**JAMES** est un outil d’analyse, de compréhension et de génération de code assisté par IA.  
Il est conçu pour :
- analyser une base de code existante,
- en extraire la structure, les patterns et l’architecture,
- indexer le code dans une base vectorielle,
- permettre à des agents IA de proposer du refactoring, de la documentation ou de la génération de code.

Le projet est conçu pour être **modulaire, extensible et piloté par agents**.

---

## 🚀 Fonctionnalités principales

### 🔍 Analyse de projet
- Scan du projet (via FileScanner)
- Détection des fichiers pertinents
- Construction d’un `ProjectStructure`
- Identification de patterns

### 🧠 Vectorisation
- Indexation du code via **ChromaDB**
- Stockage des chunks + métadonnées
- Recherche sémantique

### 🤖 Agents IA
- Agent d’analyse
- Agent de génération
- Pipeline extensible
- Support multi-modèles (Ollama)

### 🧱 Architecture modulaire
- Séparation claire :
    - analyse
    - vectorisation
    - génération
    - orchestration

---

## 🗂️ Structure du projet

```
james/
│
├── agents/
│ ├── agent_manager.py
│ ├── analysis_agent.py
│ ├── generation_agent.py
│ └── project_analyzer.py
│
├── config/
│ ├── config_loader.py
│ ├── file_scanner.py
│
├── models/
│ ├── analysis_context.py
│ ├── code_chunk.py
│ ├── project_structure.py
│
├── vector/
│ ├── vector_store.py
│
├── main_analysis.py
├── rule_generator.py
└── README.md
```

---

## 🧩 Modèle de données

### `ProjectStructure`
Représente l’état du projet analysé :
- nom
- fichiers
- modules
- patterns détectés

### `CodeChunk`
Représente un fragment indexé :
- contenu
- fichier
- type
- métadonnées

### `AnalysisContext`
Objet central partagé entre agents :
- structure du projet
- chunks vectorisés
- configuration
- modèle IA utilisé

---

## 🔎 Vector Store

Basé sur **ChromaDB** :
- stockage persistant
- métadonnées strictes (pas de listes ou None)
- compatible RAG

⚠️ Les métadonnées doivent être :
```
str | int | float | bool | None
```

---

## 🧠 Modèles IA recommandés 

| Usage           | Modèle              |
|-----------------|---------------------|
| Analyse de code | deepseek-coder:6.7b |
| Refactoring     | deepseek-coder      |
| Génération      | codellama:13b       |
| Documentation   | llama3.1:8b         |

---

## ▶️ Lancer une analyse

```bash
python main_analysis.py
```


## 🧪 Debug & Logs

Les logs permettent de voir :

- fichiers indexés
- chunks créés
- erreurs d’indexation
- appels IA

## 🎯 Objectif long terme

- Génération automatique de refactorings
- Suggestions d’architecture
- Documentation automatique
- Agent capable de faire évoluer un projet complet

## 🧠 Philosophie

“Le code doit pouvoir s’expliquer, se transformer et s’améliorer de lui-même.”

**JAMES** est conçu comme un assistant d’ingénierie logicielle, pas comme un simple générateur de code.

---

# 🧠 Prompt maître pour ChatGPT (à garder précieusement)

Tu peux l’utiliser tel quel 👇

---

## 🧩 PROMPT — Assistant d’évolution du projet JAMES

Tu es un expert en architecture logicielle, IA appliquée au code, et en refactoring.

Tu travailles sur un projet nommé JAMES, dont l’objectif est :
- analyser une base de code existante
- la vectoriser
- permettre à des agents IA de comprendre
- améliorer et générer du code.

Le projet est structuré autour de :
- ProjectAnalyzer
- VectorStore (ChromaDB)
- AnalysisAgent / GenerationAgent
- AnalysisContext
- ProjectStructure
- CodeChunk

Contraintes importantes :
- Le code doit rester modulaire
- Les modèles IA sont appelés via Ollama
- Les métadonnées doivent être compatibles avec ChromaDB
- Le projet doit rester extensible et testable
- Pas de logique "magique" ou implicite

Ton rôle :
- Analyser l’architecture existante
- Identifier les incohérences ou manques
- Proposer des améliorations progressives
- Fournir du code propre, typé, maintenable
- Ne jamais inventer de fonctions non cohérentes avec l’existant
- Toujours expliquer les décisions techniques

Tu peux proposer :
- refactoring
- nouvelles classes
- amélioration du pipeline
- amélioration des modèles de données
- meilleure séparation analyse / génération
- amélioration du vector store
- Commence toujours par analyser l’existant avant de proposer une modification.

---

## 🚀 Prochaine étape recommandée

👉 **Étape suivante idéale** :  
Créer un **pipeline clair** :

```Scan → Analyse → Vectorisation → Raisonnement → Génération```

Si tu veux, je peux te proposer :
- 🔧 une version nettoyée de `main_analysis.py`
- 🧠 un vrai `AnalysisPipeline`
- 🧱 une architecture hexagonale
- 🤖 un AgentManager intelligent
- 🧪 un mode test / dry-run

Dis-moi simplement :
👉 **“on continue avec [X]”**
