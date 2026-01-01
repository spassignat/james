# src/main.py
#!/usr/bin/env python3

import argparse
import logging
import sys
import os

from config.config_loader import ConfigLoader
from parsers.multilanguage_analyzer import MultiLanguageAnalyzer
from vector.code_vectorizer import CodeVectorizer

# Ajouter le chemin src pour les imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


# Importer vos analyseurs existants

def setup_logging(log_config: dict):
    """Configure le logging"""
    logging.basicConfig(
        level=getattr(logging, log_config.get('level', 'INFO')),
        format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_config.get('file', './vectorization.log'))
        ]
    )

def main():
    parser = argparse.ArgumentParser(description="Vectorisation de code pour Ollama")
    parser.add_argument('--config', '-c', default='config.yaml',
                       help='Fichier de configuration')
    parser.add_argument('--search', help='Rechercher du code similaire')
    parser.add_argument('--top-k', type=int, default=5, help='Nombre de résultats de recherche')
    parser.add_argument('--stats', action='store_true', help='Afficher les statistiques')
    
    args = parser.parse_args()
    
    try:
        # Chargement configuration
        config_loader = ConfigLoader(args.config)
        config = config_loader.config
        
        # Configuration logging
        setup_logging(config.get('logging', {}))
        
        # Initialisation avec VOS analyseurs existants
        analyzer = MultiLanguageAnalyzer(config)
        
        # Vectorisation
        vectorizer = CodeVectorizer(config, analyzer)
        
        if args.search:
            # Mode recherche
            results = vectorizer.search_similar_code(args.search, args.top_k)
            print(f"\n🔍 Résultats pour: '{args.search}'\n")
            for i, result in enumerate(results, 1):
                print(f"--- Résultat {i} ---")
                print(f"Fichier: {result['metadata']['file_path']}")
                print(f"Type: {result['metadata']['chunk_type']}")
                print(f"Contenu: {result['content'][:200]}...")
                print()
        
        elif args.stats:
            # Mode statistiques
            stats = vectorizer.get_database_stats()
            print("📊 Statistiques de la base vectorielle:")
            print(f"  Chunks indexés: {stats['total_chunks']}")
            print(f"  Collection: {stats['collection_name']}")
        
        else:
            # Mode vectorisation
            print("🚀 Démarrage de la vectorisation...")
            stats = vectorizer.vectorize_project()
            print(f"\n✅ Vectorisation terminée!")
            print(f"📁 Fichiers traités: {stats['processed_files']}/{stats['total_files']}")
            print(f"🪓 Chunks créés: {stats['total_chunks']}")
            print(f"📊 Types de chunks: {stats['chunks_by_type']}")
    
    except Exception as e:
        logging.error(f"Erreur: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()