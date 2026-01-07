import logging

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class TransformerFactory:
    def __init__(self):
        pass

    @staticmethod
    def create_transformer(config) -> SentenceTransformer:
        """Initialise le modèle embedding avec gestion des erreurs"""
        vconfig = config.get('vectorization', {})
        model_name = vconfig.get('model_name', 'all-MiniLM-L6-v2')
        try:
            model = SentenceTransformer(model_name)
            logger.info(
                f"Modèle embedding chargé: {model_name} (dimension: {model.get_sentence_embedding_dimension()})")
            return model
        except Exception as e:
            logger.error(f"Erreur chargement modèle {model_name}: {e}")
            logger.info("Tentative avec modèle de fallback...")
            try:
                # Fallback vers un modèle plus léger
                model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
                logger.info(f"Modèle de fallback chargé: paraphrase-MiniLM-L3-v2")
                return model
            except Exception as fallback_error:
                logger.error(f"Erreur chargement modèle de fallback: {fallback_error}")
                raise RuntimeError(f"Impossible de charger un modèle embedding: {e}")
