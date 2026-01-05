# src/chunk_strategies.py
import re
from typing import List, Dict, Any, Optional
import logging

from file.file_info import FileInfo
from parsers.analysis_result import AnalysisResult
from vector.chunk.generic_chunk import GenericChunkStrategy
from vector.chunk.java_chunk import JavaChunkStrategy
from vector.chunk.javascript_chunk import JavaScriptChunkStrategy
from vector.chunk.vuejs_chunk import VueJSChunkStrategy

logger = logging.getLogger(__name__)

class ChunkStrategyRegistry:
    def __init__(self, vectorization_config: Dict[str, Any]):
        self.strategies = {}
        self.default_chunk_config = vectorization_config.get('chunk_strategies', {}).get('default', {})
        self._initialize_strategies(vectorization_config)
    
    def _initialize_strategies(self, config: Dict[str, Any]):
        """Initialise les stratégies de chunking par extension"""
        chunk_configs = config.get('chunk_strategies', {})
        
        # Stratégies spécifiques
        self.strategies['.java'] = JavaChunkStrategy(
            **chunk_configs.get('java', self.default_chunk_config)
        )
        self.strategies['.vue'] = VueJSChunkStrategy(
            **chunk_configs.get('vue', self.default_chunk_config)
        )
        self.strategies['.js'] = JavaScriptChunkStrategy(
            **chunk_configs.get('js', self.default_chunk_config)
        )
        self.strategies['.ts'] = JavaScriptChunkStrategy(
            **chunk_configs.get('ts', self.default_chunk_config)
        )
        
        # Stratégie générique par défaut
        self.default_strategy = GenericChunkStrategy(**self.default_chunk_config)
    
    def create_chunks(self, file_extension: str, analysis: AnalysisResult, file_info: FileInfo) -> List[Dict[str, Any]]:
        """Crée des chunks pour un fichier en utilisant la stratégie appropriée"""
        strategy = self.strategies.get(file_extension, self.default_strategy)
        return strategy.create_chunks(analysis, file_info)