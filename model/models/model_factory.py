from typing import Dict, Any

from models.code_chunk import CodeChunk
from models.search_intent import SearchIntent
from util.Util import infer_language_from_path, infer_category_from_type


class ModelFactory:

    @staticmethod
    def create_code_chunk(raw_chunk: Dict[str, Any]) -> CodeChunk:
        metadata = raw_chunk.get("metadata", {})
        return CodeChunk(
            content=raw_chunk.get("content", ""),
            file_path=metadata.get("file_path", ""),
            filename=metadata.get("filename", ""),
            language=infer_language_from_path(metadata.get("file_path", "")),
            category=infer_category_from_type(metadata.get("chunk_type", ""), metadata.get("file_path", "")),
            chunk_type=metadata.get("chunk_type", "unknown"),
        )

    @staticmethod
    def build_query(intent: SearchIntent) -> str:
        return f"""
        Goal: {intent.goal}
        Domains: {intent.domain}
        Focus: {', '.join(intent.focus)}
        Depth: {intent.depth}
        """
