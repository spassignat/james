import logging
from typing import Dict, Any

from parsers.analyzer import Analyzer

logger = logging.getLogger(__name__)


class GenericAnalyzer(Analyzer):
    def analyze(self, file_path: str) -> Dict[str, Any]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return self.analyze_content(content, file_path)
        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {e}")
            return {
                'file_type': 'generic',
                'file_path': file_path,
                'error': str(e)
            }

    def analyze_content(self, content: str, file_path: str) -> Dict[str, Any]:
        return {
            'file_type': 'generic',
            'file_path': file_path,
            'line_count': len(content.splitlines()),
            'size_bytes': len(content),
            'content': content,
            'analysis': {'content_preview': content[:500]}
        }
