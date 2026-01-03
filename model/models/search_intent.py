from dataclasses import dataclass
from typing import List, Literal


@dataclass
class SearchIntent:
    goal: str
    domain: Literal[
        "architecture",
        "analysis",
        "vector",
        "generation",
        "refactoring",
        "documentation",
        "security",
        "performance"
    ]
    focus: List[str]
    depth: Literal["low", "medium", "high"]
