from dataclasses import dataclass
from typing import List


@dataclass
class AgentResult:
    agent_name: str
    summary: str
    recommendations: List[str]
    raw_output: str
