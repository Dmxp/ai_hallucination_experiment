from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Question:
    id: str
    category: str
    question: str
    expected: Optional[str]


@dataclass
class Claim:
    text: str
    status: str
    evidence: list[str] = field(default_factory=list)


@dataclass
class EvaluationResult:
    question_id: str
    mode: str
    answer: str
    classification: str
    claims: list[Claim] = field(default_factory=list)


@dataclass
class ExperimentResult:
    question_id: str
    mode: str
    question: str
    answer: str
    evaluation: Optional[EvaluationResult] = None