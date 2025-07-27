import uuid
from enum import Enum

from pydantic import BaseModel, Field, validator


class Reason(str, Enum):
    UNEXPECTED_BEHAVIOR = "unexpected_behavior"
    UI_ERROR = "ui_error"
    WRONG_VALUES = "wrong_values"
    INCOME_NOT_VERIFIABLE = "income_not_verifiable"
    PROCESS_TOO_COMPLEX = "process_too_complex"
    WAITING_PERIOD_TOO_LONG = "waiting_period_too_long"
    FEES_UNEXPECTED = "fees_unexpected"
    TECHNICAL_ISSUE = "technical_issue"
    CHANGED_MIND = "changed_mind"
    OTHER = "other"


class Step(str, Enum):
    PRESENT_OPTIONS = "Present Options"
    BORROWER_SELECTION = "Borrower Selection"
    CREDIT_CHECK = "Credit Check"
    VERIFY_LIABILITIES = "Verify Liabilities"
    COLLECT_DECLARATIONS = "Collect Declarations"
    FINALIZE_APPLICATION = "Finalize Application"
    PREPARE_DISCLOSURE = "Prepare Disclosure"
    ISSUE_DISCLOSURE = "Issue Disclosure"


class Example(BaseModel):
    """Single training example with a generated UUID."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    reason: Reason
    free_text: str
    step: Step

    @validator("free_text")
    def text_length(cls, v: str) -> str:  # noqa: N805
        if len(v) < 30:
            raise ValueError("free_text too short – likely low-quality output")
        return v


class StructuredExample(BaseModel):
    """Used only for OpenAI response-schema validation."""

    reason: Reason
    free_text: str
    step: Step
