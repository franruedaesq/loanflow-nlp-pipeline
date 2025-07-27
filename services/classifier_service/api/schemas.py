from pydantic import BaseModel


class TextRequest(BaseModel):
    text: str


class Prediction(BaseModel):
    label: str | None
    score: float
    candidates: list[tuple[str, float]] | None = None
