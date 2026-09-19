from pydantic import BaseModel


class RetrievedDocument(BaseModel):
    score: float
    text: str
    metadata: dict | None = None
