from pydantic import BaseModel


class ChatPayloadSchema(BaseModel):
    question: str
