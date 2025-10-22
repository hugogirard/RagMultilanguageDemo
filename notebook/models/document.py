from pydantic import BaseModel, Field
from typing import List, Optional

class Document(BaseModel):
    """Model representing a document with id, language code, and text content."""
    id: str = Field(..., description="Unique identifier for the document")
    language_code: str = Field(..., description="ISO language code (e.g., 'en', 'fr', 'ja')")
    text: str = Field(..., description="The text content of the document")
    vector: Optional[List[float]] = Field(None,description="The vector of the text")
