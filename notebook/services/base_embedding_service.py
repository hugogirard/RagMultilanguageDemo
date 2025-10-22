from abc import ABC, abstractmethod
from typing import Dict, List
from models.document import Document

class BaseEmbeddingService(ABC):
    
    @abstractmethod
    async def embed_documents(self, documents: List[Document]) -> List[Document]:
        pass


