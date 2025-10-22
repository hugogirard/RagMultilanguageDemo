from services.base_embedding_service import BaseEmbeddingService
from azure.ai.inference.aio import EmbeddingsClient
from azure.core.credentials import AzureKeyCredential
from typing import Dict, List
from models.document import Document

class GithubCohereEmbeddingService(BaseEmbeddingService):
    
    def __init__(self,endpoint:str, model_name:str, token:str):
        self.client = EmbeddingsClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(token)
        )
        self.model_name = model_name
        
    async def embed_documents(self, documents: List[Document]) -> List[Document]:        
        """Implement the abstract method to embed documents"""

        # We will send in this case 10 documents at the times
        idx = -1        
        number_of_documents = len(documents) - 1
        documents_to_embed:List[str] = []
        
        while idx <= number_of_documents:
                    
            idx+=1                        
            documents_to_embed.append(documents[0].text)

            if idx % 10 == 0:
                vectors = self._create_embeddings(documents_to_embed)
                
                documents_to_embed.clear()

        # doc = documents[0]

        # response = await self.client.embed(input=[doc.text],
        #                                    model=self.model_name)
        
        # doc.vector = response.data[0]['embedding']

        #return doc

    async def _create_embeddings(self,documents:List[str]) -> List[float]:
        """Call Azure AI Inference endpoint using Github Model Cohere 3"""
        
        vectors:List[float] = []
        response = await self.client.embed(input=documents,
                                           model=self.model_name)
        
        for data in response.data:
            vectors.append(data['embedding'])

        return vectors
        

