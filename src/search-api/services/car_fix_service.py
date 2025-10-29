from azure.core.credentials import AzureKeyCredential
from azure.search.documents.aio import SearchClient
from azure.search.documents.models import VectorizedQuery
from openai import AsyncAzureOpenAI
from config import Config
from models import CarFix
from typing import List

class CarFixService:
    def __init__(self):
        self.search_client = SearchClient(
            endpoint=Config.search_endpoint(),
            credential=AzureKeyCredential(Config.search_api_key()),
            index_name=Config.search_index_name()
        )

        self.openai_client = AsyncAzureOpenAI(
            azure_endpoint=Config.openai_endpoint(),
            api_key=Config.openai_key(),
            api_version="2024-12-01-preview"
        )

    async def get_car_fix(self,brand:str, model:str, fault:str) -> List[CarFix]:
        response = await self.openai_client.embeddings.create(input=fault, 
                                                              model=Config.openai_embedding_model(), 
                                                              dimensions=1536)   
        embedding = response.data[0].embedding
        return await self._do_search(brand,model,fault,embedding)


    async def _do_search(self,brand:str,model:str,fault:str,embedding:List[float]) -> List[CarFix]:
        """
        Performs a hybrid search combining vector similarity and fuzzy text matching.
        Builds a fuzzy query from brand/model, executes vector search on embeddings,
        and returns top 5 results as JSON.
        
        Args:
            brand: Car brand for filtering
            model: Car model for filtering
            fault: Fault description (used as fallback query)
            embedding: Vector embedding for semantic search
            search_client: Azure Search client instance
            
        Returns:
            JSON string containing top 5 matching documents with scores
        """
        try:
            # Vectorize the vault
            query = fault

            # Fuzzy search
            query = ""
            if brand:
                query = f" {brand.lower()}~"

            if model:
                query = query + f" {model.lower()}~"
            
            if len(query) == 0:
                query=fault

            vector_query = VectorizedQuery(vector=embedding, k_nearest_neighbors=50, fields="vector")

            results = await self.search_client.search(  
                search_text=query,  
                search_fields=['brand','model'],
                vector_queries= [vector_query],
                top=5
            )  
            
            documents:List[CarFix] = []
            
            async for result in results:
                document = CarFix(
                    id=result['id'],
                    score=result['@search.score'],
                    brand=result['brand'],
                    model=result['model'],
                    fault=result['fault'],
                    fix=result['fix']
                )
                documents.append(document)     
            
            return documents


        except Exception as ex:
            return ex           
