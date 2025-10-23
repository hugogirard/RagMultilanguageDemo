from dotenv import load_dotenv
from azure.search.documents.indexes.aio import SearchIndexClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes.models import (
    SearchField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SearchIndex,    
    SearchFieldDataType
)
from config import Config
from infrastructure.enum import IndexNameType
from typing import List
from request import IndexCreationResult
import os

class IndexingService:
    
    def __init__(self,config:Config):
        self.index_client = SearchIndexClient(
                                endpoint=config.search_endpoint,
                                credential=AzureKeyCredential(config.search_key)
                            )
        
    async def create_indexes(self) -> List[IndexCreationResult]:

        index_creation_results:List[IndexCreationResult] = []

        vector_search = VectorSearch(  
            algorithms=[  
                HnswAlgorithmConfiguration(name="myHnsw"),
            ],  
            profiles=[  
                VectorSearchProfile(  
                    name="vector-profile",  
                    algorithm_configuration_name="myHnsw"
                )
            ]
        )       

        result = await self._create_index_multi_languages(vector_search)

        index_creation_results.append(result)

        return index_creation_results
    
    async def delete_indexes(self):

        indexes = [IndexNameType.MULTI_LANGUAGE]

        for index in indexes:
            await self.index_client.delete_index(index)

    async def _create_index_multi_languages(self,vector_search:VectorSearch) -> IndexCreationResult:

        index_creation_result = IndexCreationResult(indexName=IndexNameType.MULTI_LANGUAGE,
                                                    isSuccess=True,
                                                    error=None)

        try:
            fields = [
                SearchField(name="id", type=SearchFieldDataType.String,key=True),   
                SearchField(name="original_language", type=SearchFieldDataType.String, searchable=False,sortable=False, facetable=False, filterable=False),            
                SearchField(name="brand", type=SearchFieldDataType.String, searchable=True,sortable=False, facetable=False, filterable=False),                  
                SearchField(name="model", type=SearchFieldDataType.String, searchable=True,sortable=False, facetable=True, filterable=True),              
                SearchField(name="fault", type=SearchFieldDataType.String, searchable=True,sortable=False, facetable=False, filterable=False),            
                SearchField(name="vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single), vector_search_dimensions=1024, vector_search_profile_name="vector-profile",searchable=True,sortable=False, facetable=False, filterable=False),
                SearchField(name="fix", type=SearchFieldDataType.String, searchable=True,sortable=False, facetable=False, filterable=False)    
            ]   

            index = SearchIndex(name=IndexNameType.MULTI_LANGUAGE, fields=fields, vector_search=vector_search)
            await self.index_client.create_or_update_index(index)     
        except Exception as ex:
            index_creation_result.is_success = False
            index_creation_result.error = str(ex)
        finally:
            return index_creation_result
            

   
        