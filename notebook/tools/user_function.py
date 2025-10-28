from openai import AsyncAzureOpenAI
from azure.core.credentials import AzureKeyCredential
from tools.models import CarFix
from azure.search.documents import SearchClient
from azure.search.documents import SearchItemPaged
from azure.search.documents.models import VectorizedQuery
from azure.identity import DefaultAzureCredential
from azure.ai.inference import EmbeddingsClient
from azure.ai.agents.models import (
    ToolSet,
    FunctionTool,
    MessageRole    
)
from azure.ai.projects import AIProjectClient
from typing import Dict, List
from dotenv import load_dotenv
from openai import AzureOpenAI
import json
import os

load_dotenv(override=True)

open_ai_endpoint = os.getenv('OPENAI_ENDPOINT')
open_ai_key = os.getenv('OPENAI_KEY')
open_ai_embedding_model = os.getenv('EMBEDDING_OPENAI_DEPLOYMENT')

cohere_key = os.getenv('COHERE_KEY')
cohere_model=os.getenv('COHERE_MODEL')
cohere_endpoint=os.getenv('COHERE_ENDPOINT')

# Search
search_endpoint = os.getenv('SEARCH_ENDPOINT')
search_api_key = os.getenv('SEARCH_API_KEY')

project_endpoint = os.getenv('AI_FOUNDRY_PROJECT_ENDPOINT')

project = AIProjectClient(
    endpoint=project_endpoint,
    credential=DefaultAzureCredential()    
)

search_client_english = SearchClient(
    endpoint=search_endpoint,
    credential=AzureKeyCredential(search_api_key),
    index_name="translated"
)

search_client_asked_language = SearchClient(
    endpoint=search_endpoint,
    credential=AzureKeyCredential(search_api_key),
    index_name="multilanguage"
)

search_client_hybrid = SearchClient(
    endpoint=search_endpoint,
    credential=AzureKeyCredential(search_api_key),
    index_name="translated_dual"    
)

client = AzureOpenAI(
    azure_endpoint=open_ai_endpoint,
    api_key=open_ai_key,
    api_version="2024-12-01-preview"
)

cohere_client = EmbeddingsClient(endpoint=cohere_endpoint,
                                 credential=AzureKeyCredential(cohere_key))

def format_result(results: SearchItemPaged[Dict]) -> List[CarFix]:
    """
    Converts Azure Search results into a list of CarFix objects.
    Iterates through search results and extracts relevant fields.
    
    Args:
        results: Paginated search results from Azure Search
        
    Returns:
        List of CarFix objects with structured troubleshooting data
    """
    documents:List[CarFix] = []
    
    for result in results:
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

def _do_search(brand:str,model:str,fault:str,embedding:List[float],search_client:SearchClient) -> str:
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
        
        #query = f"{object_description.lower()}~ {object_type_description.lower()}~"

        if len(query) == 0:
            query=fault

        vector_query = VectorizedQuery(vector=embedding, k_nearest_neighbors=50, fields="vector")

        results = search_client.search(  
            search_text=query,  
            search_fields=['brand','model'],
            vector_queries= [vector_query],
            top=5
        )  
        
        documents = format_result(results)

        # Convert to JSON string for tool response
        response_data = []
        for item in documents:
            response_data.append({
                "id": item.id,
                "score": item.score,
                "brand": item.brand,
                "model": item.model,
                "fault": item.fault,
                "fix": item.fix
            })
            
        return json.dumps(response_data, indent=2)
    except Exception as ex:
        return ex        

def get_resolution_asked_language(brand: str, model:str, fault:str) -> str:
    """
    Searches for troubleshooting resolutions in the user's requested language.
    Uses Cohere embeddings for multilanguage support and searches the multilanguage index.

    Args:
        brand: Brand of the car
        model: Model of the car
        fault: Description of the problem/fault
        
    Returns:
        JSON string with troubleshooting documents in the requested language
    """

    try:

        response = cohere_client.embed(input=[fault],model=cohere_model)
        embedding = response.data[0]['embedding']

        return _do_search(brand,model,fault,embedding,search_client_asked_language)
        
    except Exception as ex:
        return ex    


def get_resolution_hybrid(default_language_english:bool,
                          brand:str, 
                          model:str,
                          brand_english:str,
                          model_english:str, 
                          fault:str, 
                          fault_english:str,) -> str:
    """.
    Performs search in both the original language and English translation, combining        

    Args:
        default_language_english: Whether the default language is English (affects vectorization strategy)
        brand: Car brand in the original language
        model: Car model in the original language
        brand_english: Car brand translated to English (only if original language is not english)
        model_english: Car model translated to English (only if original language is not english)
        fault: Fault description in the original language
        fault_english: Fault description translated to English (only if original language is not english)
        
    Returns:
        JSON string with troubleshooting documents in the requested language and
        english if the original language was not in english
    """    
    try:

        # Fuzzy search
        query = ""
        if brand:
            query = f"{brand.lower()}~"

        if brand_english:
            query = query + f"{ brand_english.lower()}~"

        if model:
            query = query + f"{ model.lower()}~"

        if model_english:
            query = query + f"{ model_english.lower()}~"            
        
        #query = f"{object_description.lower()}~ {object_type_description.lower()}~"

        if len(query) == 0:
            query=fault
            if not default_language_english:
                query = query + f"{ fault_english}~"  


        texts_to_vectorize:List[str] = []

        # Always add the original text
        texts_to_vectorize.append(fault)

        # Validate if the default language is english
        if not default_language_english:
            texts_to_vectorize.append(fault_english)
        
        response = cohere_client.embed(input=texts_to_vectorize,model=cohere_model)

        vector_queries:List[VectorizedQuery] = []

        vector_queries.append(VectorizedQuery(vector=response.data[0]['embedding'], k_nearest_neighbors=50, fields="vector"))
        
        if not default_language_english:
            vector_queries.append(VectorizedQuery(vector=response.data[1]['embedding'], k_nearest_neighbors=50, fields="vector_en"))

        results = search_client_hybrid.search(  
            search_text=query,  
            search_fields=['brand','model','brand_en','model_en'],
            vector_queries= vector_queries,
            top=5
        )  
        
        response_data = []
        for result in results:
            doc = {
                'id':result['id'],
                'score':result['@search.score'],
                'brand':result['brand'],
                'brand_en':result['brand_en'],
                'model':result['model'],
                'model_en':result['model_en'],
                'fault':result['fault'],
                'fault_en':result['fault_en'],
                'fix':result['fix'],
                'fix_en':result['fix_en']
            }
            response_data.append(doc)     
        
        return json.dumps(response_data, indent=2)

    except Exception as ex:
        return ex

def get_resolution_english(brand:str, model:str, fault:str) -> str:
    """
    Searches for troubleshooting resolutions in English.
    Uses Azure OpenAI embeddings and searches the English translated index.

    Args:
        brand: Brand of the car
        model: Model of the car
        fault: Description of the problem/fault
        
    Returns:
        JSON string with English troubleshooting documents
    """

    try:
        # Vectorize the vault
        query = fault
        response = client.embeddings.create(input=query, model=open_ai_embedding_model, dimensions=1536)
        embedding = response.data[0].embedding 

        # Fuzzy search
        query = ""
        if brand:
            query = f"{ brand.lower()}~"

        if model:
            query = query + f"{ model.lower()}~"
        
        #query = f"{object_description.lower()}~ {object_type_description.lower()}~"

        if len(query) == 0:
            query=fault

        vector_query = VectorizedQuery(vector=embedding, k_nearest_neighbors=50, fields="vector")

        results = search_client_english.search(  
            search_text=query,  
            search_fields=['brand','model'],
            vector_queries= [vector_query],
            top=5
        )  
        
        documents = format_result(results)

        # Convert to JSON string for tool response
        response_data = []
        for item in documents:
            response_data.append({
                "id": item.id,
                "score": item.score,
                "brand": item.brand,
                "model": item.model,
                "fault": item.fault,
                "fix": item.fix
            })
            
        return json.dumps(response_data, indent=2)
    except Exception as ex:
        return ex