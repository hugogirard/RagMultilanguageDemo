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

client = AzureOpenAI(
    azure_endpoint=open_ai_endpoint,
    api_key=open_ai_key,
    api_version="2024-12-01-preview"
)

cohere_client = EmbeddingsClient(endpoint=cohere_endpoint,
                                 credential=AzureKeyCredential(cohere_key))

def format_result(results: SearchItemPaged[Dict]) -> List[CarFix]:
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
    try:
        # Vectorize the vault
        query = fault

        # Fuzzy search
        query = ""
        if brand:
            query = f"{brand.lower()}~"

        if model:
            query = query + f"{model.lower()}~"
        
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
    This function searches for troubleshooting resolutions based on brand, model, and fault.

    Args:
        brand: Brand of the car
        model: Model of the car
        fault: Description of the problem/fault
        
    Returns:
        TroubleShooting containing car troubleshooting documents
    """

    try:

        response = cohere_client.embed(input=[fault],model=cohere_model)
        embedding = response.data[0]['embedding']

        return _do_search(brand,model,fault,embedding,search_client_asked_language)
        
    except Exception as ex:
        return ex    


def get_resolution_english(brand:str, model:str, fault:str) -> str:
    """
    This function searches for troubleshooting resolutions based on brand, model, and fault.

    Args:
        brand: Brand of the car
        model: Model of the car
        fault: Description of the problem/fault
        
    Returns:
        TroubleShooting containing car troubleshooting documents
    """

    try:
        # Vectorize the vault
        query = fault
        response = client.embeddings.create(input=query, model=open_ai_embedding_model, dimensions=1536)
        embedding = response.data[0].embedding 

        # Fuzzy search
        query = ""
        if brand:
            query = f"{brand.lower()}~"

        if model:
            query = query + f"{model.lower()}~"
        
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