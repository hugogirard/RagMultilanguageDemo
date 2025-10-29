from dotenv import load_dotenv
import os

load_dotenv(override=True)

class Config:

    @staticmethod
    def search_endpoint() -> str:
        return os.getenv('SEARCH_ENDPOINT')

    @staticmethod
    def search_api_key() -> str:
        return os.getenv('SEARCH_API_KEY')
    
    @staticmethod
    def search_index_name() -> str:
        return os.getenv("SEARCH_INDEX_NAME")
    
    @staticmethod
    def openai_endpoint() -> str:
        return os.getenv('OPENAI_ENDPOINT')
    
    @staticmethod
    def openai_key() -> str:
        return os.getenv('OPENAI_KEY')
    
    @staticmethod
    def openai_embedding_model() -> str:
        return os.getenv('EMBEDDING_OPENAI_DEPLOYMENT')
