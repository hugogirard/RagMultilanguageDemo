from dotenv import load_dotenv
import os

class Config:

    def __init__(self):        
       load_dotenv(override=True)

    @property
    def search_endpoint(self) -> str:
       return os.getenv('SEARCH_ENDPOINT')
    
    @property
    def search_key(self) -> str:
       return os.getenv('SEARCH_API_KEY')