
from services.base_embedding_service import BaseEmbeddingService
from services.embedding_service import GithubCohereEmbeddingService


class EmbeddingFactory:

    @staticmethod
    def create_client(endpoint:str, model_name:str, token:str) -> BaseEmbeddingService:
        return GithubCohereEmbeddingService(endpoint=endpoint,
                                           model_name=model_name,
                                           token=token)