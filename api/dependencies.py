from fastapi import Request, HTTPException
from services import (
    IndexingService
)

def get_indexing_service(request:Request) -> IndexingService:
    return request.app.state.indexing_service