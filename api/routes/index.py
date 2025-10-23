from fastapi import APIRouter, Depends, HTTPException
from typing import Annotated
from services import IndexingService
from dependencies import get_indexing_service
from request import IndexCreationResult
from typing import List

router = APIRouter(
    prefix="/index"
)

@router.post('/',description='Create all indexes in Azure AI Search')
async def create_index(index_service: Annotated[IndexingService, Depends(get_indexing_service)]) -> List[IndexCreationResult]:
    try:
        return await index_service.create_indexes()
    except Exception as err:        
        raise HTTPException(status_code=500, detail='Internal Server Error') 

@router.delete('/',description='Delete all indexes in Azure AI Search')
async def create_index(index_service: Annotated[IndexingService, Depends(get_indexing_service)]) -> List[IndexCreationResult]:
    try:
        return await index_service.delete_indexes()
    except Exception as err:        
        raise HTTPException(status_code=500, detail='Internal Server Error')    