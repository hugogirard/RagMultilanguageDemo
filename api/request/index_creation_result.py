from pydantic import BaseModel,Field
from typing import Optional

class IndexCreationResult(BaseModel):
    index_name:str = Field(None,alias='indexName',description='The name of the index created')
    is_success: bool = Field(True,alias='isSuccess',description='If the index was created successfully')
    error: Optional[str] = Field(None,description='Error message during creation')