from pydantic import BaseModel, Field
from typing import Optional

class TroubleInformation(BaseModel):
    brand: Optional[str] = Field(None, description="The brand of the car")
    model: Optional[str] = Field(None, description="The model of the car")
    fault: str = Field(None, description="The fault problem faced")
