from dataclasses import dataclass

@dataclass
class Resolution:
    id: str
    score: float
    brand: str
    model:str
    fault:str
    fix:str