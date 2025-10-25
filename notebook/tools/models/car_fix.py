from dataclasses import dataclass


@dataclass
class CarFix:
    id: str
    score: str
    brand:str
    model:str
    fault:str
    fix:str