from pydantic import BaseModel
from typing import List, Optional

class RegionBase(BaseModel):
    name: str

class RegionCreate(RegionBase):
    pass

class Region(RegionBase):
    id: int
    owner_id: int

    class Config:
        from_attributes = True

class UserBase(BaseModel):
    patta_id: str
    family_details: str
    spouse_details: str
    taluka: str
    city: str
    village: str
    state: str

class UserCreate(UserBase):
    pass

class User(UserBase):
    id: int
    regions: List[Region] = []

    class Config:
        from_attributes = True
