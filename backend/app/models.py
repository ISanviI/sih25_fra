from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship
from geoalchemy2 import Geometry
from .database import Base

class Region(Base):
    __tablename__ = "regions"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    geom = Column(Geometry)

    owner_id = Column(Integer, ForeignKey("users.id"))
    owner = relationship("User", back_populates="regions")

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    patta_id = Column(String, unique=True, index=True)
    family_details = Column(String)
    spouse_details = Column(String)
    taluka = Column(String)
    city = Column(String)
    village = Column(String)
    state = Column(String)

    regions = relationship("Region", back_populates="owner")
