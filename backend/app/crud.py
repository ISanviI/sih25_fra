from sqlalchemy.orm import Session
from . import models
from database import schemas
from geoalchemy2.functions import ST_Contains, ST_MakePoint

def get_region(db: Session, region_id: int):
    return db.query(models.Region).filter(models.Region.id == region_id).first()

def get_regions(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Region).offset(skip).limit(limit).all()

def create_region(db: Session, region: schemas.RegionCreate, owner_id: int):
    db_region = models.Region(**region.dict(), owner_id=owner_id)
    db.add(db_region)
    db.commit()
    db.refresh(db_region)
    return db_region

def get_user(db: Session, user_id: int):
    return db.query(models.User).filter(models.User.id == user_id).first()

def get_users(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.User).offset(skip).limit(limit).all()

def create_user(db: Session, user: schemas.UserCreate):
    db_user = models.User(**user.dict())
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def get_region_at_point(db: Session, lat: float, lon: float):
    return db.query(models.Region).filter(ST_Contains(models.Region.geom, ST_MakePoint(lon, lat))).first()
