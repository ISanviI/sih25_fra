from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from sqlalchemy.orm import Session
from geoalchemy2.shape import to_shape
import ee
import rasterio
from rasterio.windows import Window
import numpy as np
import os
import time
import torch
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from . import crud, models, ee_processing
from database import schemas
from .database import SessionLocal, engine
import io
import mercantile
from PIL import Image
import rasterio.warp
from shapely.geometry import box

# models.Base.metadata.create_all(bind=engine) # This is handled by Alembic

app = FastAPI()

origins = [
    "http://localhost:5173",
    "http://localhost:5174",
    "http://localhost:5175",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

processed_regions = {}

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/ee-tile-url")
def get_ee_tile_url():
    try:
        # Initialize Earth Engine if not already initialized
        try:
            ee.Initialize(ee.ServiceAccountCredentials(os.getenv("EE_SERVICE_ACCOUNT_EMAIL"), os.getenv("EE_KEY_FILE")))
        except Exception:
            ee.Authenticate()
            ee.Initialize()

        # Get a recent, cloud-free Sentinel-2 image.
        image = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterDate('2024-01-01', '2024-03-31') \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10)) \
            .median()

        # Visualization parameters for a true-color image.
        vis_params = {
            'bands': ['B4', 'B3', 'B2'],
            'min': 0,
            'max': 3000,
            'gamma': 1.4,
        }

        map_id = image.getMapId(vis_params)
        tile_url = map_id['tile_fetcher'].url_format
        return {"tile_url": tile_url}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating EE tile URL: {e}")

@app.post("/process-region")
def process_region(lat: float, lon: float):
    """Processes a circular region around the given lat/lon."""
    try:
        # Pass lat/lon directly to the processing function
        result = ee_processing.process_geometry(lat=lat, lon=lon)
        
        # Use a static key for the processed image path, since we don't have a region ID
        processed_regions["last_click"] = result["stitched_image_path"]
        
        # The frontend expects a "region_id" to build the tile URL. We'll return our static key.
        return {"region_id": "last_click", "stats": result["stats"]}
    except Exception as e:
        import traceback
        print("--- DETAILED ERROR IN /process-region ---")
        traceback.print_exc()
        print("-----------------------------------------")
        raise HTTPException(status_code=500, detail=f"Error processing region: {e}")

@app.get("/tiles/{region_id}/{z}/{x}/{y}")
async def get_tile(region_id: str, z: int, x: int, y: int):
    if region_id not in processed_regions:
        raise HTTPException(status_code=404, detail="Processed region not found.")

    stitched_image_path = processed_regions[region_id]
    if not os.path.exists(stitched_image_path):
        raise HTTPException(status_code=404, detail="Stitched image not found.")

    try:
        with rasterio.open(stitched_image_path) as src:
            # Get the tile bounds directly in WGS84 (EPSG:4326), which matches the source raster
            wgs_bbox = mercantile.bounds(x, y, z)
            
            # Create a window from the WGS84 bounds
            window = src.window(wgs_bbox.west, wgs_bbox.south, wgs_bbox.east, wgs_bbox.north)
            
            # Read the data in the window, ensuring output is 256x256
            data = src.read(window=window, boundless=True, out_shape=(256, 256))

            # Convert data to a colored paletted image
            img_data = data[0] # Assuming single band
            img = Image.fromarray(img_data.astype('uint8'), 'P') # 'P' for paletted

            # Define a color palette for the 6 classes
            # [Background, Forest, Agriculture, Water, Built-up, Degraded Forest]
            palette = [
                0, 0, 0,        # 0: Black (Transparent - will be set below)
                34, 139, 34,    # 1: ForestGreen
                154, 205, 50,   # 2: YellowGreen
                30, 144, 255,   # 3: DodgerBlue
                128, 128, 128,  # 4: Gray
                210, 180, 140,  # 5: Tan
            ]
            # Fill the rest of the 256-color palette
            palette.extend([0] * (256 * 3 - len(palette)))
            img.putpalette(palette)

            # Save image to a byte stream with transparency for the background
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG', transparency=0)
            img_byte_arr.seek(0)

            return StreamingResponse(img_byte_arr, media_type="image/png")

    except Exception as e:
        # Log the error for debugging
        print(f"Error generating tile for {region_id} at {z}/{x}/{y}: {e}")
        return HTTPException(status_code=500, detail=f"Error generating tile: {e}")

@app.get("/regions/")
def read_regions(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    regions = crud.get_regions(db, skip=skip, limit=limit)
    return regions

@app.get("/users/")
def read_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    users = crud.get_users(db, skip=skip, limit=limit)
    return users

@app.get("/regions/{region_id}")
def read_region(region_id: int, db: Session = Depends(get_db)):
    db_region = crud.get_region(db, region_id=region_id)
    if db_region is None:
        raise HTTPException(status_code=404, detail="Region not found")
    return db_region

@app.get("/users/{user_id}")
def read_user(user_id: int, db: Session = Depends(get_db)):
    db_user = crud.get_user(db, user_id=user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user

@app.get("/api/regions.geojson")
def read_regions_geojson(db: Session = Depends(get_db)):
    regions = crud.get_regions(db)
    features = []
    for region in regions:
        shape = to_shape(region.geom)
        feature = {
            "type": "Feature",
            "geometry": shape.__geo_interface__,
            "properties": {
                "id": region.id,
                "name": region.name,
                "owner": region.owner.patta_id if region.owner else None
            }
        }
        features.append(feature)

    return JSONResponse(content={
        "type": "FeatureCollection",
        "features": features
    })

@app.post("/ee/process")
def ee_process(geometry: dict, db: Session = Depends(get_db)):
    try:
        result = ee_processing.process_geometry(geometry)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/regions/at_point")
def get_region_at_point(lat: float, lon: float, db: Session = Depends(get_db)):
    region = crud.get_region_at_point(db, lat=lat, lon=lon)
    if region is None:
        raise HTTPException(status_code=404, detail="Region not found")
    return region

@app.get("/images/{image_name}")
async def get_image(image_name: str):
    return FileResponse(f"./{image_name}")
