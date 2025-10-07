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
            # Get tile bounds in WGS84
            wgs_bounds = mercantile.bounds(x, y, z)
            
            # Get the bounds of the source raster
            src_bounds = src.bounds
            
            print(f"Tile {z}/{x}/{y} bounds in WGS84: {wgs_bounds}")
            print(f"Source raster bounds: {src_bounds}")
            print(f"Source raster CRS: {src.crs}")
            print(f"Source raster shape: {src.shape}")
            
            # Transform tile bounds from WGS84 to the source CRS
            if src.crs.to_string() != 'EPSG:4326':
                from rasterio.warp import transform_bounds
                tile_bounds_in_src_crs = transform_bounds(
                    'EPSG:4326', 
                    src.crs,
                    wgs_bounds.west, 
                    wgs_bounds.south, 
                    wgs_bounds.east, 
                    wgs_bounds.north
                )
                print(f"Tile bounds in source CRS: {tile_bounds_in_src_crs}")
            else:
                tile_bounds_in_src_crs = (
                    wgs_bounds.west, 
                    wgs_bounds.south, 
                    wgs_bounds.east, 
                    wgs_bounds.north
                )
            
            # Check if tile bounds intersect with source bounds
            if (tile_bounds_in_src_crs[2] < src_bounds.left or 
                tile_bounds_in_src_crs[0] > src_bounds.right or 
                tile_bounds_in_src_crs[3] < src_bounds.bottom or 
                tile_bounds_in_src_crs[1] > src_bounds.top):
                print(f"Tile {z}/{x}/{y} does not intersect with source raster")
                # Return empty transparent tile
                img = Image.new('RGBA', (256, 256), (0, 0, 0, 0))
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='PNG')
                img_byte_arr.seek(0)
                return StreamingResponse(img_byte_arr, media_type="image/png")
            
            # Create window from bounds
            window = rasterio.windows.from_bounds(
                tile_bounds_in_src_crs[0],  # left
                tile_bounds_in_src_crs[1],  # bottom
                tile_bounds_in_src_crs[2],  # right
                tile_bounds_in_src_crs[3],  # top
                src.transform
            )
            
            print(f"Window: {window}")
            
            # Read the data
            data = src.read(
                1,
                window=window,
                out_shape=(256, 256),
                boundless=True,
                fill_value=0
            )
            
            print(f"Data shape: {data.shape}, dtype: {data.dtype}")
            print(f"Data range: min={data.min()}, max={data.max()}")
            print(f"Unique values: {np.unique(data)}")
            
            # Create colored image
            img_array = np.zeros((256, 256, 4), dtype=np.uint8)
            
            # Define colors for each class (RGBA)
            colors = {
                0: [0, 0, 0, 0],          # Background - Transparent
                1: [34, 139, 34, 255],     # Forest - ForestGreen
                2: [154, 205, 50, 255],    # Agriculture - YellowGreen
                3: [30, 144, 255, 255],    # Water - DodgerBlue
                4: [128, 128, 128, 255],   # Built-up - Gray
                5: [210, 180, 140, 255],   # Degraded Forest - Tan
            }
            
            # Apply colors
            for class_id, color in colors.items():
                mask = data == class_id
                img_array[mask] = color
            
            # Convert to PIL Image
            img = Image.fromarray(img_array, 'RGBA')
            
            # Save to bytes
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            
            print(f"Tile {z}/{x}/{y} generated successfully")
            
            return StreamingResponse(img_byte_arr, media_type="image/png")

    except Exception as e:
        print(f"Error generating tile for {region_id} at {z}/{x}/{y}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating tile: {e}")
    
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

@app.get("/debug/segmentation-info")
def debug_segmentation():
    if "last_click" not in processed_regions:
        return {"error": "No processed region"}
    
    path = processed_regions["last_click"]
    if not os.path.exists(path):
        return {"error": "File not found"}
    
    with rasterio.open(path) as src:
        data = src.read(1)
        unique, counts = np.unique(data, return_counts=True)
        
        return {
            "path": path,
            "shape": data.shape,
            "crs": str(src.crs),
            "bounds": src.bounds,
            "transform": list(src.transform),
            "unique_values": dict(zip(unique.tolist(), counts.tolist())),
            "min": int(data.min()),
            "max": int(data.max())
        }