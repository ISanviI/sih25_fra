import ee
import rasterio
from rasterio.windows import Window
import numpy as np
import os
import time
import torch
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import requests
from dotenv import load_dotenv
from PIL import Image

def calculate_stats(seg_map):
    class_names = ['Background', 'Forest', 'Agriculture', 'Water', 'Built-up', 'Degraded Forest']
    unique, counts = np.unique(seg_map, return_counts=True)
    total_pixels = np.sum(counts)
    stats = {}
    for cls_id, count in zip(unique, counts):
        if cls_id < len(class_names):
            class_name = class_names[int(cls_id)]
            percentage = (count / total_pixels * 100) if total_pixels > 0 else 0
            stats[class_name] = percentage
    return stats

def get_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = smp.Unet(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=4,  # RGB + NIR
        classes=6,      # 6 FRA classes
        activation='softmax2d'
    )
    model = model.to(device)
    model.eval()
    return model, device

def preprocess_tile_4channel(tile_path):
    with rasterio.open(tile_path) as src:
        bands = src.read()
        img = np.moveaxis(bands, 0, -1)
        img = np.clip(img / 3000.0, 0, 1)
        img = np.nan_to_num(img, nan=0.0)
    return img

def process_geometry(lat: float, lon: float):
    """Processes a circular region around a given lat/lon point."""
    try:
        ee_service_account_email = os.getenv("EE_SERVICE_ACCOUNT_EMAIL")
        ee_key_file = os.getenv("EE_KEY_FILE")
        ee.Initialize(ee.ServiceAccountCredentials(ee_service_account_email, ee_key_file))
    except Exception as e:
        raise Exception(f"Error initializing Earth Engine: {e}")

    # Create a point from the lat/lon, then buffer it to create a circular region.
    click_point = ee.Geometry.Point([lon, lat])
    ee_geometry = click_point.buffer(5000)  # Buffer by 5000 meters (5km radius)

    image_collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')         .filterBounds(ee_geometry)         .filterDate('2024-01-01', '2025-09-22')         .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))         .sort('system:time_start', False)
    image = image_collection.first().select(['B2', 'B3', 'B4', 'B8'])

    url = image.getDownloadURL({
        'scale': 10,  # Back to high resolution
        'crs': 'EPSG:4326',
        'region': ee_geometry.getInfo()['coordinates'],
        'format': 'GEO_TIFF'  # Explicitly request a GeoTIFF instead of a ZIP
    })

    response = requests.get(url)
    tif_path = 'sentinel2_tile_4band.tif'
    with open(tif_path, 'wb') as f:
        f.write(response.content)

    tile_size = 512
    output_dir = 'tiles_4band'
    os.makedirs(output_dir, exist_ok=True)

    with rasterio.open(tif_path) as src:
        height, width = src.height, src.width
        meta = src.meta.copy()

    num_tiles_h = (height + tile_size - 1) // tile_size
    num_tiles_w = (width + tile_size - 1) // tile_size

    for tile_i in range(num_tiles_h):
        for tile_j in range(num_tiles_w):
            i_start = tile_i * tile_size
            j_start = tile_j * tile_size
            i_end = min(i_start + tile_size, height)
            j_end = min(j_start + tile_size, width)

            with rasterio.open(tif_path) as tile_src:
                window = Window(j_start, i_start, j_end - j_start, i_end - i_start)
                tile_data = tile_src.read(window=window)

                tile_path = os.path.join(output_dir, f'tile_{tile_i}_{tile_j}.tif')
                meta.update({
                    'height': window.height,
                    'width': window.width,
                    'transform': tile_src.window_transform(window),
                    'count': 4,
                    'dtype': 'uint16'
                })

                with rasterio.open(tile_path, 'w', **meta) as dst:
                    dst.write(tile_data)

    model, device = get_model()
    class_names = ['Background', 'Forest', 'Agriculture', 'Water', 'Built-up', 'Degraded Forest']
    seg_dir = 'segmented_tiles_4band'
    os.makedirs(seg_dir, exist_ok=True)

    tile_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.tif')])

    for tile_file in tile_files:
        tile_path = os.path.join(output_dir, tile_file)
        img = preprocess_tile_4channel(tile_path)
        h, w = img.shape[:2]

        input_tensor = torch.tensor(img).permute(2, 0, 1).float().unsqueeze(0).to(device)

        with torch.no_grad():
            prediction = model(input_tensor)
            prediction = torch.softmax(prediction, dim=1)
            seg_map = torch.argmax(prediction, dim=1).squeeze().cpu().numpy()

        nir, red = img[:,:,3], img[:,:,2]
        ndvi = (nir - red) / (nir + red + 1e-10)

        forest_mask = (ndvi > 0.4) & ((seg_map == 0) | (seg_map == 3))
        agri_mask = (ndvi > 0.2) & (ndvi <= 0.4) & (seg_map == 3)
        degraded_mask = (ndvi > 0.0) & (ndvi <= 0.2) & (seg_map == 1)

        seg_map[forest_mask] = 1
        seg_map[agri_mask] = 2
        seg_map[degraded_mask] = 5

        # Save as raw PNG with PIL instead of matplotlib
        seg_filename = tile_file.replace('.tif', '_seg.png')
        seg_path = os.path.join(seg_dir, seg_filename)
        
        # Convert to PIL Image and save (this preserves exact pixel values)
        seg_img = Image.fromarray(seg_map.astype(np.uint8), mode='L')  # 'L' for grayscale
        seg_img.save(seg_path, compress_level=1)
        
        print(f"Saved {seg_filename} with classes: {np.unique(seg_map)}")

    stitched_path = 'stitched_segmentation_4band.tif'
    tile_files = sorted([f for f in os.listdir(seg_dir) if f.endswith('_seg.png')])

    if not tile_files:
        raise Exception("No segmentation tiles found!")

    indices = [tuple(map(int, f.split('_')[1:3])) for f in tile_files]
    max_i, max_j = max(idx[0] for idx in indices), max(idx[1] for idx in indices)

    # Get the first tile to extract metadata and geotransform
    first_tile_path = os.path.join(output_dir, 'tile_0_0.tif')
    with rasterio.open(first_tile_path) as src:
        tile_h, tile_w = src.height, src.width
        base_transform = src.transform
        crs = src.crs
        bounds = src.bounds
        
        print(f"Base transform: {base_transform}")
        print(f"CRS: {crs}")
        print(f"Tile size: {tile_h}x{tile_w}")
        print(f"First tile bounds: {bounds}")

    # Calculate full dimensions
    full_h = (max_i + 1) * tile_h
    full_w = (max_j + 1) * tile_w
    full_seg = np.zeros((full_h, full_w), dtype=np.uint8)

    print(f"Stitching {len(tile_files)} tiles into {full_h}x{full_w} image")

    # Stitch all tiles together
    for tile_file in tile_files:
        parts = tile_file.replace('_seg.png','').split('_')
        ti, tj = int(parts[1]), int(parts[2])
        tile_path = os.path.join(seg_dir, tile_file)
        
        # Read the segmentation tile as grayscale using PIL to preserve exact values
        seg_tile_img = Image.open(tile_path)
        seg_tile = np.array(seg_tile_img).astype(np.uint8)
        
        th, tw = seg_tile.shape
        full_seg[ti*tile_h:ti*tile_h+th, tj*tile_w:tj*tile_w+tw] = seg_tile
        
        print(f"Stitched tile {ti},{tj} - unique values: {np.unique(seg_tile)}")

    # Check the full segmentation
    unique_vals, counts = np.unique(full_seg, return_counts=True)
    print(f"Full segmentation unique values: {dict(zip(unique_vals, counts))}")

    # Write the stitched file with proper georeferencing
    meta = {
        'driver': 'GTiff',
        'height': full_h,
        'width': full_w,
        'count': 1,
        'dtype': 'uint8',
        'crs': crs,
        'transform': base_transform,
        'compress': 'lzw'
    }

    with rasterio.open(stitched_path, 'w', **meta) as dst:
        dst.write(full_seg, 1)

    print(f"Stitched segmentation saved to {stitched_path}")

    # Recalculate stats from the actual stitched image
    stats = calculate_stats(full_seg)
    print(f"Final statistics: {stats}")

    return {"message": "Processing complete", "stitched_image_path": stitched_path, "stats": stats}