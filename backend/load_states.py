import json
import os
import requests
from sqlalchemy.orm import Session
from shapely.geometry import shape

# Make sure the script can find the app modules
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from app.database import SessionLocal, engine
from app.models import Region, Base

# URL for the GeoJSON data of Indian states
GEOJSON_URL = "https://raw.githubusercontent.com/geohacker/india/master/state/india_state.geojson"

def load_states_data():
    """Fetches Indian states GeoJSON and populates the regions table."""
    db: Session = SessionLocal()

    try:
        # Check if the table is empty before proceeding
        if db.query(Region).count() > 0:
            print("The 'regions' table is not empty. Aborting data load.")
            return

        print(f"Downloading states data from {GEOJSON_URL}...")
        response = requests.get(GEOJSON_URL)
        response.raise_for_status()  # Raise an exception for bad status codes
        data = response.json()

        print("Processing and loading states into the database...")
        states_added = 0
        for feature in data["features"]:
            properties = feature.get("properties", {})
            geometry = feature.get("geometry")

            if not properties or not geometry:
                continue

            state_name = properties.get("NAME_1")
            if not state_name:
                continue

            # Convert GeoJSON geometry to a Shapely shape
            geom_shape = shape(geometry)

            # Create a new Region object
            # The geometry is automatically handled by GeoAlchemy2
            new_region = Region(
                name=state_name,
                geom=f'SRID=4326;{geom_shape.wkt}' # Set SRID for WGS 84
            )
            db.add(new_region)
            states_added += 1

        print(f"Adding {states_added} states to the session.")
        db.commit()
        print("Successfully committed states to the database.")

    except Exception as e:
        print(f"An error occurred: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    print("--- Starting script to load Indian states into the database ---")
    # Alembic should handle table creation.
    # Base.metadata.create_all(bind=engine) # This line is removed.
    load_states_data()
    print("--- Script finished ---")
    print("To check state regions, run the following command in psql - \nSELECT id, name, ST_Summary(geom) as geometry_summary FROM regions LIMIT 5;")
