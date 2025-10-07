# FRA WebGIS

This project is a web-based GIS application for visualizing and managing land records. It uses a FastAPI backend, a React frontend with Leaflet, and a PostGIS database to allow users to click on a map, trigger a satellite image segmentation process, and view the results as a tile layer.

## Core Technologies

- **Backend:** FastAPI, PostgreSQL, PostGIS, GeoAlchemy2, SQLAlchemy
- **Frontend:** React, Leaflet, React-Leaflet
- **GIS & Image Processing:** Google Earth Engine, Sentinel-2, U-Net (via `segmentation-models-pytorch`)
- **Database Migrations:** Alembic

## Project Setup

### 1. Database Setup

1.  **Install and run PostgreSQL with the PostGIS extension.**
2.  Create a new database for this project.
3.  Connect to your new database and run the following SQL command to enable PostGIS:
    ```sql
    CREATE EXTENSION postgis;
    ```

### 2. Backend Setup

1.  **Navigate to the `backend` directory.**
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
3.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Configure your environment variables:**

    - Create a copy of `.env.example` and name it `.env`.
    - Fill in the values for your database connection and your Google Earth Engine service account:

      ```env
      # Earth Engine Credentials
      EE_SERVICE_ACCOUNT_EMAIL=your-service-account-email@your-project-id.iam.gserviceaccount.com
      EE_KEY_FILE=/path/to/your/ee-keyfile.json

      # Database Connection
      POSTGRES_USER=your_db_user
      POSTGRES_PASSWORD=your_db_password
      POSTGRES_SERVER=localhost # Or your DB host
      POSTGRES_DB=your_db_name
      ```

5.  **Run Database Migrations:**
    - Before running the application, you need to create the database tables from the models defined in the code.
    - Alembic is used to manage the database schema. To apply all migrations, run:
      ```bash
      alembic upgrade head
      ```
6.  **Run the backend server:**
    ```bash
    uvicorn app.main:app --reload
    ```
    The backend will be available at `http://localhost:8000`.

### 3. Frontend Setup

1.  **Navigate to the `frontend` directory.**
2.  **Install npm packages:**
    ```bash
    npm install
    ```
3.  **Run the frontend development server:**
    ```bash
    npm run dev
    ```
    The frontend will be available at `http://localhost:5173` (or another port if 5173 is busy).

## Database Migrations with Alembic

This project uses **Alembic** to manage database schema changes.

### Generating a New Migration

After you have made changes to your database models (in `backend/app/models.py`), you will need to generate a new migration script. To do this, run the following command from the `backend` directory:

```bash
alembic revision --autogenerate -m "Your migration message"
```

> Edit the first migration file with `import geoalchemy2` and remove index in both upgrade and downgrade commands.

### Applying Migrations

To apply migrations to your database, run the following command from the `backend` directory:

```bash
alembic upgrade head
```
