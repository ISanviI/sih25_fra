import React, { useState, useEffect } from 'react';
import { MapContainer, TileLayer, LayersControl, useMapEvents } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import axios from 'axios';

const MapClickHandler = ({ setOverlayUrl }) => {
  const [loading, setLoading] = useState(false);

  useMapEvents({
    click: async (e) => {
      const { lat, lng } = e.latlng;
      setLoading(true);
      console.log(`Clicked at: ${lat}, ${lng}`);

      try {
        const response = await axios.post('http://localhost:8000/process-region', null, {
          params: {
            lat: lat,
            lon: lng
          }
        });
        
        const regionId = response.data.region_id;
        console.log(`Processed region ID: ${regionId}`);
        setOverlayUrl(`http://localhost:8000/tiles/${regionId}/{z}/{x}/{y}`);

      } catch (error) {
        console.error("Error processing region:", error);
        alert("Could not process the selected region. Please check the console for details.");
      } finally {
        setLoading(false);
      }
    },
  });

  return loading ? <div className="loading-indicator">Processing Region...</div> : null;
};

function MapComponent() {
  const [baseLayerUrl, setBaseLayerUrl] = useState(null);
  const [overlayUrl, setOverlayUrl] = useState(null);

  useEffect(() => {
    const fetchBaseLayer = async () => {
      try {
        const response = await axios.get('http://localhost:8000/ee-tile-url');
        setBaseLayerUrl(response.data.tile_url);
      } catch (error) {
        console.error("Error fetching base layer:", error);
        alert("Could not load the Sentinel-2 base map. Please check the console and ensure the backend is running.");
      }
    };

    fetchBaseLayer();
  }, []);

  if (!baseLayerUrl) {
    return <div className="loading-indicator">Loading Sentinel-2 Map...</div>;
  }

  return (
    <MapContainer center={[20.5937, 78.9629]} zoom={5} style={{ height: "100vh", width: "100%" }}>
      <LayersControl position="topright">
        <LayersControl.BaseLayer checked name="Sentinel-2">
          <TileLayer url={baseLayerUrl} />
        </LayersControl.BaseLayer>
      </LayersControl>

      {overlayUrl && (
        <LayersControl.Overlay checked name="Segmented Layer">
            <TileLayer url={overlayUrl} key={overlayUrl} />
        </LayersControl.Overlay>
      )}

      <MapClickHandler setOverlayUrl={setOverlayUrl} />
    </MapContainer>
  );
}

export default MapComponent;