import React, { useState, useEffect } from 'react';
import { MapContainer, TileLayer, LayersControl, useMapEvents, useMap } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import axios from 'axios';

const MapClickHandler = ({ setOverlayUrl, setStats, setProcessing }) => {
  useMapEvents({
    click: async (e) => {
      const { lat, lng } = e.latlng;
      setProcessing(true);
      console.log(`Clicked at: ${lat}, ${lng}`);

      try {
        const response = await axios.post('http://localhost:8000/process-region', null, {
          params: {
            lat: lat,
            lon: lng
          }
        });
        
        const regionId = response.data.region_id;
        const stats = response.data.stats;
        
        console.log(`Processed region ID: ${regionId}`);
        console.log('Stats:', stats);
        
        // Set the overlay URL - this will trigger a re-render
        const newOverlayUrl = `http://localhost:8000/tiles/${regionId}/{z}/{x}/{y}`;
        console.log('Setting overlay URL:', newOverlayUrl);
        setOverlayUrl(newOverlayUrl);
        setStats(stats);
        
        // Small delay to ensure state is updated
        setTimeout(() => {
          console.log('Overlay URL set, should be visible now');
          alert('Processing complete! Segmented layer added to map.');
        }, 100);

      } catch (error) {
        console.error("Error processing region:", error);
        alert("Could not process the selected region. Please check the console for details.");
      } finally {
        setProcessing(false);
      }
    },
  });

  return null;
};

// Component to force map refresh when overlay changes
const OverlayLayer = ({ url }) => {
  const map = useMap();
  
  useEffect(() => {
    // Force map to refresh
    if (url) {
      console.log('OverlayLayer mounted with URL:', url);
      setTimeout(() => {
        map.invalidateSize();
        console.log('Map invalidated');
      }, 100);
    }
  }, [url, map]);

  if (!url) {
    console.log('OverlayLayer: No URL provided');
    return null;
  }

  console.log('OverlayLayer: Rendering TileLayer with URL:', url);
  return (
    <TileLayer 
      url={url} 
      opacity={0.8}
      zIndex={1000}
      eventHandlers={{
        tileloadstart: () => console.log('Tile load started'),
        tileload: () => console.log('Tile loaded'),
        tileerror: (error) => console.error('Tile error:', error)
      }}
    />
  );
};

function MapComponent() {
  const [baseLayerUrl, setBaseLayerUrl] = useState(null);
  const [overlayUrl, setOverlayUrl] = useState(null);
  const [stats, setStats] = useState(null);
  const [processing, setProcessing] = useState(false);

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
    <div style={{ position: 'relative', height: '100vh', width: '100%' }}>
      <MapContainer 
        center={[20.5937, 78.9629]} 
        zoom={5} 
        style={{ height: "100vh", width: "100%" }}
      >
        <LayersControl position="topright">
          <LayersControl.BaseLayer checked name="Sentinel-2">
            <TileLayer url={baseLayerUrl} />
          </LayersControl.BaseLayer>
        </LayersControl>

        {/* Overlay layer rendered separately */}
        <OverlayLayer url={overlayUrl} />

        <MapClickHandler 
          setOverlayUrl={setOverlayUrl} 
          setStats={setStats}
          setProcessing={setProcessing}
        />
      </MapContainer>

      {/* Loading indicator */}
      {processing && (
        <div style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          backgroundColor: 'rgba(0, 0, 0, 0.8)',
          color: 'white',
          padding: '20px 40px',
          borderRadius: '8px',
          zIndex: 1000,
          fontSize: '18px',
          fontWeight: 'bold'
        }}>
          🛰️ Processing Satellite Imagery...
          <div style={{ fontSize: '14px', marginTop: '10px', fontWeight: 'normal' }}>
            This may take 30-60 seconds
          </div>
        </div>
      )}

      {/* Stats panel */}
      {stats && (
        <div style={{
          position: 'absolute',
          bottom: '20px',
          right: '20px',
          backgroundColor: 'white',
          padding: '15px',
          borderRadius: '8px',
          boxShadow: '0 2px 8px rgba(0,0,0,0.3)',
          zIndex: 1000,
          minWidth: '200px'
        }}>
          <h3 style={{ margin: '0 0 10px 0', fontSize: '16px' }}>Land Cover Statistics</h3>
          {Object.entries(stats).map(([className, percentage]) => (
            <div key={className} style={{ marginBottom: '5px', fontSize: '14px' }}>
              <strong>{className}:</strong> {percentage.toFixed(2)}%
            </div>
          ))}
          <button 
            onClick={() => setStats(null)}
            style={{
              marginTop: '10px',
              padding: '5px 10px',
              backgroundColor: '#f0f0f0',
              border: '1px solid #ccc',
              borderRadius: '4px',
              cursor: 'pointer',
              width: '100%'
            }}
          >
            Close
          </button>
        </div>
      )}

      {/* Debug panel */}
      {overlayUrl && (
        <div style={{
          position: 'absolute',
          top: '20px',
          right: '20px',
          backgroundColor: 'rgba(255, 255, 255, 0.9)',
          padding: '10px',
          borderRadius: '8px',
          boxShadow: '0 2px 8px rgba(0,0,0,0.3)',
          zIndex: 1000,
          fontSize: '12px',
          maxWidth: '300px'
        }}>
          <div style={{ marginBottom: '5px' }}>
            <strong>Overlay Active</strong>
          </div>
          <div style={{ fontSize: '10px', wordBreak: 'break-all', color: '#666' }}>
            {overlayUrl}
          </div>
          <button 
            onClick={() => {
              setOverlayUrl(null);
              setTimeout(() => setOverlayUrl(overlayUrl), 100);
            }}
            style={{
              marginTop: '5px',
              padding: '5px 10px',
              backgroundColor: '#4CAF50',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
              width: '100%',
              fontSize: '11px'
            }}
          >
            Refresh Overlay
          </button>
        </div>
      )}
    </div>
  );
}

export default MapComponent;