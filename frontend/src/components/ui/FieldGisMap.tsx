import React, { useEffect, useRef, useState } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import type { GridCell } from '../../types';

interface FieldGisMapProps {
  imageUrl: string | null;
  heatmapBase64?: string | null;
  gridData?: GridCell[];
  diseaseType?: string;
  averageSeverity?: number;
  lat?: number;
  lon?: number;
}

export const FieldGisMap: React.FC<FieldGisMapProps> = ({
  imageUrl,
  heatmapBase64,
  gridData = [],
  diseaseType = 'Rice Blast',
  lat: rawLat = 13.9299, // Shivamogga, Karnataka (PESITM / Malnad Agricultural Zone)
  lon: rawLon = 75.5681,
}) => {
  const lat = typeof rawLat === 'number' && !isNaN(rawLat) ? rawLat : 13.9299;
  const lon = typeof rawLon === 'number' && !isNaN(rawLon) ? rawLon : 75.5681;
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<L.Map | null>(null);
  const overlayLayerRef = useRef<L.LayerGroup | null>(null);
  const [activeLayer, setActiveLayer] = useState<'satellite' | 'street'>('satellite');
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [showGridPatches, setShowGridPatches] = useState(true);

  // Approximate field bounding box (~150m x 150m plot)
  const dLat = 0.00135;
  const dLon = 0.00135;
  const bounds: L.LatLngBoundsExpression = [
    [lat - dLat, lon - dLon],
    [lat + dLat, lon + dLon],
  ];

  useEffect(() => {
    if (!mapContainerRef.current) return;

    // Fix default marker icon assets in Leaflet
    delete (L.Icon.Default.prototype as any)._getIconUrl;
    L.Icon.Default.mergeOptions({
      iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
      iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
      shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
    });

    const map = L.map(mapContainerRef.current, {
      center: [lat, lon],
      zoom: 17,
      zoomControl: true,
      attributionControl: false,
    });
    mapInstanceRef.current = map;

    // Basemaps
    const satelliteLayer = L.tileLayer(
      'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
      { maxZoom: 19 }
    );
    const streetLayer = L.tileLayer(
      'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
      { maxZoom: 19 }
    );

    if (activeLayer === 'satellite') {
      satelliteLayer.addTo(map);
    } else {
      streetLayer.addTo(map);
    }

    // Scale control
    L.control.scale({ imperial: false, metric: true }).addTo(map);

    const overlayGroup = L.layerGroup().addTo(map);
    overlayLayerRef.current = overlayGroup;

    map.fitBounds(bounds);

    return () => {
      map.remove();
      mapInstanceRef.current = null;
    };
  }, [lat, lon]);

  // Update tile layer on toggle
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map) return;

    map.eachLayer((layer) => {
      if (layer instanceof L.TileLayer) {
        map.removeLayer(layer);
      }
    });

    const tileUrl =
      activeLayer === 'satellite'
        ? 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
        : 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png';

    L.tileLayer(tileUrl, { maxZoom: 19 }).addTo(map);
  }, [activeLayer]);

  // Render Field Boundary, Canopy Drone Overlay & Patch Markers
  useEffect(() => {
    const overlayGroup = overlayLayerRef.current;
    if (!overlayGroup) return;

    overlayGroup.clearLayers();

    // 1. Field Perimeter Polygon (Golden dashed boundary)
    const fieldPerimeter = L.rectangle(bounds, {
      color: '#f59e0b',
      weight: 2,
      dashArray: '5, 5',
      fillColor: '#10b981',
      fillOpacity: 0.08,
    });
    overlayGroup.addLayer(fieldPerimeter);

    // 2. Heatmap Image Overlay or Drone Canopy Image Overlay
    if (showHeatmap && heatmapBase64) {
      const heatmapOverlay = L.imageOverlay(`data:image/png;base64,${heatmapBase64}`, bounds, {
        opacity: 0.72,
        interactive: false,
      });
      overlayGroup.addLayer(heatmapOverlay);
    } else if (imageUrl) {
      const droneOverlay = L.imageOverlay(imageUrl, bounds, {
        opacity: 0.85,
        interactive: false,
      });
      overlayGroup.addLayer(droneOverlay);
    }

    // 3. Vector Grid Patch Markers (Hotspots)
    if (showGridPatches && Array.isArray(gridData) && gridData.length > 0) {
      const southWest = L.latLng(lat - dLat, lon - dLon);
      const northEast = L.latLng(lat + dLat, lon + dLon);
      const latSpan = northEast.lat - southWest.lat;
      const lonSpan = northEast.lng - southWest.lng;

      // Group into high severity hotspots
      gridData
        .filter((cell) => (cell?.severity ?? 0) > 35)
        .slice(0, 30) // Top 30 hotspot indicators
        .forEach((cell, idx) => {
          // Normalize position within 0..1 bounding box using x_percent / y_percent
          const rawX = cell.x_percent ?? (cell as any).x ?? 50;
          const rawY = cell.y_percent ?? (cell as any).y ?? 50;
          const relX = Math.min(Math.max((Number(rawX) || 0) / 100.0, 0), 1);
          const relY = Math.min(Math.max((Number(rawY) || 0) / 100.0, 0), 1);

          const cellLat = northEast.lat - relY * latSpan;
          const cellLon = southWest.lng + relX * lonSpan;

          if (isNaN(cellLat) || isNaN(cellLon)) return;

          const color = (cell.severity ?? 0) > 60 ? '#ef4444' : '#f59e0b';
          const circle = L.circleMarker([cellLat, cellLon], {
            radius: 5,
            fillColor: color,
            color: '#ffffff',
            weight: 1.5,
            opacity: 1,
            fillOpacity: 0.85,
          });

          const cellSev = typeof cell.severity === 'number' ? cell.severity.toFixed(1) : '0.0';
          circle.bindPopup(`
            <div style="font-family: inherit; font-size: 12px; line-height: 1.4; color: #0f172a;">
              <strong style="color: ${color};">Hotspot #${idx + 1}</strong><br/>
              <b>Pathogen:</b> ${cell.disease || diseaseType}<br/>
              <b>Severity:</b> ${cellSev}%<br/>
              <b>Geo:</b> ${cellLat.toFixed(5)}°N, ${cellLon.toFixed(5)}°E
            </div>
          `);

          overlayGroup.addLayer(circle);
        });
    }
  }, [imageUrl, heatmapBase64, gridData, showHeatmap, showGridPatches, diseaseType]);

  return (
    <div style={{ position: 'relative', width: '100%', height: 480, borderRadius: 12, overflow: 'hidden' }}>
      <div ref={mapContainerRef} style={{ width: '100%', height: '100%', zIndex: 1 }} />

      {/* Floating GIS Toolbar */}
      <div
        style={{
          position: 'absolute',
          top: 12,
          right: 12,
          zIndex: 1000,
          display: 'flex',
          gap: 6,
          background: 'rgba(15, 23, 42, 0.85)',
          backdropFilter: 'blur(8px)',
          padding: '4px 6px',
          borderRadius: 8,
          border: '1px solid rgba(255, 255, 255, 0.1)',
        }}
      >
        <button
          className={`btn-icon ${activeLayer === 'satellite' ? 'active' : ''}`}
          onClick={() => setActiveLayer('satellite')}
          style={{ fontSize: '0.72rem', padding: '4px 8px', borderRadius: 6 }}
        >
          🛰️ Satellite
        </button>
        <button
          className={`btn-icon ${activeLayer === 'street' ? 'active' : ''}`}
          onClick={() => setActiveLayer('street')}
          style={{ fontSize: '0.72rem', padding: '4px 8px', borderRadius: 6 }}
        >
          🗺️ Street
        </button>
        <button
          className={`btn-icon ${showHeatmap ? 'active' : ''}`}
          onClick={() => setShowHeatmap(!showHeatmap)}
          style={{ fontSize: '0.72rem', padding: '4px 8px', borderRadius: 6 }}
        >
          🔥 Heatmap
        </button>
        <button
          className={`btn-icon ${showGridPatches ? 'active' : ''}`}
          onClick={() => setShowGridPatches(!showGridPatches)}
          style={{ fontSize: '0.72rem', padding: '4px 8px', borderRadius: 6 }}
        >
          📍 Hotspots
        </button>
      </div>

      {/* Bottom GIS Coordinates & Scale Bar Note */}
      <div
        style={{
          position: 'absolute',
          bottom: 10,
          left: 12,
          zIndex: 1000,
          background: 'rgba(15, 23, 42, 0.85)',
          backdropFilter: 'blur(8px)',
          padding: '4px 10px',
          borderRadius: 6,
          border: '1px solid rgba(255, 255, 255, 0.1)',
          fontSize: '0.72rem',
          color: '#e2e8f0',
          display: 'flex',
          alignItems: 'center',
          gap: 8,
        }}
      >
        <span style={{ color: '#10b981', fontWeight: 700 }}>● GIS Orthomosaic</span>
        <span>{lat.toFixed(4)}°N, {lon.toFixed(4)}°E</span>
        <span style={{ opacity: 0.6 }}>|</span>
        <span>Plot: 150m × 150m</span>
      </div>
    </div>
  );
};

export default FieldGisMap;
