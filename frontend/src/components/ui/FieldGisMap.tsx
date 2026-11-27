import React, { useEffect, useRef, useState, useCallback, useMemo } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import {
  MapPin, Navigation, Crosshair, ExternalLink, Copy, Check,
  ArrowUp, ArrowDown, ArrowLeft, ArrowRight, LocateFixed,
  Maximize2, Lock, Unlock, Layers, Flame, Image as ImageIcon
} from 'lucide-react';
import type { GridCell } from '../../types';

interface FieldGisMapProps {
  imageUrl: string | null;
  heatmapDataUrl?: string | null;
  gridData?: GridCell[];
  averageSeverity?: number;
  lat?: number;
  lon?: number;
  onSelectCell?: (cell: GridCell) => void;
}

const METERS_PER_DEG_LAT = 111320;

function metersToLatLonDelta(lat: number, widthMeters: number, heightMeters: number) {
  const cosLat = Math.max(0.1, Math.cos((lat * Math.PI) / 180));
  const dLat = heightMeters / 2 / METERS_PER_DEG_LAT;
  const dLon = widthMeters / 2 / (METERS_PER_DEG_LAT * cosLat);
  return { dLat, dLon };
}

/**
 * Parses common Google Maps coordinate formats and URLs:
 * - "13.9299, 75.5681" / "13.9299 75.5681"
 * - "13.9299° N, 75.5681° E"
 * - "https://www.google.com/maps/@13.9299,75.5681,17z"
 * - "https://maps.google.com/?q=13.9299,75.5681"
 * - "...!3d13.9299!4d75.5681..."
 * - DMS: `13°55'47.6"N 75°34'05.2"E`
 */
export function parseGoogleMapsCoordinates(raw: string): { lat: number; lon: number } | null {
  const input = raw.trim();
  if (!input) return null;

  // 1. Google Maps URL patterns (!3d...!4d..., @lat,lon, ?q=lat,lon, ll=lat,lon)
  const bangMatch = input.match(/!3d(-?\d+(?:\.\d+)?)!4d(-?\d+(?:\.\d+)?)/);
  if (bangMatch) {
    const lat = parseFloat(bangMatch[1]);
    const lon = parseFloat(bangMatch[2]);
    if (isValidLatLon(lat, lon)) return { lat, lon };
  }

  const atMatch = input.match(/@(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)/);
  if (atMatch) {
    const lat = parseFloat(atMatch[1]);
    const lon = parseFloat(atMatch[2]);
    if (isValidLatLon(lat, lon)) return { lat, lon };
  }

  const queryMatch = input.match(/[?&](?:q|ll|center)=(-?\d+(?:\.\d+)?)(?:,|%2C)\s*(-?\d+(?:\.\d+)?)/i);
  if (queryMatch) {
    const lat = parseFloat(queryMatch[1]);
    const lon = parseFloat(queryMatch[2]);
    if (isValidLatLon(lat, lon)) return { lat, lon };
  }

  // 2. DMS format: 13°55'47.6"N 75°34'05.2"E
  const dmsRegex = /(\d+(?:\.\d+)?)[°\s]+(\d+(?:\.\d+)?)['′\s]+(\d+(?:\.\d+)?)["″\s]*([NS])[,\s]+(\d+(?:\.\d+)?)[°\s]+(\d+(?:\.\d+)?)['′\s]+(\d+(?:\.\d+)?)["″\s]*([EW])/i;
  const dmsMatch = input.match(dmsRegex);
  if (dmsMatch) {
    const dmsToDec = (d: string, m: string, s: string, dir: string) => {
      const dec = parseFloat(d) + parseFloat(m) / 60 + parseFloat(s) / 3600;
      return /[SW]/i.test(dir) ? -dec : dec;
    };
    const lat = dmsToDec(dmsMatch[1], dmsMatch[2], dmsMatch[3], dmsMatch[4]);
    const lon = dmsToDec(dmsMatch[5], dmsMatch[6], dmsMatch[7], dmsMatch[8]);
    if (isValidLatLon(lat, lon)) return { lat, lon };
  }

  // 3. Decimal degrees with optional N/S/E/W and degree symbols
  const decRegex = /(-?\d+(?:\.\d+)?)\s*°?\s*([NS])?[\s,;]+(-?\d+(?:\.\d+)?)\s*°?\s*([EW])?/i;
  const decMatch = input.match(decRegex);
  if (decMatch) {
    let lat = parseFloat(decMatch[1]);
    let lon = parseFloat(decMatch[3]);
    if (decMatch[2] && /S/i.test(decMatch[2])) lat = -Math.abs(lat);
    if (decMatch[4] && /W/i.test(decMatch[4])) lon = -Math.abs(lon);
    if (isValidLatLon(lat, lon)) return { lat, lon };
  }

  return null;
}

function isValidLatLon(lat: number, lon: number): boolean {
  return !isNaN(lat) && !isNaN(lon) && lat >= -90 && lat <= 90 && lon >= -180 && lon <= 180;
}

function escapeHtml(value: string): string {
  const entities: Record<string, string> = {
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;',
  };
  return value.replace(/[&<>"']/g, (character) => entities[character]);
}

const SIZE_PRESETS = [
  { label: '1 Acre (~64m)', w: 64, h: 64 },
  { label: '1 Ha (100m)', w: 100, h: 100 },
  { label: '150×150m', w: 150, h: 150 },
  { label: '250×250m', w: 250, h: 250 },
  { label: '500×500m', w: 500, h: 500 },
];

export const FieldGisMap: React.FC<FieldGisMapProps> = ({
  imageUrl,
  heatmapDataUrl,
  gridData = [],
  lat,
  lon,
  onSelectCell,
}) => {
  const hasInitialCoordinates = typeof lat === 'number' && typeof lon === 'number' && isValidLatLon(lat, lon);
  const safeInitLat = hasInitialCoordinates ? lat : 0;
  const safeInitLon = hasInitialCoordinates ? lon : 0;

  // Coordinates and plot geometry are only shown after an explicit location choice.
  const [locationSelected, setLocationSelected] = useState(hasInitialCoordinates);
  const [locationSource, setLocationSource] = useState<string | null>(hasInitialCoordinates ? 'Provided coordinates' : null);
  const [centerLat, setCenterLat] = useState<number>(safeInitLat);
  const [centerLon, setCenterLon] = useState<number>(safeInitLon);
  const [widthMeters, setWidthMeters] = useState<number>(150);
  const [heightMeters, setHeightMeters] = useState<number>(150);
  const [lockSquare, setLockSquare] = useState<boolean>(true);
  const [nudgeStepMeters, setNudgeStepMeters] = useState<number>(15);

  // Coordinate Input & Feedback
  const [coordInput, setCoordInput] = useState<string>(hasInitialCoordinates ? `${safeInitLat.toFixed(6)}, ${safeInitLon.toFixed(6)}` : '');
  const [coordError, setCoordError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [locatingGps, setLocatingGps] = useState(false);

  // Interactive Map Mode & Layers
  const [activeLayer, setActiveLayer] = useState<'satellite' | 'street'>('satellite');
  const [showPhoto, setShowPhoto] = useState<boolean>(true);
  const [showHeatmap, setShowHeatmap] = useState<boolean>(true);
  const [showGridPatches, setShowGridPatches] = useState<boolean>(true);
  const [overlayOpacity, setOverlayOpacity] = useState<number>(0.78);
  const [tapToPlace, setTapToPlace] = useState<boolean>(false);

  // Leaflet Refs
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<L.Map | null>(null);
  const tileLayerRef = useRef<L.TileLayer | null>(null);
  const overlayLayerRef = useRef<L.LayerGroup | null>(null);
  const tapToPlaceRef = useRef<boolean>(tapToPlace);
  tapToPlaceRef.current = tapToPlace;

  // Keep text box synced when coordinates move via drag/nudge/GPS
  useEffect(() => {
    if (locationSelected) setCoordInput(`${centerLat.toFixed(6)}, ${centerLon.toFixed(6)}`);
  }, [centerLat, centerLon, locationSelected]);

  const { dLat, dLon } = useMemo(
    () => metersToLatLonDelta(centerLat, widthMeters, heightMeters),
    [centerLat, widthMeters, heightMeters]
  );

  const bounds: L.LatLngBoundsExpression = useMemo(
    () => [
      [centerLat - dLat, centerLon - dLon],
      [centerLat + dLat, centerLon + dLon],
    ],
    [centerLat, centerLon, dLat, dLon]
  );

  const areaStats = useMemo(() => {
    const sqMeters = widthMeters * heightMeters;
    const hectares = sqMeters / 10000;
    const acres = sqMeters / 4046.8564224;
    return { hectares, acres };
  }, [widthMeters, heightMeters]);

  const fitFieldView = useCallback((targetLat = centerLat, targetLon = centerLon, wM = widthMeters, hM = heightMeters) => {
    const map = mapInstanceRef.current;
    if (!map) return;
    const delta = metersToLatLonDelta(targetLat, wM, hM);
    map.fitBounds(
      [
        [targetLat - delta.dLat, targetLon - delta.dLon],
        [targetLat + delta.dLat, targetLon + delta.dLon],
      ],
      { padding: [44, 44], maxZoom: 19, animate: true }
    );
  }, [centerLat, centerLon, widthMeters, heightMeters]);

  // 1. Initialize Leaflet Map ONCE
  useEffect(() => {
    if (!mapContainerRef.current || mapInstanceRef.current) return;

    const map = L.map(mapContainerRef.current, {
      center: hasInitialCoordinates ? [safeInitLat, safeInitLon] : [20, 0],
      zoom: hasInitialCoordinates ? 17 : 2,
      maxZoom: 20,
      zoomControl: true,
      attributionControl: false,
    });
    mapInstanceRef.current = map;

    const satelliteLayer = L.tileLayer(
      'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
      { maxZoom: 20, maxNativeZoom: 19 }
    );
    satelliteLayer.addTo(map);
    tileLayerRef.current = satelliteLayer;

    L.control.scale({ imperial: false, metric: true, position: 'bottomright' }).addTo(map);

    const overlayGroup = L.layerGroup().addTo(map);
    overlayLayerRef.current = overlayGroup;

    // Click-to-place handler
    map.on('click', (e: L.LeafletMouseEvent) => {
      if (tapToPlaceRef.current) {
        setCenterLat(e.latlng.lat);
        setCenterLon(e.latlng.lng);
        setLocationSelected(true);
        setLocationSource('Map-selected position');
        setCoordError(null);
      }
    });

    if (hasInitialCoordinates) {
      const initDelta = metersToLatLonDelta(safeInitLat, 150, 150);
      map.fitBounds(
        [
          [safeInitLat - initDelta.dLat, safeInitLon - initDelta.dLon],
          [safeInitLat + initDelta.dLat, safeInitLon + initDelta.dLon],
        ],
        { padding: [44, 44] }
      );
    }

    return () => {
      map.remove();
      mapInstanceRef.current = null;
      tileLayerRef.current = null;
      overlayLayerRef.current = null;
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // 2. Swap Basemap Layer Smoothly
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map) return;

    if (tileLayerRef.current) {
      map.removeLayer(tileLayerRef.current);
    }

    const tileUrl =
      activeLayer === 'satellite'
        ? 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
        : 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png';

    const newLayer = L.tileLayer(tileUrl, { maxZoom: 20, maxNativeZoom: 19 });
    newLayer.addTo(map);
    newLayer.bringToBack();
    tileLayerRef.current = newLayer;
  }, [activeLayer]);

  // Render the user-positioned illustrative extent and its aligned overlays.
  useEffect(() => {
    const overlayGroup = overlayLayerRef.current;
    if (!overlayGroup) return;

    overlayGroup.clearLayers();

    // A. User-positioned image underlay; no georeferenced boundary is supplied.
    if (locationSelected && showPhoto && imageUrl) {
      const photoOverlay = L.imageOverlay(imageUrl, bounds, {
        opacity: showHeatmap ? Math.max(0.25, overlayOpacity * 0.65) : overlayOpacity,
        interactive: false,
      });
      overlayGroup.addLayer(photoOverlay);
    }

    // B. Heatmap aligned to the illustrative extent, not a measured footprint.
    if (locationSelected && showHeatmap && heatmapDataUrl) {
      const heatmapOverlay = L.imageOverlay(heatmapDataUrl, bounds, {
        opacity: overlayOpacity,
        interactive: false,
      });
      overlayGroup.addLayer(heatmapOverlay);
    }

    // C. User-positioned extent (illustrative; no measured boundary is provided by this screen).
    if (locationSelected) {
      const fieldPerimeter = L.rectangle(bounds, {
      color: '#f59e0b',
      weight: 2.5,
      dashArray: '6, 6',
      fillColor: '#10b981',
      fillOpacity: showPhoto || showHeatmap ? 0.03 : 0.1,
      interactive: false,
      });
      overlayGroup.addLayer(fieldPerimeter);
    }

    // D. Draggable Center Pin (lets user drag the entire field plot directly on the satellite map)
    if (locationSelected) {
      const pinIcon = L.divIcon({
        className: 'gis-center-pin-wrapper',
        html: `<div class="gis-center-pin" title="Drag to reposition the illustrative plot"><span></span></div>`,
        iconSize: [28, 28],
        iconAnchor: [14, 14],
      });
      const centerMarker = L.marker([centerLat, centerLon], {
        icon: pinIcon,
        draggable: true,
        zIndexOffset: 500,
      });
      centerMarker.bindTooltip('Drag to reposition the illustrative plot', {
        direction: 'top',
        offset: [0, -14],
        opacity: 0.9,
      });
      centerMarker.on('dragend', () => {
        const pos = centerMarker.getLatLng();
        setCenterLat(pos.lat);
        setCenterLon(pos.lng);
        setLocationSource('Dragged map marker');
        setCoordError(null);
      });
      overlayGroup.addLayer(centerMarker);
    }

    // E. Model patches placed within the user-positioned illustrative extent.
    if (locationSelected && showGridPatches && Array.isArray(gridData) && gridData.length > 0) {
      const southWest = L.latLng(centerLat - dLat, centerLon - dLon);
      const northEast = L.latLng(centerLat + dLat, centerLon + dLon);
      const latSpan = northEast.lat - southWest.lat;
      const lonSpan = northEast.lng - southWest.lng;

      gridData
        .filter((cell) => (cell?.severity ?? 0) > 35)
        .slice(0, 35)
        .forEach((cell, idx) => {
          const cxPct = (cell.x_percent ?? 50) + (cell.w_percent ?? 0) * 0.5;
          const cyPct = (cell.y_percent ?? 50) + (cell.h_percent ?? 0) * 0.5;
          const relX = Math.min(Math.max(cxPct / 100.0, 0), 1);
          const relY = Math.min(Math.max(cyPct / 100.0, 0), 1);

          const cellLat = northEast.lat - relY * latSpan;
          const cellLon = southWest.lng + relX * lonSpan;
          if (isNaN(cellLat) || isNaN(cellLon)) return;

          const color = (cell.severity ?? 0) > 60 ? '#ef4444' : '#f59e0b';
          const circle = L.circleMarker([cellLat, cellLon], {
            radius: 5.5,
            fillColor: color,
            color: '#ffffff',
            weight: 1.5,
            opacity: 1,
            fillOpacity: 0.9,
          });

          const cellSev = typeof cell.severity === 'number' ? cell.severity.toFixed(1) : '0.0';
          circle.bindPopup(`
            <div style="font-family: inherit; font-size: 12px; line-height: 1.45; color: #0f172a;">
              <strong style="color: ${color};">Model patch #${idx + 1}</strong><br/>
              <b>Model output:</b> ${escapeHtml(cell.disease || 'Not provided')}<br/>
              <b>Severity:</b> ${cellSev}%<br/>
              <b>Approx. overlay position:</b> ${cellLat.toFixed(6)}, ${cellLon.toFixed(6)}
            </div>
          `);
          circle.on('click', () => {
            if (onSelectCell) onSelectCell(cell);
          });

          overlayGroup.addLayer(circle);
        });
    }
  }, [
    centerLat, centerLon, dLat, dLon, bounds, locationSelected,
    imageUrl, heatmapDataUrl, gridData,
    showPhoto, showHeatmap, showGridPatches, overlayOpacity,
      onSelectCell,
  ]);

  // Handlers
  const handleApplyCoords = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    const parsed = parseGoogleMapsCoordinates(coordInput);
    if (!parsed) {
      setCoordError('Invalid format. Paste Google Maps coords (e.g. 13.9299, 75.5681) or a Google Maps URL.');
      return;
    }
    setCoordError(null);
    setCenterLat(parsed.lat);
    setCenterLon(parsed.lon);
    setLocationSelected(true);
    setLocationSource('Pasted coordinates');
    fitFieldView(parsed.lat, parsed.lon, widthMeters, heightMeters);
  };

  const handleUseMyGps = () => {
    if (!navigator.geolocation) {
      setCoordError('Geolocation is not supported in this browser.');
      return;
    }
    setLocatingGps(true);
    setCoordError(null);
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        const nextLat = pos.coords.latitude;
        const nextLon = pos.coords.longitude;
        setCenterLat(nextLat);
        setCenterLon(nextLon);
        setLocationSelected(true);
        setLocationSource('Device GPS');
        setLocatingGps(false);
        fitFieldView(nextLat, nextLon, widthMeters, heightMeters);
      },
      () => {
        setLocatingGps(false);
        setCoordError('GPS permission denied or unavailable. You can paste Google Maps coordinates instead.');
      },
      { enableHighAccuracy: true, timeout: 8000 }
    );
  };

  const handleSnapToMapCenter = () => {
    const map = mapInstanceRef.current;
    if (!map) return;
    const c = map.getCenter();
    setCenterLat(c.lat);
    setCenterLon(c.lng);
    setLocationSelected(true);
    setLocationSource('Map view center');
    setCoordError(null);
  };

  const handleNudge = (northMeters: number, eastMeters: number) => {
    const cosLat = Math.max(0.1, Math.cos((centerLat * Math.PI) / 180));
    const nextLat = centerLat + northMeters / METERS_PER_DEG_LAT;
    const nextLon = centerLon + eastMeters / (METERS_PER_DEG_LAT * cosLat);
    setCenterLat(nextLat);
    setCenterLon(nextLon);
    setLocationSelected(true);
    setLocationSource('Adjusted map position');
  };

  const handleWidthChange = (nextW: number) => {
    const clamped = Math.min(1500, Math.max(20, nextW || 20));
    setWidthMeters(clamped);
    if (lockSquare) setHeightMeters(clamped);
  };

  const handleHeightChange = (nextH: number) => {
    const clamped = Math.min(1500, Math.max(20, nextH || 20));
    setHeightMeters(clamped);
    if (lockSquare) setWidthMeters(clamped);
  };

  const handleCopyCoords = () => {
    const text = `${centerLat.toFixed(6)}, ${centerLon.toFixed(6)}`;
    navigator.clipboard?.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  };

  const googleMapsUrl = `https://www.google.com/maps?q=${centerLat.toFixed(6)},${centerLon.toFixed(6)}`;

  return (
    <div className="gis-studio-wrapper">
      {/* ── Google Maps Coordinate Bar ── */}
      <form className="gis-coord-bar" onSubmit={handleApplyCoords}>
        <div className="gis-coord-input-group">
          <MapPin size={15} className="gis-coord-icon" />
          <input
            type="text"
            className="gis-coord-input"
            value={coordInput}
            onChange={(e) => { setCoordInput(e.target.value); if (coordError) setCoordError(null); }}
            placeholder="Paste Google Maps coordinates (e.g. 13.9299, 75.5681) or Google Maps URL…"
            aria-label="Google Maps coordinates or link"
          />
          <button
            type="button"
            className="gis-mini-btn"
            onClick={handleCopyCoords}
            title="Copy coordinates"
            aria-label="Copy current coordinates"
            disabled={!locationSelected}
          >
            {copied ? <Check size={13} color="var(--success)" /> : <Copy size={13} />}
          </button>
          {locationSelected && <a
            href={googleMapsUrl}
            target="_blank"
            rel="noreferrer"
            className="gis-mini-btn"
            title="Open the user-positioned location in Google Maps"
            aria-label="Open user-positioned location in Google Maps"
          >
            <ExternalLink size={13} />
          </a>}
        </div>

        <div className="gis-coord-actions">
          <button type="submit" className="btn-primary gis-action-btn">
            Locate Field
          </button>
          <button
            type="button"
            className="btn-icon gis-action-btn"
            onClick={handleUseMyGps}
            disabled={locatingGps}
            title="Use current device GPS"
          >
            <Navigation size={14} />
            <span>{locatingGps ? 'Locating…' : 'My GPS'}</span>
          </button>
        </div>
      </form>

      {coordError && <div className="gis-error-banner">{coordError}</div>}

      <div className="gis-provenance-note" role="note">
        {locationSelected
          ? `Illustrative placement from ${locationSource || 'user-provided coordinates'}. The adjustable rectangle is not a measured field boundary. Image and patch overlays are aligned to this extent; map patch markers use a display threshold of severity above 35 and approximate positions.`
          : 'No field location was provided. Paste coordinates, approve device GPS, or use map placement before adding an illustrative extent.'}
      </div>

      {/* ── Leaflet Satellite Map Stage ── */}
      <div className={`gis-map-stage ${tapToPlace ? 'gis-cursor-crosshair' : ''}`}>
        {!locationSelected && <div className="gis-map-empty-note">Choose a location above to show the illustrative plot extent.</div>}
        <div ref={mapContainerRef} className="gis-leaflet-canvas" />

        {/* Floating Layer Toolbar */}
        <div className="gis-floating-toolbar">
          <button
            type="button"
            className={`gis-toolbar-btn ${activeLayer === 'satellite' ? 'active' : ''}`}
            onClick={() => setActiveLayer('satellite')}
            aria-pressed={activeLayer === 'satellite'}
          >
            Satellite
          </button>
          <button
            type="button"
            className={`gis-toolbar-btn ${activeLayer === 'street' ? 'active' : ''}`}
            onClick={() => setActiveLayer('street')}
            aria-pressed={activeLayer === 'street'}
          >
            Street
          </button>
          <button
            type="button"
            className={`gis-toolbar-btn ${showPhoto ? 'active' : ''}`}
            onClick={() => setShowPhoto((v) => !v)}
            aria-pressed={showPhoto}
            title="Toggle field image underlay aligned to the illustrative extent"
          >
            <ImageIcon size={13} /> Photo
          </button>
          <button
            type="button"
            className={`gis-toolbar-btn ${showHeatmap ? 'active' : ''}`}
            onClick={() => setShowHeatmap((v) => !v)}
            aria-pressed={showHeatmap}
            title="Toggle the severity overlay aligned to the illustrative extent"
          >
            <Flame size={13} /> Heatmap
          </button>
          <button
            type="button"
            className={`gis-toolbar-btn ${showGridPatches ? 'active' : ''}`}
            onClick={() => setShowGridPatches((v) => !v)}
            aria-pressed={showGridPatches}
            title="Show model patches with severity above 35; map positions are approximate on this illustrative extent"
          >
            Patches &gt;35
          </button>
        </div>

        <div className="gis-telemetry-pill" role="status">
          {locationSelected ? (
            <>
              <span className="gis-telemetry-dot">Illustrative placement · {locationSource}</span>
              <span>{centerLat.toFixed(5)}°, {centerLon.toFixed(5)}°</span>
              <span className="gis-sep">|</span>
              <span>Estimated extent: {widthMeters}m × {heightMeters}m</span>
            </>
          ) : <span>Location not set · no field boundary available</span>}
        </div>
      </div>

      {/* ── Field Size, Map Movement & Alignment Controls ── */}
      <div className="gis-controls-panel">
        {/* Field Movement & Positioning */}
        <div className="gis-control-section">
          <div className="gis-control-title">
            <Crosshair size={14} color="var(--primary)" />
            <span>Map Movement &amp; Field Alignment</span>
          </div>

          <div className="gis-movement-row">
            <div className="gis-movement-buttons">
              <button
                type="button"
                className={`btn-icon ${tapToPlace ? 'active' : ''}`}
                    onClick={() => setTapToPlace((v) => !v)}
                    aria-pressed={tapToPlace}
                    title="Click anywhere on the map to set an illustrative center"
              >
                <MapPin size={14} />
                    <span>{tapToPlace ? 'Tap map to place' : locationSelected ? 'Tap map to move' : 'Place on map'}</span>
              </button>
              <button
                type="button"
                className="btn-icon"
                onClick={handleSnapToMapCenter}
                title="Pan the map to your farm, then click to snap field overlay to view center"
              >
                <LocateFixed size={14} />
                <span>Snap Field to View</span>
              </button>
              <button
                type="button"
                className="btn-icon"
                onClick={() => fitFieldView()}
                disabled={!locationSelected}
                title="Zoom & center map on the illustrative extent"
              >
                <Maximize2 size={14} />
                <span>Fit View</span>
              </button>
            </div>

            {/* Fine Nudge D-Pad */}
            <div className="gis-nudge-group">
              <span className="gis-nudge-label">Nudge ({nudgeStepMeters}m):</span>
              <div className="gis-nudge-pad">
                <button type="button" className="gis-nudge-btn"                 onClick={() => handleNudge(0, -nudgeStepMeters)} title="Move West" disabled={!locationSelected}><ArrowLeft size={13} /></button>
                <button type="button" className="gis-nudge-btn" onClick={() => handleNudge(nudgeStepMeters, 0)} title="Move North" disabled={!locationSelected}><ArrowUp size={13} /></button>
                <button type="button" className="gis-nudge-btn" onClick={() => handleNudge(-nudgeStepMeters, 0)} title="Move South" disabled={!locationSelected}><ArrowDown size={13} /></button>
                <button type="button" className="gis-nudge-btn" onClick={() => handleNudge(0, nudgeStepMeters)} title="Move East" disabled={!locationSelected}><ArrowRight size={13} /></button>
              </div>
              <select
                className="gis-step-select"
                value={nudgeStepMeters}
                onChange={(e) => setNudgeStepMeters(Number(e.target.value))}
                aria-label="Nudge step in meters"
              >
                <option value={5}>5m</option>
                <option value={15}>15m</option>
                <option value={50}>50m</option>
              </select>
            </div>
          </div>
        </div>

        {/* Field Size & Overlay Controls */}
        <div className="gis-control-section">
          <div className="gis-control-header-row">
            <div className="gis-control-title">
              <Layers size={14} color="var(--primary)" />
              <span>Illustrative dimensions &amp; overlay</span>
            </div>
            <div className="gis-area-badge" aria-label={`Illustrative area estimate: ${areaStats.hectares.toFixed(2)} hectares, ${areaStats.acres.toFixed(2)} acres`}>
              Illustrative area estimate: {areaStats.hectares.toFixed(2)} ha · {areaStats.acres.toFixed(2)} acres
            </div>
          </div>

          <div className="gis-sliders-grid">
            <div className="gis-slider-box">
              <div className="gis-slider-top">
                <label htmlFor="gis-width-range">Width (E–W)</label>
                <div className="gis-num-wrap">
                  <input
                    id="gis-width-num"
                    type="number"
                    min={20}
                    max={1500}
                    step={5}
                    value={widthMeters}
                    onChange={(e) => handleWidthChange(Number(e.target.value))}
                    className="gis-num-input"
                  />
                  <span>m</span>
                </div>
              </div>
              <input
                id="gis-width-range"
                type="range"
                min={20}
                max={600}
                step={5}
                value={Math.min(600, widthMeters)}
                onChange={(e) => handleWidthChange(Number(e.target.value))}
                className="gis-range"
              />
            </div>

            <div className="gis-slider-box">
              <div className="gis-slider-top">
                <label htmlFor="gis-height-range">Length (N–S)</label>
                <div className="gis-num-wrap">
                  <input
                    id="gis-height-num"
                    type="number"
                    min={20}
                    max={1500}
                    step={5}
                    value={heightMeters}
                    onChange={(e) => handleHeightChange(Number(e.target.value))}
                    className="gis-num-input"
                  />
                  <span>m</span>
                </div>
              </div>
              <input
                id="gis-height-range"
                type="range"
                min={20}
                max={600}
                step={5}
                value={Math.min(600, heightMeters)}
                onChange={(e) => handleHeightChange(Number(e.target.value))}
                className="gis-range"
              />
            </div>

            <div className="gis-slider-box">
              <div className="gis-slider-top">
                <label htmlFor="gis-opacity-range">Overlay Opacity</label>
                <span className="gis-opacity-val">{Math.round(overlayOpacity * 100)}%</span>
              </div>
              <input
                id="gis-opacity-range"
                type="range"
                min={0.15}
                max={1.0}
                step={0.05}
                value={overlayOpacity}
                onChange={(e) => setOverlayOpacity(parseFloat(e.target.value))}
                className="gis-range"
              />
            </div>
          </div>

          <div className="gis-presets-row">
            <button
              type="button"
              className={`btn-icon gis-lock-btn ${lockSquare ? 'active' : ''}`}
              onClick={() => {
                setLockSquare((v) => !v);
                if (!lockSquare) setHeightMeters(widthMeters);
              }}
            >
              {lockSquare ? <Lock size={12} /> : <Unlock size={12} />}
              <span>{lockSquare ? 'Square Locked' : 'Free Aspect'}</span>
            </button>

            <div className="gis-presets-list">
              {SIZE_PRESETS.map((p) => {
                const isActive = widthMeters === p.w && heightMeters === p.h;
                return (
                  <button
                    key={p.label}
                    type="button"
                    className={`gis-preset-pill ${isActive ? 'active' : ''}`}
                    onClick={() => {
                      setWidthMeters(p.w);
                      setHeightMeters(p.h);
                      fitFieldView(centerLat, centerLon, p.w, p.h);
                    }}
                  >
                    {p.label}
                  </button>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default FieldGisMap;
