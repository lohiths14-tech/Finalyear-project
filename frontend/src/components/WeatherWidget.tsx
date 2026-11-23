/**
 * AgriPulse 2.0 — WeatherWidget
 *
 * Displays real-time weather, location, and pest risk data
 * fetched from the backend /api/v1/environment endpoint.
 *
 * Data sources (all real, no fabrication):
 *   - Open-Meteo    → weather + agro vars
 *   - Nominatim     → Indian district/state name
 *   - GBIF          → crop pest occurrence counts near field
 *
 * Behaviour:
 *   - Requests browser geolocation → passes to backend
 *   - If denied → shows Thanjavur fallback with note
 *   - If API unavailable → shows "data unavailable" per AGENTS.md policy
 */

import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import {
  Cloud, Droplets, Thermometer, MapPin,
  Bug, AlertTriangle, Loader2, WifiOff, Leaf,
  Sun, CloudRain, Navigation, RefreshCw
} from 'lucide-react';

// ─── Types ────────────────────────────────────────────────────────────────────

interface WeatherCurrent {
  temperature_c: number | null;
  feels_like_c: number | null;
  relative_humidity_pct: number | null;
  precipitation_mm: number | null;
  rain_mm: number | null;
  wind_speed_kmh: number | null;
}

interface AgroData {
  soil_temperature_c: number | null;
  et0_evapotranspiration_mm: number | null;
}

interface ForecastDay {
  day: number;
  precip_mm: number | null;
  temp_max: number | null;
  temp_min: number | null;
}

interface WeatherData {
  available: boolean;
  reason?: string;
  source?: string;
  season?: string;
  current?: WeatherCurrent;
  agro?: AgroData;
  disease_risk_from_weather?: string;
  forecast_3day?: ForecastDay[];
}

interface LocationData {
  available: boolean;
  reason?: string;
  location_string?: string;
  district?: string;
  state?: string;
  country?: string;
}

interface PestEntry {
  label: string;
  occurrence_count_nearby: number | null;
  risk_level: string;
}

interface PestData {
  available: boolean;
  reason?: string;
  search_radius_km?: number;
  pathogens?: Record<string, PestEntry>;
}

interface EnvironmentResponse {
  using_default_location: boolean;
  default_location_note?: string | null;
  weather: WeatherData;
  location: LocationData;
  pest_occurrences: PestData;
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function riskColor(level: string): string {
  if (level === 'High') return 'var(--danger)';
  if (level === 'Moderate') return 'var(--warning)';
  if (level === 'Low') return 'var(--success)';
  return 'var(--text-muted)';
}

function weatherRiskColor(level: string | undefined): string {
  if (level === 'High') return 'var(--danger)';
  if (level === 'Moderate') return 'var(--warning)';
  return 'var(--success)';
}

function weatherIcon(temp: number | null, rain: number | null) {
  if (rain && rain > 1) return <CloudRain size={32} color="var(--primary)" />;
  if (temp !== null && temp > 32) return <Sun size={32} color="#f59e0b" />;
  return <Cloud size={32} color="var(--text-muted)" />;
}

const fmt = (val: number | null | undefined, decimals = 1) =>
  val != null ? val.toFixed(decimals) : 'N/A';

// ─── Component ────────────────────────────────────────────────────────────────

export default function WeatherWidget() {
  const [data, setData] = useState<EnvironmentResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [geoError, setGeoError] = useState<string | null>(null);

  const SHIVAMOGGA_LAT = 13.9299;
  const SHIVAMOGGA_LON = 75.5681;

  const API_BASE = import.meta.env.VITE_API_URL || '/api/v1';

  const fetchEnvironment = (lat?: number, lon?: number) => {
    const base = API_BASE.replace(/\/$/, '');
    const effectiveLat = lat ?? SHIVAMOGGA_LAT;
    const effectiveLon = lon ?? SHIVAMOGGA_LON;
    const url = `${base}/environment?lat=${effectiveLat}&lon=${effectiveLon}`;

    setLoading(true);
    fetch(url)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json() as Promise<EnvironmentResponse>;
      })
      .then((d) => {
        setData(d);
        setLoading(false);
      })
      .catch((err) => {
        console.debug('WeatherWidget: fetch unavailable or offline', err);
        setLoading(false);
      });
  };

  const requestLiveLocation = () => {
    if (!navigator.geolocation) {
      setGeoError('Geolocation is not supported by your browser.');
      return;
    }
    setLoading(true);
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        setGeoError(null);
        fetchEnvironment(pos.coords.latitude, pos.coords.longitude);
      },
      (err) => {
        const msg =
          err.code === 1
            ? 'Browser location permission blocked. Showing Shivamogga, Karnataka (default).'
            : 'GPS signal timed out. Showing Shivamogga, Karnataka (default).';
        setGeoError(msg);
        fetchEnvironment(SHIVAMOGGA_LAT, SHIVAMOGGA_LON);
      },
      { timeout: 8000, maximumAge: 0, enableHighAccuracy: true }
    );
  };

  useEffect(() => {
    if (!navigator.geolocation) {
      setGeoError('Geolocation not supported by browser. Showing Shivamogga, Karnataka (default).');
      fetchEnvironment(SHIVAMOGGA_LAT, SHIVAMOGGA_LON);
      return;
    }

    navigator.geolocation.getCurrentPosition(
      (pos) => {
        setGeoError(null);
        fetchEnvironment(pos.coords.latitude, pos.coords.longitude);
      },
      (err) => {
        const msg =
          err.code === 1
            ? 'Location access not enabled. Showing data for Shivamogga, Karnataka (PESITM / default).'
            : 'Unable to retrieve GPS fix. Showing data for Shivamogga, Karnataka (default).';
        setGeoError(msg);
        fetchEnvironment(SHIVAMOGGA_LAT, SHIVAMOGGA_LON);
      },
      { timeout: 8000, maximumAge: 60000 }
    );
  }, []);

  // ── Render: Loading ────────────────────────────────────────────────────────
  if (loading) {
    return (
      <div className="card glass" style={{ padding: '24px', display: 'flex', alignItems: 'center', gap: 12 }}>
        <Loader2 size={20} style={{ animation: 'spin 1s linear infinite', color: 'var(--primary)' }} />
        <span style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
          Fetching real-time field environment data…
        </span>
      </div>
    );
  }

  // ── Render: Full API failure ───────────────────────────────────────────────
  if (!data) {
    return (
      <div className="card glass" style={{ padding: '20px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, color: 'var(--text-muted)' }}>
          <WifiOff size={18} />
          <span style={{ fontSize: '0.875rem' }}>Environment data unavailable. Check backend connection.</span>
        </div>
      </div>
    );
  }

  const { weather, location, pest_occurrences } = data;
  const current = weather?.current;
  const agro    = weather?.agro;

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      style={{ display: 'flex', flexDirection: 'column', gap: 16 }}
    >
      {/* ── Default location notice ────────────────────────────────────────── */}
      {(data.using_default_location || geoError) && (
        <div style={{
          background: 'rgba(245, 158, 11, 0.1)',
          border: '1px solid rgba(245, 158, 11, 0.3)',
          borderRadius: 10,
          padding: '10px 14px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: 10,
          fontSize: '0.8rem',
          color: '#f59e0b',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, flex: 1, minWidth: 220 }}>
            <MapPin size={16} />
            <span>{geoError || data.default_location_note || 'Showing field data for Shivamogga, Karnataka (default).'}</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
            <button
              onClick={requestLiveLocation}
              style={{
                background: 'var(--primary)',
                color: '#ffffff',
                border: 'none',
                borderRadius: 6,
                padding: '5px 12px',
                fontSize: '0.75rem',
                fontWeight: 600,
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: 5,
                boxShadow: '0 2px 4px rgba(0,0,0,0.15)',
              }}
              title="Request GPS fix from browser"
            >
              <Navigation size={13} />
              Detect My Live Location
            </button>
            <button
              onClick={() => { setGeoError(null); fetchEnvironment(SHIVAMOGGA_LAT, SHIVAMOGGA_LON); }}
              style={{
                background: 'rgba(255, 255, 255, 0.08)',
                color: 'var(--text-main)',
                border: '1px solid var(--border)',
                borderRadius: 6,
                padding: '5px 10px',
                fontSize: '0.75rem',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: 4,
              }}
              title="Lock location to Shivamogga (PESITM)"
            >
              <RefreshCw size={12} />
              Shivamogga (PESITM)
            </button>
          </div>
        </div>
      )}

      {/* ── Main weather + location row ────────────────────────────────────── */}
      <div className="stats-grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))' }}>

        {/* Location Card */}
        <div className="card glass" style={{ padding: '18px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
            <p className="card-label" style={{ margin: 0 }}>Field Location</p>
            <MapPin size={16} color="var(--primary)" />
          </div>
          {location?.available ? (
            <>
              <h4 style={{ fontSize: '1rem', fontWeight: 600, marginBottom: 4, color: 'var(--text-main)' }}>
                {location.district || location.state || 'India'}
              </h4>
              <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', margin: 0 }}>
                {location.state}{location.country ? `, ${location.country}` : ''}
              </p>
            </>
          ) : (
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', margin: 0 }}>
              {location?.reason || 'Location data unavailable.'}
            </p>
          )}

          {/* Season indicator */}
          {weather?.season && (
            <div style={{
              marginTop: 10,
              padding: '4px 8px',
              background: 'rgba(34, 197, 94, 0.1)',
              borderRadius: 6,
              fontSize: '0.72rem',
              color: 'var(--success)',
              fontWeight: 500,
              display: 'flex',
              alignItems: 'center',
              gap: 4,
            }}>
              <Leaf size={11} />
              {weather.season}
            </div>
          )}
        </div>

        {/* Temperature Card */}
        <div className="card glass" style={{ padding: '18px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
            <p className="card-label" style={{ margin: 0 }}>Temperature</p>
            <Thermometer size={16} color="#f59e0b" />
          </div>
          {weather?.available && current ? (
            <>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                {weatherIcon(current.temperature_c, current.rain_mm)}
                <h3 style={{ fontSize: '2rem', fontWeight: 700, margin: 0, color: 'var(--text-main)' }}>
                  {fmt(current.temperature_c, 1)}°C
                </h3>
              </div>
              <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', margin: '4px 0 0' }}>
                Feels like {fmt(current.feels_like_c)}°C
              </p>
              {agro?.soil_temperature_c != null && (
                <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', margin: '2px 0 0' }}>
                  Soil temp: {fmt(agro.soil_temperature_c)}°C
                </p>
              )}
            </>
          ) : (
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              {weather?.reason || 'Weather data unavailable.'}
            </p>
          )}
        </div>

        {/* Humidity & Rain Card */}
        <div className="card glass" style={{ padding: '18px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
            <p className="card-label" style={{ margin: 0 }}>Humidity & Rain</p>
            <Droplets size={16} color="var(--primary)" />
          </div>
          {weather?.available && current ? (
            <>
              <h3 style={{ fontSize: '1.8rem', fontWeight: 700, margin: 0, color: 'var(--text-main)' }}>
                {fmt(current.relative_humidity_pct, 0)}%
              </h3>
              <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', margin: '4px 0 0' }}>
                Relative humidity
              </p>
              <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', margin: '2px 0 0' }}>
                Rain: {fmt(current.rain_mm, 1)} mm &nbsp;|&nbsp; Wind: {fmt(current.wind_speed_kmh, 0)} km/h
              </p>
              {agro?.et0_evapotranspiration_mm != null && (
                <p style={{ fontSize: '0.73rem', color: 'var(--text-muted)', marginTop: 2 }}>
                  ET₀: {fmt(agro.et0_evapotranspiration_mm, 2)} mm/hr
                </p>
              )}
            </>
          ) : (
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              {weather?.reason || 'Humidity data unavailable.'}
            </p>
          )}
        </div>

        {/* Disease Risk from Weather */}
        <div className="card glass" style={{ padding: '18px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
            <p className="card-label" style={{ margin: 0 }}>Weather Risk Index</p>
            <AlertTriangle size={16} color="var(--warning)" />
          </div>
          {weather?.available ? (
            <>
              <h3 style={{
                fontSize: '1.5rem',
                fontWeight: 700,
                margin: '4px 0',
                color: weatherRiskColor(weather.disease_risk_from_weather),
              }}>
                {weather.disease_risk_from_weather}
              </h3>
              <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', margin: 0 }}>
                Disease risk from current weather
              </p>
              <p style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: 4 }}>
                Based on ICAR humidity/temp thresholds
              </p>
            </>
          ) : (
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              Risk assessment unavailable.
            </p>
          )}
        </div>
      </div>

      {/* ── 3-Day Forecast Row ─────────────────────────────────────────────── */}
      {weather?.available && weather.forecast_3day && weather.forecast_3day.length > 0 && (
        <div className="card glass" style={{ padding: '18px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 14 }}>
            <Cloud size={16} color="var(--primary)" />
            <h4 style={{ margin: 0, fontSize: '0.9rem', fontWeight: 600 }}>3-Day Weather Forecast</h4>
            <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginLeft: 'auto' }}>
              Source: Open-Meteo • IST
            </span>
          </div>
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
            {weather.forecast_3day.map((day) => (
              <div key={day.day} style={{
                flex: '1 1 80px',
                background: 'rgba(255,255,255,0.04)',
                borderRadius: 10,
                padding: '10px 12px',
                textAlign: 'center',
              }}>
                <p style={{ fontSize: '0.72rem', color: 'var(--text-muted)', margin: '0 0 4px' }}>
                  Day +{day.day}
                </p>
                <p style={{ fontSize: '1rem', fontWeight: 600, margin: '0 0 2px', color: 'var(--text-main)' }}>
                  {fmt(day.temp_max, 0)}° / {fmt(day.temp_min, 0)}°C
                </p>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 4 }}>
                  <Droplets size={11} color="var(--primary)" />
                  <p style={{ fontSize: '0.72rem', color: 'var(--text-muted)', margin: 0 }}>
                    {fmt(day.precip_mm, 1)} mm
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── GBIF Pest Occurrence Panel ─────────────────────────────────────── */}
      <div className="card glass" style={{ padding: '18px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 14 }}>
          <Bug size={16} color="var(--warning)" />
          <h4 style={{ margin: 0, fontSize: '0.9rem', fontWeight: 600 }}>
            Nearby Crop Pest Occurrences (India)
          </h4>
          {pest_occurrences?.available && (
            <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginLeft: 'auto' }}>
              Source: GBIF • {pest_occurrences.search_radius_km}km radius
            </span>
          )}
        </div>

        {pest_occurrences?.available && pest_occurrences.pathogens ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {Object.entries(pest_occurrences.pathogens).map(([species, info]) => (
              <div key={species} style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                padding: '8px 12px',
                background: 'rgba(255,255,255,0.03)',
                borderRadius: 8,
                borderLeft: `3px solid ${riskColor(info.risk_level)}`,
              }}>
                <div>
                  <p style={{ fontSize: '0.8rem', fontWeight: 600, margin: 0, color: 'var(--text-main)' }}>
                    {info.label}
                  </p>
                  <p style={{ fontSize: '0.7rem', color: 'var(--text-muted)', margin: '2px 0 0', fontStyle: 'italic' }}>
                    {species}
                  </p>
                </div>
                <div style={{ textAlign: 'right' }}>
                  <span style={{
                    fontSize: '0.75rem',
                    fontWeight: 600,
                    color: riskColor(info.risk_level),
                    background: `${riskColor(info.risk_level)}18`,
                    padding: '3px 8px',
                    borderRadius: 6,
                  }}>
                    {info.risk_level}
                  </span>
                  {info.occurrence_count_nearby != null && (
                    <p style={{ fontSize: '0.65rem', color: 'var(--text-muted)', margin: '2px 0 0' }}>
                      {info.occurrence_count_nearby} records
                    </p>
                  )}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', margin: 0 }}>
            {pest_occurrences?.reason || 'Pest occurrence data unavailable.'}
          </p>
        )}
      </div>
    </motion.div>
  );
}
