export interface Toast {
  id: number;
  message: string;
  type: 'success' | 'error' | 'warning' | 'info';
}

export interface BackendHealth {
  status: 'healthy' | 'degraded';
  version: string;
  device: string;
  models_loaded: boolean;
  weights_loaded: boolean;
  database: 'connected' | 'error';
}

export type ScanType = 'leaf' | 'field';

export interface LeafResult {
  type?: string;
  disease_type: string;
  severity: number;
  confidence: number;
  withheld?: boolean;
  temporal_available?: boolean;
  status: string;
  treatment: string;
  alternatives?: Array<{ disease: string; confidence: number }>;
  scan_id?: string;
  disclaimer?: string;
  farmer_friendly?: FarmerFriendly;
  quality?: {
    quality: {
      quality_label: string;
      iqi_score: number;
      issues: string[];
      suggestions: string[];
      proceed_to_inference: boolean;
    };
    confidence: {
      confidence: number;
      level: string;
      tone: string;
      message: string;
    };
    reliability: {
      score: number;
      label: string;
    };
    recommendation: string;
    disease_label: string;
  };
  reliability?: {
    score: number;
    is_reliable: boolean;
    rejection_reason?: string;
  };
  ood?: {
    is_ood: boolean;
    ood_score: number;
    reason?: string;
  };
  recommendation?: string;
}

export interface FieldResult {
  disease_type: string;
  average_severity: number;
  infection_rate: number;
  previous_infection_rate?: number | null;
  spread_risk: string;
  treatment: string;
  veg_coverage: number;
  withheld?: boolean;
  temporal_available?: boolean;
  grid_data: GridCell[];
  progression_data: Array<{ day: string; actual: number | null; predicted: number; simulated?: boolean; point_type?: string }>;
  spectral_radar: Array<{ subject: string; A: number; B: number }>;
  ngvc_data?: Array<{ intensity: number; value: number }>;
  heatmap_base64?: string;
  percentage_healthy_area?: number;
  percentage_infected_area?: number;
  disease_increasing_pct?: number;
  clusters?: Cluster[];
  confidence: number;
  spatial_morans_i?: number;
  alternatives?: Array<{ disease: string; confidence: number }>;
  scan_id?: string;
  farmer_friendly?: FarmerFriendly & { field_health_message?: string };
}

export interface Cluster {
  id: number;
  disease: string;
  coverage_pct: number;
}

export interface GridCell {
  x_percent: number;
  y_percent: number;
  w_percent: number;
  h_percent: number;
  severity: number;
  disease: string;
}

export interface FarmerFriendly {
  simple_label: string;
  certainty: 'High Certainty' | 'Medium Certainty' | 'Low Certainty';
  what_is_it: string;
  what_to_do: string;
  field_health_message?: string;
}

export interface ScanHistoryItem {
  id: string;
  scan_type: ScanType;
  filename: string;
  disease_type: string;
  severity: number;
  confidence: number;
  spread_risk?: string;
  treatment: string;
  timestamp: string;
}

export type FarmerView = 'home' | 'result' | 'history';

export type Language = 'en' | 'hi' | 'kn' | 'te' | 'ta';

export interface OfflineQueueItem {
  id: number;
  type: ScanType;
  name: string;
  data: string;
}

export interface FieldConsultPayload {
  question: string;
  disease_type: string;
  average_severity: number;
  infection_rate: number;
  spread_risk: string;
  clusters?: Cluster[];
  vegetation_coverage?: number;
  land_area?: number;
  area_unit?: string;
  history?: Array<{ role: string; content: string }>;
}

export interface FieldConsultResponse {
  answer: string;
  provider: string;
  model: string;
  grounding_source: string;
  field_telemetry?: {
    infection_rate: number;
    average_severity: number;
    spread_risk: string;
    cluster_count: number;
  };
  timestamp: string;
}

export interface AiProviderStatus {
  active_provider: string;
  model_id: string;
  display_name: string;
  gemini_configured: boolean;
  openai_configured: boolean;
  rag_knowledge_base: string;
  capabilities: string[];
}
