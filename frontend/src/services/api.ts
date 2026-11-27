import axios, { AxiosInstance, AxiosError, InternalAxiosRequestConfig } from 'axios';
import { LeafResult, FieldResult, ScanHistoryItem } from '../types';

export interface ApiError {
  message?: string;
  detail?: string;
}

const getBaseUrl = (): string => {
  const envUrl = import.meta.env.VITE_API_URL;
  if (envUrl) {
    const trimmed = envUrl.trim().replace(/\/+$/, '');
    return trimmed.endsWith('/api/v1') ? trimmed : `${trimmed}/api/v1`;
  }
  // In production builds (Vercel), default directly to our live Render backend:
  if (import.meta.env.PROD) {
    return 'https://crop-disease-backend-p5ur.onrender.com/api/v1';
  }
  return '/api/v1';
};

const API_BASE_URL = getBaseUrl();

// Default timeout: 90s for leaf/health calls.
// Field analysis on CPU can take 2-3 minutes — overridden per-request below.
const apiClient: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: 90000, // 90 seconds default
});

apiClient.interceptors.response.use(
  (response) => response,
  async (error: AxiosError<ApiError>) => {
    const config = error.config as InternalAxiosRequestConfig & { __isRetryRequest?: boolean; retryCount?: number } | undefined;

    // Handle timeout differently for field analysis
    if (error.code === 'ECONNABORTED' && config?.url?.includes('/analyze/field')) {
      console.error('[Field Analysis] Request timeout - field analysis takes 2-3 minutes on CPU');
      const message = 'Field analysis is taking longer than expected. This is normal on CPU processing. Please wait up to 5 minutes for results.';
      return Promise.reject({ ...error, message });
    }

    // Only retry on genuine network failures (no response), not on timeouts or HTTP errors.
    const isNetworkError = !error.response && error.code !== 'ECONNABORTED';
    const isSafeRead = config?.method?.toUpperCase() === 'GET';
    if (isNetworkError && isSafeRead && config && !config.__isRetryRequest && (config.retryCount ?? 0) < 2) {
      config.__isRetryRequest = true;
      config.retryCount = (config.retryCount ?? 0) + 1;

      console.warn(`[Network] Retrying request (${config.retryCount}/2)...`);

      const backoff = Math.pow(2, config.retryCount) * 1000;
      await new Promise((resolve) => setTimeout(resolve, backoff));

      return apiClient(config);
    }

    const message = error.response?.data?.detail || error.message || 'Connection lost. Please check your internet.';
    console.error(`[API Error] ${config?.url}:`, message);
    return Promise.reject({ ...error, message });
  }
);

export const analyzeLeaf = async (formData: FormData): Promise<LeafResult> => {
  const response = await apiClient.post<LeafResult>('/analyze/leaf', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return response.data;
};

export const analyzeField = async (formData: FormData): Promise<FieldResult> => {
  // Field inference on CPU can take 2-3 minutes — use a generous 5-minute timeout.
  const response = await apiClient.post<FieldResult>('/analyze/field', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 300000, // 5 minutes
  });
  return response.data;
};

export interface FieldStreamProgress {
  stage: string;
  percent: number;
  message: string;
}

export const analyzeFieldStream = async (
  formData: FormData,
  onProgress?: (progress: FieldStreamProgress) => void
): Promise<FieldResult> => {
  const response = await fetch(`${API_BASE_URL}/analyze/field/stream`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorText = await response.text();
    let errorMsg = `Server error (${response.status})`;
    try {
      const json = JSON.parse(errorText);
      errorMsg = json.detail || json.message || errorMsg;
    } catch {
      // Preserve the status-based message when the server response is not JSON.
    }
    throw new Error(errorMsg);
  }
  if (!response.body) throw new Error('Response body is null');

  const reader = response.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buffer = '';
  let finalResult: FieldResult | null = null;

  const consumeEvent = (eventBlock: string) => {
    let eventName = 'message';
    const dataLines: string[] = [];
    for (const line of eventBlock.replace(/\r/g, '').split('\n')) {
      if (line.startsWith('event:')) eventName = line.slice(6).trim();
      else if (line.startsWith('data:')) dataLines.push(line.slice(5).trim());
    }
    if (dataLines.length === 0) return;
    const payload = JSON.parse(dataLines.join('\n'));
    if (eventName === 'progress') onProgress?.(payload as FieldStreamProgress);
    else if (eventName === 'complete') finalResult = payload as FieldResult;
    else if (eventName === 'error') throw new Error(payload.error || 'Inference error');
  };

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const blocks = buffer.split(/\r?\n\r?\n/);
    buffer = blocks.pop() || '';
    blocks.forEach(consumeEvent);
  }
  buffer += decoder.decode();
  if (buffer.trim()) consumeEvent(buffer);
  if (!finalResult) throw new Error('The analysis stream ended without a completion acknowledgement.');
  return finalResult;
};

export interface ScanHistoryDetail extends ScanHistoryItem {
  full_results: Record<string, unknown>;
}

export const getHistory = async (): Promise<ScanHistoryItem[]> => {
  const response = await apiClient.get<ScanHistoryItem[]>('/history');
  return response.data;
};

export const getScanDetail = async (scanId: string): Promise<ScanHistoryDetail> => {
  const response = await apiClient.get<ScanHistoryDetail>(`/history/${encodeURIComponent(scanId)}`);
  return response.data;
};

export const consultField = async (payload: import('../types').FieldConsultPayload): Promise<import('../types').FieldConsultResponse> => {
  const response = await apiClient.post<import('../types').FieldConsultResponse>('/analyze/field/consult', payload);
  return response.data;
};

export const getAiStatus = async (): Promise<import('../types').AiProviderStatus> => {
  const response = await apiClient.get<import('../types').AiProviderStatus>('/agronomy/ai-status');
  return response.data;
};

export default apiClient;
