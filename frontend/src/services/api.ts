import axios, { AxiosInstance, AxiosError, InternalAxiosRequestConfig } from 'axios';
import { LeafResult, FieldResult, ScanHistoryItem } from '../types';

export interface ApiError {
  message?: string;
  detail?: string;
}

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api/v1';

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
    if (isNetworkError && config && !config.__isRetryRequest && (config.retryCount ?? 0) < 2) {
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
  try {
    const response = await apiClient.post<FieldResult>('/analyze/field', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 300000, // 5 minutes
    });
    return response.data;
  } catch (error: any) {
    // If proxy fails with connection reset, try direct connection to backend
    if (error.code === 'ECONNRESET' || error.message?.includes('ECONNRESET')) {
      console.warn('Proxy connection reset, trying direct backend connection...');
      const directClient = axios.create({
        baseURL: 'http://localhost:8000/api/v1',
        timeout: 300000,
      });
      
      const response = await directClient.post<FieldResult>('/analyze/field', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      return response.data;
    }
    throw error;
  }
};

export const getHistory = async (): Promise<ScanHistoryItem[]> => {
  const response = await apiClient.get<ScanHistoryItem[]>('/history');
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
