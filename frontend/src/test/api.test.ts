import { AxiosError } from 'axios';
import { afterEach, describe, expect, it, vi } from 'vitest';
import apiClient, { analyzeFieldStream } from '../services/api';

const originalAdapter = apiClient.defaults.adapter;

afterEach(() => {
  apiClient.defaults.adapter = originalAdapter;
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('analysis request reliability', () => {
  it('does not automatically retry a POST after a network failure', async () => {
    const adapter = vi.fn((config) => Promise.reject(new AxiosError('Network Error', 'ERR_NETWORK', config)));
    apiClient.defaults.adapter = adapter;

    await expect(apiClient.post('/analyze/field', new FormData())).rejects.toThrow();
    expect(adapter).toHaveBeenCalledTimes(1);
  });

  it('does not replay a failed stream as a second field analysis', async () => {
    const post = vi.spyOn(apiClient, 'post').mockResolvedValue({ data: {} });
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Network failure')));

    await expect(analyzeFieldStream(new FormData())).rejects.toThrow('Network failure');
    expect(fetch).toHaveBeenCalledTimes(1);
    expect(post).not.toHaveBeenCalled();
  });
});
