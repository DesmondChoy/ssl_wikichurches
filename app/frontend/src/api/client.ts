/**
 * API client for the SSL Attention Visualization backend.
 */

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

class APIError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = 'APIError';
    this.status = status;
  }
}

async function fetchJSON<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const url = `${API_BASE}${endpoint}`;

  const response = await fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new APIError(response.status, error.detail || 'Request failed');
  }

  return response.json();
}

// Images API
export const imagesAPI = {
  list: (params?: { style?: string; limit?: number; offset?: number }) => {
    const query = new URLSearchParams();
    if (params?.style) query.set('style', params.style);
    if (params?.limit) query.set('limit', String(params.limit));
    if (params?.offset) query.set('offset', String(params.offset));
    const queryStr = query.toString();
    return fetchJSON<import('../types').ImageListItem[]>(`/images${queryStr ? `?${queryStr}` : ''}`);
  },

  getStyles: () => fetchJSON<string[]>('/images/styles'),

  getDetail: (imageId: string) =>
    fetchJSON<import('../types').ImageDetail>(`/images/${imageId}`),

  getImageUrl: (imageId: string, size?: number) => {
    const query = size ? `?size=${size}` : '';
    return `${API_BASE}/images/${imageId}/file${query}`;
  },

  getThumbnailUrl: (imageId: string) =>
    `${API_BASE}/images/${imageId}/thumbnail`,

  getWithBboxesUrl: (imageId: string) =>
    `${API_BASE}/images/${imageId}/with_bboxes`,
};

// Similarity response type
export interface SimilarityResponse {
  similarity: number[];
  patch_grid: [number, number];
  min_similarity: number;
  max_similarity: number;
  bbox_patch_indices: number[];
}

// Attention API
export const attentionAPI = {
  getHeatmapUrl: (imageId: string, model: string, layer: number, method?: string) => {
    const params = new URLSearchParams({ model, layer: String(layer) });
    if (method) params.set('method', method);
    return `${API_BASE}/attention/${imageId}/heatmap?${params}`;
  },

  getOverlayUrl: (imageId: string, model: string, layer: number, showBboxes = false, method?: string) => {
    const params = new URLSearchParams({
      model,
      layer: String(layer),
      show_bboxes: String(showBboxes),
    });
    if (method) params.set('method', method);
    return `${API_BASE}/attention/${imageId}/overlay?${params}`;
  },

  getLayerUrls: (imageId: string, model: string, showBboxes = false, method?: string) => {
    const params = new URLSearchParams({
      model,
      show_bboxes: String(showBboxes),
    });
    if (method) params.set('method', method);
    return fetchJSON<{
      image_id: string;
      model: string;
      method: string;
      show_bboxes: boolean;
      layers: Record<string, string>;
    }>(`/attention/${imageId}/layers?${params}`);
  },

  getModels: () => fetchJSON<import('../types').ModelsResponse>('/attention/models'),

  getRawAttention: (imageId: string, model: string, layer: number, method?: string) => {
    const params = new URLSearchParams({ model, layer: String(layer) });
    if (method) params.set('method', method);
    return fetchJSON<import('../types').RawAttentionResponse>(
      `/attention/${imageId}/raw?${params}`
    );
  },

  getSimilarity: (
    imageId: string,
    bbox: { left: number; top: number; width: number; height: number; label?: string },
    model: string,
    layer: number
  ) =>
    fetchJSON<SimilarityResponse>(
      `/attention/${imageId}/similarity?model=${model}&layer=${layer}`,
      {
        method: 'POST',
        body: JSON.stringify(bbox),
      }
    ),
};

// Metrics API
export const metricsAPI = {
  getLeaderboard: (percentile = 90) =>
    fetchJSON<import('../types').LeaderboardEntry[]>(`/metrics/leaderboard?percentile=${percentile}`),

  getSummary: () => fetchJSON<{
    models: Record<string, {
      best_layer: string;
      best_iou: number;
      layer_progression: Record<string, number>;
    }>;
    leaderboard: Array<{ model: string; best_iou: number }>;
  }>('/metrics/summary'),

  getImageMetrics: (imageId: string, model: string, layer: number, percentile = 90) =>
    fetchJSON<import('../types').IoUResult>(
      `/metrics/${imageId}?model=${model}&layer=${layer}&percentile=${percentile}`
    ),

  getImageMetricsAllModels: (imageId: string, layer: number, percentile = 90) =>
    fetchJSON<{
      image_id: string;
      layer: string;
      percentile: number;
      models: Record<string, import('../types').IoUResult>;
    }>(`/metrics/${imageId}/all_models?layer=${layer}&percentile=${percentile}`),

  getLayerProgression: (model: string, percentile = 90) =>
    fetchJSON<import('../types').LayerProgression>(
      `/metrics/model/${model}/progression?percentile=${percentile}`
    ),

  getStyleBreakdown: (model: string, layer: number, percentile = 90) =>
    fetchJSON<import('../types').StyleBreakdown>(
      `/metrics/model/${model}/style_breakdown?layer=${layer}&percentile=${percentile}`
    ),

  getFeatureBreakdown: (
    model: string,
    layer: number,
    percentile = 90,
    sortBy: 'mean_iou' | 'bbox_count' | 'feature_name' | 'feature_label' = 'mean_iou',
    minCount = 0
  ) =>
    fetchJSON<import('../types').FeatureBreakdown>(
      `/metrics/model/${model}/feature_breakdown?layer=${layer}&percentile=${percentile}&sort_by=${sortBy}&min_count=${minCount}`
    ),

  getAggregate: (model: string, layer: number, percentile = 90) =>
    fetchJSON<{
      model: string;
      layer: string;
      percentile: number;
      mean_iou: number;
      std_iou: number;
      median_iou: number;
      mean_coverage: number;
      num_images: number;
    }>(`/metrics/model/${model}/aggregate?layer=${layer}&percentile=${percentile}`),
};

// Comparison API
export const comparisonAPI = {
  compareModels: (imageId: string, models: string[], layer: number, percentile = 90) => {
    const modelsParam = models.map(m => `models=${m}`).join('&');
    return fetchJSON<import('../types').ModelComparison>(
      `/compare/models?image_id=${imageId}&${modelsParam}&layer=${layer}&percentile=${percentile}`
    );
  },

  compareFrozenVsFinetuned: (imageId: string, model: string, layer: number) =>
    fetchJSON<{
      image_id: string;
      model: string;
      layer: string;
      frozen: { available: boolean; url: string | null };
      finetuned: { available: boolean; url: string | null; note: string };
    }>(`/compare/frozen_vs_finetuned?image_id=${imageId}&model=${model}&layer=${layer}`),

  compareLayers: (imageId: string, model: string, percentile = 90) =>
    fetchJSON<import('../types').LayerComparison>(
      `/compare/layers?image_id=${imageId}&model=${model}&percentile=${percentile}`
    ),

  getAllModelsSummary: (percentile = 90) =>
    fetchJSON<{
      percentile: number;
      models: Record<string, {
        rank: number;
        best_iou: number;
        best_layer: string;
        layer_progression: Record<string, number>;
      }>;
      leaderboard: import('../types').LeaderboardEntry[];
    }>(`/compare/all_models_summary?percentile=${percentile}`),
};

export { APIError };
