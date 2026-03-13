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
  getLeaderboard: (percentile = 90, metric: import('../types').DashboardMetric = 'iou') =>
    fetchJSON<import('../types').LeaderboardEntry[]>(
      `/metrics/leaderboard?percentile=${percentile}&metric=${metric}`
    ),

  getSummary: () => fetchJSON<{
    models: Record<string, {
      best_layer: string;
      best_iou: number;
      layer_progression: Record<string, number>;
    }>;
    leaderboard: Array<{ model: string; best_iou: number }>;
  }>('/metrics/summary'),

  getImageLayerProgression: (
    imageId: string,
    model: string,
    percentile = 90,
    method?: string,
    bboxIndex?: number | null
  ) => {
    const params = new URLSearchParams({ model, percentile: String(percentile) });
    if (method) params.set('method', method);
    if (bboxIndex !== null && bboxIndex !== undefined) params.set('bbox_index', String(bboxIndex));
    return fetchJSON<import('../types').ImageLayerProgression>(
      `/metrics/${imageId}/progression?${params}`
    );
  },

  getImageMetricsAllModels: (imageId: string, layer: number, percentile = 90, method?: string) => {
    const params = new URLSearchParams({ layer: String(layer), percentile: String(percentile) });
    if (method) params.set('method', method);
    return fetchJSON<{
      image_id: string;
      layer: string;
      percentile: number;
      models: Record<string, import('../types').IoUResult>;
    }>(`/metrics/${imageId}/all_models?${params}`);
  },

  getLayerProgression: (
    model: string,
    percentile = 90,
    metric: import('../types').DashboardMetric = 'iou',
    method?: string
  ) => {
    const params = new URLSearchParams({ percentile: String(percentile), metric });
    if (method) params.set('method', method);
    return fetchJSON<import('../types').LayerProgression>(
      `/metrics/model/${model}/progression?${params}`
    );
  },

  getStyleBreakdown: (model: string, layer: number, percentile = 90, method?: string) => {
    const params = new URLSearchParams({ layer: String(layer), percentile: String(percentile) });
    if (method) params.set('method', method);
    return fetchJSON<import('../types').StyleBreakdown>(
      `/metrics/model/${model}/style_breakdown?${params}`
    );
  },

  getFeatureBreakdown: (
    model: string,
    layer: number,
    percentile = 90,
    sortBy: 'mean_iou' | 'bbox_count' | 'feature_name' | 'feature_label' = 'mean_iou',
    minCount = 0,
    method?: string
  ) => {
    const params = new URLSearchParams({
      layer: String(layer),
      percentile: String(percentile),
      sort_by: sortBy,
      min_count: String(minCount),
    });
    if (method) params.set('method', method);
    return fetchJSON<import('../types').FeatureBreakdown>(
      `/metrics/model/${model}/feature_breakdown?${params}`
    );
  },

  getAggregate: (model: string, layer: number, percentile = 90, method?: string) => {
    const params = new URLSearchParams({ layer: String(layer), percentile: String(percentile) });
    if (method) params.set('method', method);
    return fetchJSON<{
      model: string;
      layer: string;
      percentile: number;
      mean_iou: number;
      std_iou: number;
      median_iou: number;
      mean_coverage: number;
      mean_mse: number;
      std_mse: number;
      median_mse: number;
      mean_kl: number;
      std_kl: number;
      median_kl: number;
      num_images: number;
    }>(`/metrics/model/${model}/aggregate?${params}`);
  },

  getQ2Summary: (params?: { percentile?: number; model?: string; strategy?: string }) => {
    const query = new URLSearchParams();
    if (params?.percentile !== undefined) query.set('percentile', String(params.percentile));
    if (params?.model) query.set('model', params.model);
    if (params?.strategy) query.set('strategy', params.strategy);
    const queryStr = query.toString();
    return fetchJSON<import('../types').Q2SummaryResponse>(
      `/metrics/q2_summary${queryStr ? `?${queryStr}` : ''}`
    );
  },

  getBboxMetrics: (
    imageId: string,
    model: string,
    layer: number,
    bboxIndex: number,
    percentile = 90,
    method?: string
  ) => {
    const params = new URLSearchParams({
      model,
      layer: String(layer),
      percentile: String(percentile),
    });
    if (method) params.set('method', method);
    return fetchJSON<import('../types').IoUResult>(
      `/metrics/${imageId}/bbox/${bboxIndex}?${params}`
    );
  },
};

// Comparison API
export const comparisonAPI = {
  compareModels: (
    imageId: string,
    models: string[],
    layer: number,
    percentile = 90,
    method?: string
  ) => {
    const params = new URLSearchParams({
      image_id: imageId,
      layer: String(layer),
      percentile: String(percentile),
    });
    for (const model of models) {
      params.append('models', model);
    }
    if (method) {
      params.set('method', method);
    }
    return fetchJSON<import('../types').ModelComparison>(
      `/compare/models?${params.toString()}`
    );
  },

  compareFrozenVsFinetuned: (
    imageId: string,
    model: string,
    layer: number,
    strategy?: string,
    showBboxes = true
  ) => {
    const query = new URLSearchParams({ image_id: imageId, model, layer: String(layer) });
    if (strategy) query.set('strategy', strategy);
    query.set('show_bboxes', String(showBboxes));
    return fetchJSON<{
      image_id: string;
      model: string;
      strategy?: string | null;
      layer: string;
      show_bboxes?: boolean;
      frozen: { available: boolean; url: string | null };
      finetuned: { available: boolean; url: string | null; note: string };
    }>(`/compare/frozen_vs_finetuned?${query}`);
  },

  compareLayers: (imageId: string, model: string, percentile = 90) =>
    fetchJSON<import('../types').LayerComparison>(
      `/compare/layers?image_id=${imageId}&model=${model}&percentile=${percentile}`
    ),

  getAllModelsSummary: (
    percentile = 90,
    metric: import('../types').DashboardMetric = 'iou'
  ) =>
    fetchJSON<import('../types').AllModelsSummary>(
      `/compare/all_models_summary?percentile=${percentile}&metric=${metric}`
    ),
};

export { APIError };
