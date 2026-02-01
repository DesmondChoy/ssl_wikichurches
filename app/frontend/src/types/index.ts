// API Response Types

export interface BoundingBox {
  left: number;
  top: number;
  width: number;
  height: number;
  label: number;
  label_name: string | null;
}

export interface ImageAnnotation {
  image_id: string;
  styles: string[];
  style_names: string[];
  num_bboxes: number;
  bboxes: BoundingBox[];
}

export interface ImageListItem {
  image_id: string;
  thumbnail_url: string;
  styles: string[];
  style_names: string[];
  num_bboxes: number;
}

export interface ImageDetail {
  image_id: string;
  image_url: string;
  thumbnail_url: string;
  annotation: ImageAnnotation;
  available_models: string[];
}

export interface IoUResult {
  image_id: string;
  model: string;
  layer: string;
  percentile: number;
  iou: number;
  coverage: number;
  attention_area: number;
  annotation_area: number;
}

export interface LeaderboardEntry {
  rank: number;
  model: string;
  best_iou: number;
  best_layer: string;
}

export interface LayerProgression {
  model: string;
  percentile: number;
  layers: string[];
  ious: number[];
  best_layer: string;
  best_iou: number;
}

export interface StyleBreakdown {
  model: string;
  layer: string;
  percentile: number;
  styles: Record<string, number>;
  style_counts: Record<string, number>;
}

export interface ModelComparison {
  image_id: string;
  models: string[];
  layer: string;
  percentile: number;
  results: IoUResult[];
  heatmap_urls: Record<string, string>;
}

export interface LayerComparison {
  image_id: string;
  model: string;
  percentile: number;
  layers: Array<{
    layer: number;
    layer_key: string;
    iou: number;
    coverage: number;
    heatmap_url: string | null;
  }>;
  best_layer: number;
  best_iou: number;
}

// App State Types

export interface ViewSettings {
  model: string;
  layer: number;
  method: string;
  percentile: number;
  showBboxes: boolean;
  heatmapOpacity: number;
}

// Models API Response
export interface ModelsResponse {
  models: string[];
  num_layers: number;  // Legacy: global default
  num_layers_per_model: Record<string, number>;  // Per-model layer counts
  methods: Record<string, string[]>;
  default_methods: Record<string, string>;
}
