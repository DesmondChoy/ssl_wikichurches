// API Response Types

export type AnalysisMetric = 'iou' | 'coverage' | 'mse' | 'kl' | 'emd';
export type DashboardMetric = Exclude<AnalysisMetric, 'coverage'>;
export type CompareVariantId = 'frozen' | 'linear_probe' | 'lora' | 'full';

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
  mse: number;
  kl: number;
  emd: number;
  attention_area: number;
  annotation_area: number;
  method?: string;
}

export type MetricDirection = 'higher' | 'lower';

export interface ImageMetricDescriptor {
  key: AnalysisMetric;
  label: string;
  direction: MetricDirection;
  default_enabled: boolean;
  percentile_dependent: boolean;
}

export interface ImageMetricSelection {
  mode: 'union' | 'bbox';
  bbox_index: number | null;
  bbox_label: string | null;
}

export interface ImageLayerMetricPoint {
  layer: number;
  layer_key: string;
  values: Record<string, number | null>;
}

export interface ImageLayerProgression {
  image_id: string;
  model: string;
  method: string;
  percentile: number;
  selection: ImageMetricSelection;
  metrics: ImageMetricDescriptor[];
  layers: ImageLayerMetricPoint[];
}

export type RankingMode = 'default_method' | 'best_available';

export interface LeaderboardEntry {
  rank: number;
  model: string;
  metric: DashboardMetric;
  score: number;
  best_layer: string;
  method_used: string;
}

export interface LayerProgression {
  model: string;
  metric: DashboardMetric;
  percentile: number;
  layers: string[];
  scores: number[];
  best_layer: string;
  best_score: number;
  method?: string;
}

export interface StyleBreakdown {
  model: string;
  layer: string;
  percentile: number;
  styles: Record<string, number>;
  style_counts: Record<string, number>;
}

export interface FeatureIoUEntry {
  feature_label: number;
  feature_name: string;
  mean_iou: number;
  std_iou: number;
  bbox_count: number;
}

export interface FeatureBreakdown {
  model: string;
  layer: string;
  percentile: number;
  features: FeatureIoUEntry[];
  total_feature_types: number;
}

export interface ModelComparison {
  image_id: string;
  models: string[];
  layer: string;
  percentile: number;
  selection: ImageMetricSelection;
  results: IoUResult[];
  heatmap_urls: Record<string, string>;
  unavailable_models: Record<string, string>;
}

export interface ComparisonVariant {
  model_key: string;
  strategy?: string | null;
  label: string;
  available: boolean;
  url: string | null;
}

export interface VariantComparison {
  image_id: string;
  model: string;
  layer: string;
  method: string;
  show_bboxes?: boolean;
  left: ComparisonVariant;
  right: ComparisonVariant;
  note: string;
}

export interface AllModelsSummaryModelEntry {
  rank: number;
  best_layer: string;
  best_score: number;
  method_used: string;
  layer_progression: Record<string, number>;
}

export interface AllModelsSummary {
  percentile: number;
  metric: DashboardMetric;
  ranking_mode: RankingMode | null;
  method: string | null;
  excluded_models: string[];
  models: Record<string, AllModelsSummaryModelEntry>;
  leaderboard: LeaderboardEntry[];
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

export interface Q2SummaryRow {
  model_name: string;
  strategy_id: string;
  metric: AnalysisMetric;
  label: string;
  direction: MetricDirection;
  percentile_dependent: boolean;
  percentile: number | null;
  method: string;
  frozen_mean: number;
  finetuned_mean: number;
  mean_delta: number;
  std_delta: number;
  delta_ci_lower: number;
  delta_ci_upper: number;
  cohens_d: number;
  p_value: number;
  corrected_p_value: number | null;
  significant: boolean;
  test_name: string;
  num_images: number;
}

export interface Q2StrategyComparison {
  model_name: string;
  metric: AnalysisMetric;
  percentile: number | null;
  strategy_a: string;
  strategy_b: string;
  mean_delta_difference: number;
  cohens_d: number;
  p_value: number;
  corrected_p_value: number | null;
  significant: boolean;
  test_name: string;
}

export interface Q2SummaryResponse {
  metric: AnalysisMetric;
  label: string;
  direction: MetricDirection;
  percentile_dependent: boolean;
  selected_percentile: number | null;
  analyzed_layer: number;
  timestamp: string | null;
  rows: Q2SummaryRow[];
  strategy_comparisons: Q2StrategyComparison[];
}

// App State Types

export type HeatmapStyle = 'smooth' | 'squares' | 'circles';

export interface ViewSettings {
  model: string;
  layer: number;
  method: string;
  percentile: number;
  showBboxes: boolean;
  heatmapOpacity: number;
  heatmapStyle: HeatmapStyle;
}

// Models API Response
export interface ModelsResponse {
  models: string[];
  num_layers: number;  // Legacy: global default
  num_layers_per_model: Record<string, number>;  // Per-model layer counts
  methods: Record<string, string[]>;
  default_methods: Record<string, string>;
}

// Raw attention response for client-side rendering
export interface RawAttentionResponse {
  attention: number[];
  shape: [number, number];
  min_value: number;
  max_value: number;
}
