import { isAnalysisMetric, isCompareVariantId } from './metricMetadata';
import { parseImageDetailMode } from './imageDetailModes';
import { Q3_DEFAULTS, Q3_PRIMARY_MODELS } from './q3Scope';
import type { AnalysisMetric, CompareVariantId, ImageDetailMode } from '../types';

export type Q3ReportView = 'head-ranking' | 'head-feature-matrix' | 'frozen-delta';

export interface ImageDetailQ3State {
  model: string;
  variant: CompareVariantId;
  layer: number;
  head: number | null;
  metric: AnalysisMetric;
  mode: ImageDetailMode;
  showBboxes: boolean;
  bboxIndex: number | null;
  featureLabel: number | null;
  featureName: string | null;
}

export interface AdvancedQ3WorkspaceState {
  primaryModel: string;
  secondaryModel: string;
  variant: CompareVariantId;
  layer: number;
  metric: AnalysisMetric;
  percentile: number;
  head: number | null;
  featureLabel: number | null;
}

export interface Q3ReportState {
  view: Q3ReportView;
  model: string;
  variant: CompareVariantId;
  layer: number;
  metric: AnalysisMetric;
  percentile: number;
  head: number | null;
  featureLabel: number | null;
}

const Q3_REPORT_DEFAULTS: Q3ReportState = {
  view: 'head-ranking',
  model: 'dinov3',
  variant: 'frozen',
  layer: 10,
  metric: 'iou',
  percentile: 90,
  head: null,
  featureLabel: null,
};

export function parseImageDetailQ3State(
  searchParams: URLSearchParams,
  options?: {
    maxLayer?: number;
    numHeads?: number;
  },
): ImageDetailQ3State {
  const model = parseQ3Model(searchParams.get('model'));
  const variant = parseQ3Variant(searchParams.get('variant'));
  const maxLayer = options?.maxLayer ?? Q3_DEFAULTS.layer;
  const numHeads = options?.numHeads ?? 12;

  return {
    model,
    variant,
    layer: clampNumber(parseOptionalNumber(searchParams.get('layer')) ?? Q3_DEFAULTS.layer, 0, maxLayer),
    head: parseHeadParam(searchParams.get('head'), numHeads),
    metric: parseQ3Metric(searchParams.get('metric')),
    mode: parseImageDetailMode(searchParams.get('mode')),
    showBboxes: parseBooleanParam(searchParams.get('show_bboxes'), Q3_DEFAULTS.showBboxes),
    bboxIndex: parseOptionalNumber(searchParams.get('bbox_index')),
    featureLabel: parseOptionalNumber(searchParams.get('feature_label')),
    featureName: searchParams.get('feature_name'),
  };
}

export function getQ3ViewerModel(model: string, variant: CompareVariantId): string {
  return variant === 'frozen' ? model : `${model}_finetuned_${variant}`;
}

export function createImageDetailQ3SearchParams(
  state: ImageDetailQ3State,
  existing?: URLSearchParams,
): URLSearchParams {
  const next = new URLSearchParams(existing);
  next.set('tab', 'q3');
  next.set('mode', state.mode);
  next.set('model', state.model);
  next.set('variant', state.variant);
  next.set('layer', String(state.layer));
  next.set('head', state.head === null ? 'all' : String(state.head));
  next.set('metric', state.metric);
  next.set('show_bboxes', String(state.showBboxes));

  setOptionalNumberParam(next, 'bbox_index', state.bboxIndex);
  setOptionalNumberParam(next, 'feature_label', state.featureLabel);
  setOptionalStringParam(next, 'feature_name', state.featureName);

  return next;
}

export function buildImageDetailQ3Href(imageId: string, state: ImageDetailQ3State): string {
  const params = createImageDetailQ3SearchParams(state);
  return `/image/${encodeURIComponent(imageId)}?${params.toString()}`;
}

export function parseAdvancedQ3WorkspaceState(
  searchParams: URLSearchParams,
  options?: {
    availableModels?: readonly string[];
    maxLayer?: number;
    numHeads?: number;
  },
): AdvancedQ3WorkspaceState {
  const availableModels = options?.availableModels ?? Q3_PRIMARY_MODELS;
  const primaryModel = parseWorkspaceModel(searchParams.get('primary_model'), availableModels);
  const secondaryModel = parseWorkspaceSecondaryModel(
    searchParams.get('secondary_model'),
    primaryModel,
    availableModels,
  );
  const maxLayer = options?.maxLayer ?? Q3_DEFAULTS.layer;
  const numHeads = options?.numHeads ?? 12;

  return {
    primaryModel,
    secondaryModel,
    variant: parseQ3Variant(searchParams.get('variant')),
    layer: clampNumber(parseOptionalNumber(searchParams.get('layer')) ?? Q3_DEFAULTS.layer, 0, maxLayer),
    metric: parseQ3Metric(searchParams.get('metric')),
    percentile: clampNumber(parseOptionalNumber(searchParams.get('percentile')) ?? Q3_DEFAULTS.percentile, 0, 100),
    head: parseHeadParam(searchParams.get('head'), numHeads),
    featureLabel: parseOptionalNumber(searchParams.get('feature_label')),
  };
}

export function parseQ3ReportState(
  searchParams: URLSearchParams,
  options?: {
    availableModels?: readonly string[];
    maxLayer?: number;
    numHeads?: number;
  },
): Q3ReportState {
  const availableModels = options?.availableModels ?? Q3_PRIMARY_MODELS;
  const maxLayer = options?.maxLayer ?? Q3_REPORT_DEFAULTS.layer;
  const numHeads = options?.numHeads ?? 12;

  return {
    view: parseQ3ReportView(searchParams.get('view')),
    model: parseWorkspaceModel(searchParams.get('model'), availableModels, Q3_REPORT_DEFAULTS.model),
    variant: parseQ3Variant(searchParams.get('variant')),
    layer: clampNumber(parseOptionalNumber(searchParams.get('layer')) ?? Q3_REPORT_DEFAULTS.layer, 0, maxLayer),
    metric: parseQ3Metric(searchParams.get('metric')),
    percentile: clampNumber(parseOptionalNumber(searchParams.get('percentile')) ?? Q3_REPORT_DEFAULTS.percentile, 0, 100),
    head: parseHeadParam(searchParams.get('head'), numHeads),
    featureLabel: parseOptionalNumber(searchParams.get('feature_label')),
  };
}

export function createQ3ReportSearchParams(
  state: Q3ReportState,
  existing?: URLSearchParams,
): URLSearchParams {
  const next = new URLSearchParams(existing);
  next.set('view', state.view);
  next.set('model', state.model);
  next.set('variant', state.variant);
  next.set('layer', String(state.layer));
  next.set('metric', state.metric);
  next.set('percentile', String(state.percentile));

  setOptionalNumberParam(next, 'head', state.head);
  setOptionalNumberParam(next, 'feature_label', state.featureLabel);

  return next;
}

export function buildQ3ReportHref(state?: Partial<Q3ReportState>): string {
  const params = createQ3ReportSearchParams({
    ...Q3_REPORT_DEFAULTS,
    ...state,
  });
  return `/q3-report?${params.toString()}`;
}

export function createAdvancedQ3WorkspaceSearchParams(
  state: AdvancedQ3WorkspaceState,
  existing?: URLSearchParams,
): URLSearchParams {
  const next = new URLSearchParams(existing);
  next.set('primary_model', state.primaryModel);
  next.set('secondary_model', state.secondaryModel);
  next.set('variant', state.variant);
  next.set('layer', String(state.layer));
  next.set('metric', state.metric);
  next.set('percentile', String(state.percentile));

  setOptionalNumberParam(next, 'head', state.head);
  setOptionalNumberParam(next, 'feature_label', state.featureLabel);

  return next;
}

export function buildAdvancedQ3WorkspaceHref(state: AdvancedQ3WorkspaceState): string {
  const params = createAdvancedQ3WorkspaceSearchParams(state);
  return `/q3?${params.toString()}`;
}

export function getDefaultSecondaryQ3Model(
  primaryModel: string,
  availableModels: readonly string[] = Q3_PRIMARY_MODELS,
): string {
  return availableModels.find((model) => model !== primaryModel) ?? primaryModel;
}

function parseQ3Model(value: string | null): string {
  return value && Q3_PRIMARY_MODELS.includes(value as (typeof Q3_PRIMARY_MODELS)[number])
    ? value
    : Q3_DEFAULTS.model;
}

function parseWorkspaceModel(value: string | null, availableModels: readonly string[]): string;
function parseWorkspaceModel(
  value: string | null,
  availableModels: readonly string[],
  fallbackModel: string,
): string;

function parseWorkspaceModel(value: string | null, availableModels: readonly string[], fallbackModel?: string): string {
  if (value && availableModels.includes(value)) {
    return value;
  }

  const preferredModel = fallbackModel ?? Q3_DEFAULTS.model;
  const defaultModel = availableModels.includes(preferredModel) ? preferredModel : availableModels[0];
  return defaultModel ?? preferredModel;
}

function parseWorkspaceSecondaryModel(
  value: string | null,
  primaryModel: string,
  availableModels: readonly string[],
): string {
  if (value && value !== primaryModel && availableModels.includes(value)) {
    return value;
  }

  return getDefaultSecondaryQ3Model(primaryModel, availableModels);
}

function parseQ3Variant(value: string | null): CompareVariantId {
  return isCompareVariantId(value) ? value : Q3_DEFAULTS.variant;
}

function parseQ3Metric(value: string | null): AnalysisMetric {
  return isAnalysisMetric(value) ? value : Q3_DEFAULTS.metric;
}

function parseQ3ReportView(value: string | null): Q3ReportView {
  if (value === 'head-feature-matrix' || value === 'frozen-delta' || value === 'head-ranking') {
    return value;
  }
  return Q3_REPORT_DEFAULTS.view;
}

function parseBooleanParam(value: string | null, fallback: boolean): boolean {
  if (value === 'true') return true;
  if (value === 'false') return false;
  return fallback;
}

function parseHeadParam(value: string | null, numHeads: number): number | null {
  if (value === null || value === 'all') {
    return Q3_DEFAULTS.head;
  }
  const parsed = parseOptionalNumber(value);
  if (parsed === null) {
    return Q3_DEFAULTS.head;
  }
  return clampNumber(parsed, 0, Math.max(0, numHeads - 1));
}

function parseOptionalNumber(value: string | null): number | null {
  if (value === null || value.trim() === '') {
    return null;
  }
  const parsed = Number(value);
  return Number.isInteger(parsed) && parsed >= 0 ? parsed : null;
}

function clampNumber(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

function setOptionalNumberParam(params: URLSearchParams, key: string, value: number | null): void {
  if (value === null) {
    params.delete(key);
    return;
  }
  params.set(key, String(value));
}

function setOptionalStringParam(params: URLSearchParams, key: string, value: string | null): void {
  if (!value) {
    params.delete(key);
    return;
  }
  params.set(key, value);
}
