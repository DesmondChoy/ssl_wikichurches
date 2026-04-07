import { isCompareVariantId } from './metricMetadata';
import { parseImageDetailMode } from './imageDetailModes';
import { Q3_DEFAULTS, Q3_PRIMARY_MODELS } from './q3Scope';
import type { CompareVariantId, ImageDetailMode } from '../types';

export interface ImageDetailQ3State {
  model: string;
  variant: CompareVariantId;
  layer: number;
  head: number | null;
  mode: ImageDetailMode;
  showBboxes: boolean;
  bboxIndex: number | null;
  featureLabel: number | null;
  featureName: string | null;
}

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

function parseQ3Model(value: string | null): string {
  return value && Q3_PRIMARY_MODELS.includes(value as (typeof Q3_PRIMARY_MODELS)[number])
    ? value
    : Q3_DEFAULTS.model;
}

function parseQ3Variant(value: string | null): CompareVariantId {
  return isCompareVariantId(value) ? value : Q3_DEFAULTS.variant;
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
