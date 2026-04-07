import type { AnalysisMetric, CompareVariantId } from '../types';

export type Q3ScopeStatus = 'primary' | 'control' | 'outside';

export const Q3_PRIMARY_MODELS = ['dinov2', 'dinov3', 'mae', 'clip'] as const;
export const Q3_OUTSIDE_MODELS = ['siglip', 'siglip2', 'resnet50'] as const;
export const Q3_PRIMARY_VARIANTS: CompareVariantId[] = ['frozen', 'lora', 'full'];
export const Q3_CONTROL_VARIANT: CompareVariantId = 'linear_probe';

export const Q3_SCOPE_COPY = {
  title: 'Primary Q3 study scope',
  summary:
    'Primary claim centers dinov2, dinov3, mae, and clip, with frozen, LoRA, and full as the headline comparison set.',
  detail:
    'Linear Probe stays available as a control. SigLIP, SigLIP2, and ResNet-50 remain explorable outside the primary claim.',
  imageDetailNote:
    'Use Dashboard Q3 to compare frozen, LoRA, and full. Linear Probe stays available there as a control condition.',
} as const;

export const Q3_DEFAULTS: {
  model: string;
  method: string;
  variant: CompareVariantId;
  layer: number;
  metric: AnalysisMetric;
  percentile: number;
} = {
  model: 'dinov2',
  method: 'cls',
  variant: 'frozen',
  layer: 11,
  metric: 'iou',
  percentile: 90,
};

const Q3_SCOPE_LABELS: Record<Q3ScopeStatus, string> = {
  primary: 'Primary study',
  control: 'Control',
  outside: 'Outside primary scope',
};

export function getQ3ModelScopeStatus(model: string): Q3ScopeStatus {
  return Q3_PRIMARY_MODELS.includes(model as (typeof Q3_PRIMARY_MODELS)[number])
    ? 'primary'
    : 'outside';
}

export function getQ3VariantScopeStatus(variant: CompareVariantId): Q3ScopeStatus {
  return variant === Q3_CONTROL_VARIANT ? 'control' : 'primary';
}

export function getQ3ScopeLabel(status: Q3ScopeStatus): string {
  return Q3_SCOPE_LABELS[status];
}

export function getQ3ScopeChipClassName(status: Q3ScopeStatus): string {
  switch (status) {
    case 'primary':
      return 'border-emerald-200 bg-emerald-50 text-emerald-800';
    case 'control':
      return 'border-amber-200 bg-amber-50 text-amber-800';
    case 'outside':
      return 'border-slate-200 bg-slate-100 text-slate-700';
    default:
      return 'border-slate-200 bg-slate-100 text-slate-700';
  }
}

export function formatQ3ScopeOptionLabel(label: string, status: Q3ScopeStatus): string {
  return `${label} (${getQ3ScopeLabel(status)})`;
}

export function getQ3SelectionHelperText(
  modelStatus: Q3ScopeStatus,
  variantStatus?: Q3ScopeStatus,
): string {
  if (modelStatus === 'outside') {
    return 'This selection remains available for exploratory inspection, but it sits outside the headline Q3 claim.';
  }

  if (variantStatus === 'control') {
    return 'Linear Probe remains visible as a control rather than a peer headline comparison condition.';
  }

  return 'This selection is inside the headline Q3 study scope.';
}
