import type { DashboardMetric } from '../types';

type AxisMode = 'unit' | 'auto';

interface DashboardMetricMetadata {
  optionLabel: string;
  shortLabel: string;
  chartLabel: string;
  hint: (percentile: number) => string;
  thresholdFree: boolean;
  infoBanner?: string;
  axisMode: AxisMode;
}

export const DASHBOARD_METRIC_METADATA: Record<DashboardMetric, DashboardMetricMetadata> = {
  iou: {
    optionLabel: 'IoU',
    shortLabel: 'IoU',
    chartLabel: 'IoU',
    hint: (percentile) => `Top ${100 - percentile}% threshold`,
    thresholdFree: false,
    axisMode: 'unit',
  },
  mse: {
    optionLabel: 'MSE',
    shortLabel: 'MSE',
    chartLabel: 'MSE (lower better)',
    hint: () => 'Lower is better',
    thresholdFree: true,
    infoBanner:
      'MSE compares each attention heatmap against the Gaussian soft-union ground truth and is threshold-free, so changing the percentile keeps the dashboard scores the same.',
    axisMode: 'unit',
  },
  kl: {
    optionLabel: 'KL',
    shortLabel: 'KL',
    chartLabel: 'KL divergence (lower better)',
    hint: () => 'Lower is better',
    thresholdFree: true,
    infoBanner:
      'KL divergence reports KL(GT || attention) after both heatmaps are converted into smoothed probability distributions, so changing the percentile keeps the dashboard scores the same.',
    axisMode: 'auto',
  },
};

export const DASHBOARD_METRIC_OPTIONS = (Object.entries(DASHBOARD_METRIC_METADATA) as Array<
  [DashboardMetric, DashboardMetricMetadata]
>).map(([value, metadata]) => ({
  value,
  label: metadata.optionLabel,
}));
