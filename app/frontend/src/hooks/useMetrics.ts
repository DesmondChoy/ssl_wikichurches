/**
 * React Query hooks for metrics data.
 */

import { useQuery } from '@tanstack/react-query';
import { metricsAPI, comparisonAPI } from '../api/client';
import type { AnalysisMetric, DashboardMetric, RankingMode } from '../types';

export function useStyleBreakdown(model: string, layer: number, percentile: number, method?: string) {
  return useQuery({
    queryKey: ['styleBreakdown', model, layer, percentile, method],
    queryFn: () => metricsAPI.getStyleBreakdown(model, layer, percentile, method),
  });
}

export function useFeatureBreakdown(
  model: string,
  layer: number,
  percentile: number,
  sortBy: 'mean_iou' | 'bbox_count' | 'feature_name' | 'feature_label' = 'mean_iou',
  minCount = 0,
  method?: string
) {
  return useQuery({
    queryKey: ['featureBreakdown', model, layer, percentile, sortBy, minCount, method],
    queryFn: () => metricsAPI.getFeatureBreakdown(model, layer, percentile, sortBy, minCount, method),
  });
}

export function useAllModelsSummary(
  percentile: number,
  metric: DashboardMetric,
  options?: { method?: string; rankingMode?: RankingMode }
) {
  return useQuery({
    queryKey: ['allModelsSummary', percentile, metric, options?.method, options?.rankingMode],
    queryFn: () => comparisonAPI.getAllModelsSummary(percentile, metric, options),
  });
}

export function useQ2Summary(
  metric: AnalysisMetric,
  percentile?: number,
  model?: string,
  strategy?: string
) {
  return useQuery({
    queryKey: ['q2Summary', metric, percentile, model, strategy],
    queryFn: () => metricsAPI.getQ2Summary({ metric, percentile, model, strategy }),
  });
}
