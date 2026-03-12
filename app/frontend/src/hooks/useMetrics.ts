/**
 * React Query hooks for metrics data.
 */

import { useQuery } from '@tanstack/react-query';
import { metricsAPI, comparisonAPI } from '../api/client';
import type { DashboardMetric } from '../types';

export function useLeaderboard(percentile: number, metric: DashboardMetric) {
  return useQuery({
    queryKey: ['leaderboard', percentile, metric],
    queryFn: () => metricsAPI.getLeaderboard(percentile, metric),
  });
}

export function useMetricsSummary() {
  return useQuery({
    queryKey: ['metricsSummary'],
    queryFn: () => metricsAPI.getSummary(),
    staleTime: 60000, // 1 minute
  });
}

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

export function useAggregateMetrics(model: string, layer: number, percentile: number, method?: string) {
  return useQuery({
    queryKey: ['aggregate', model, layer, percentile, method],
    queryFn: () => metricsAPI.getAggregate(model, layer, percentile, method),
  });
}

export function useAllModelsSummary(percentile: number, metric: DashboardMetric) {
  return useQuery({
    queryKey: ['allModelsSummary', percentile, metric],
    queryFn: () => comparisonAPI.getAllModelsSummary(percentile, metric),
  });
}

export function useQ2Summary(
  percentile?: number,
  model?: string,
  strategy?: string
) {
  return useQuery({
    queryKey: ['q2Summary', percentile, model, strategy],
    queryFn: () => metricsAPI.getQ2Summary({ percentile, model, strategy }),
  });
}
