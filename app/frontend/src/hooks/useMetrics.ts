/**
 * React Query hooks for metrics data.
 */

import { useQuery } from '@tanstack/react-query';
import { metricsAPI, comparisonAPI } from '../api/client';

export function useLeaderboard(percentile: number) {
  return useQuery({
    queryKey: ['leaderboard', percentile],
    queryFn: () => metricsAPI.getLeaderboard(percentile),
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

export function useAllModelsSummary(percentile: number) {
  return useQuery({
    queryKey: ['allModelsSummary', percentile],
    queryFn: () => comparisonAPI.getAllModelsSummary(percentile),
  });
}
