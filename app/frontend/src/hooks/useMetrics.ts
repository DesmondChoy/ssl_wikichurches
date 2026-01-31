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

export function useStyleBreakdown(model: string, layer: number, percentile: number) {
  return useQuery({
    queryKey: ['styleBreakdown', model, layer, percentile],
    queryFn: () => metricsAPI.getStyleBreakdown(model, layer, percentile),
  });
}

export function useAggregateMetrics(model: string, layer: number, percentile: number) {
  return useQuery({
    queryKey: ['aggregate', model, layer, percentile],
    queryFn: () => metricsAPI.getAggregate(model, layer, percentile),
  });
}

export function useAllModelsSummary(percentile: number) {
  return useQuery({
    queryKey: ['allModelsSummary', percentile],
    queryFn: () => comparisonAPI.getAllModelsSummary(percentile),
  });
}
