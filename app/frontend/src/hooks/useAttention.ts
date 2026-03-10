/**
 * React Query hooks for attention data.
 */

import { useQuery, keepPreviousData } from '@tanstack/react-query';
import { attentionAPI, metricsAPI, comparisonAPI } from '../api/client';
import type { DashboardMetric } from '../types';

export function useModels() {
  return useQuery({
    queryKey: ['models'],
    queryFn: () => attentionAPI.getModels(),
    staleTime: Infinity, // Models don't change
  });
}

export function useLayerUrls(imageId: string | undefined, model: string, showBboxes: boolean) {
  return useQuery({
    queryKey: ['layerUrls', imageId, model, showBboxes],
    queryFn: () => attentionAPI.getLayerUrls(imageId!, model, showBboxes),
    enabled: !!imageId,
  });
}

export function useImageLayerProgression(
  imageId: string | undefined,
  model: string,
  percentile: number,
  method?: string,
  bboxIndex?: number | null,
  enabled = true
) {
  return useQuery({
    queryKey: ['imageLayerProgression', imageId, model, percentile, method, bboxIndex],
    queryFn: () => metricsAPI.getImageLayerProgression(imageId!, model, percentile, method, bboxIndex),
    enabled: !!imageId && enabled,
    placeholderData: keepPreviousData,
  });
}

export function useLayerProgression(
  model: string,
  percentile: number,
  metric: DashboardMetric,
  method?: string
) {
  return useQuery({
    queryKey: ['layerProgression', model, percentile, metric, method],
    queryFn: () => metricsAPI.getLayerProgression(model, percentile, metric, method),
  });
}

export function useLayerComparison(
  imageId: string | undefined,
  model: string,
  percentile: number
) {
  return useQuery({
    queryKey: ['layerComparison', imageId, model, percentile],
    queryFn: () => comparisonAPI.compareLayers(imageId!, model, percentile),
    enabled: !!imageId,
  });
}

export function useModelComparison(
  imageId: string | undefined,
  models: string[],
  layer: number,
  percentile: number
) {
  return useQuery({
    queryKey: ['modelComparison', imageId, models, layer, percentile],
    queryFn: () => comparisonAPI.compareModels(imageId!, models, layer, percentile),
    enabled: !!imageId && models.length > 0,
  });
}
