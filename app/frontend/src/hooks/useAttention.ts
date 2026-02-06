/**
 * React Query hooks for attention data.
 */

import { useQuery, keepPreviousData } from '@tanstack/react-query';
import { attentionAPI, metricsAPI, comparisonAPI } from '../api/client';

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

export function useImageMetrics(
  imageId: string | undefined,
  model: string,
  layer: number,
  percentile: number,
  method?: string
) {
  return useQuery({
    queryKey: ['imageMetrics', imageId, model, layer, percentile, method],
    queryFn: () => metricsAPI.getImageMetrics(imageId!, model, layer, percentile, method),
    enabled: !!imageId,
    placeholderData: keepPreviousData,
  });
}

export function useBboxMetrics(
  imageId: string | undefined,
  model: string,
  layer: number,
  bboxIndex: number | null,
  percentile: number,
  method?: string
) {
  return useQuery({
    queryKey: ['bboxMetrics', imageId, model, layer, bboxIndex, percentile, method],
    queryFn: () => metricsAPI.getBboxMetrics(imageId!, model, layer, bboxIndex!, percentile, method),
    enabled: !!imageId && bboxIndex !== null,
    placeholderData: keepPreviousData,
  });
}

export function useLayerProgression(model: string, percentile: number, method?: string) {
  return useQuery({
    queryKey: ['layerProgression', model, percentile, method],
    queryFn: () => metricsAPI.getLayerProgression(model, percentile, method),
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
