/**
 * Variant comparison slider for frozen-vs-fine-tuned and strategy-vs-strategy views.
 */

import { useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { ReactCompareSlider } from 'react-compare-slider';
import { attentionAPI, comparisonAPI, imagesAPI, metricsAPI } from '../../api/client';
import { Card, CardContent } from '../ui/Card';
import { InteractiveBboxOverlay } from '../attention/InteractiveBboxOverlay';
import { useModels } from '../../hooks/useAttention';
import { useHeatmapOpacity, useHeatmapStyle } from '../../store/viewStore';
import { computeSimilarityStats, renderHeatmap, renderHeatmapLegend } from '../../utils/renderHeatmap';
import type { BoundingBox, ComparisonVariant } from '../../types';

type ComparisonMode = 'frozen' | 'methods';
type ViewMode = 'side-by-side' | 'slider';

interface FrozenVsFinetunedProps {
  imageId: string;
  model: string;
  layer: number;
  strategy?: string;
  strategyA?: string;
  strategyB?: string;
  mode?: ComparisonMode;
  bboxes?: BoundingBox[];
  showBboxes?: boolean;
}

interface CompareCanvasProps {
  imageSrc: string;
  imageAlt: string;
  overlaySrc?: string | null;
  overlayAlt?: string;
}

interface NormalizedVariantComparison {
  left: ComparisonVariant;
  right: ComparisonVariant;
  note: string;
  linearProbeInvolved: boolean;
}

function CompareCanvas({ imageSrc, imageAlt, overlaySrc, overlayAlt }: CompareCanvasProps) {
  return (
    <div className="relative h-full w-full overflow-hidden rounded-lg bg-gray-950">
      <img
        src={imageSrc}
        alt={imageAlt}
        className="absolute inset-0 h-full w-full object-cover"
      />
      {overlaySrc && (
        <img
          src={overlaySrc}
          alt={overlayAlt || imageAlt}
          className="pointer-events-none absolute inset-0 h-full w-full object-cover"
        />
      )}
    </div>
  );
}

function strategyLabel(strategy?: string | null) {
  if (!strategy) return 'Fine-tuned';
  if (strategy === 'linear_probe') return 'Linear Probe';
  if (strategy === 'lora') return 'LoRA';
  if (strategy === 'full') return 'Full Fine-tune';
  return strategy;
}

export function FrozenVsFinetuned({
  imageId,
  model,
  layer,
  strategy,
  strategyA = 'linear_probe',
  strategyB = 'full',
  mode = 'frozen',
  bboxes = [],
  showBboxes = true,
}: FrozenVsFinetunedProps) {
  const [selectedBboxIndex, setSelectedBboxIndex] = useState<number | null>(null);
  const [viewMode, setViewMode] = useState<ViewMode>('side-by-side');
  const { data: modelsData, isLoading: modelsLoading } = useModels();
  const heatmapOpacity = useHeatmapOpacity();
  const heatmapStyle = useHeatmapStyle();
  const maxLayer = modelsData?.num_layers_per_model?.[model];
  const effectiveLayer = typeof maxLayer === 'number' ? Math.min(layer, maxLayer - 1) : layer;

  const {
    data,
    isLoading,
    error,
  } = useQuery({
    queryKey: [
      'variant-compare',
      mode,
      imageId,
      model,
      effectiveLayer,
      strategy,
      strategyA,
      strategyB,
      showBboxes,
    ],
    queryFn: async (): Promise<NormalizedVariantComparison> => {
      if (mode === 'methods') {
        const response = await comparisonAPI.compareFinetunedVariants(
          imageId,
          model,
          effectiveLayer,
          strategyA,
          strategyB,
          showBboxes
        );
        return {
          left: response.left,
          right: response.right,
          note: response.note,
          linearProbeInvolved:
            response.left.strategy === 'linear_probe' || response.right.strategy === 'linear_probe',
        };
      }

      const response = await comparisonAPI.compareFrozenVsFinetuned(
        imageId,
        model,
        effectiveLayer,
        strategy,
        showBboxes
      );
      return {
        left: {
          model_key: model,
          strategy: null,
          label: 'Frozen (Pretrained)',
          available: response.frozen.available,
          url: response.frozen.url,
        },
        right: {
          model_key: response.strategy ? `${model}_finetuned_${response.strategy}` : `${model}_finetuned`,
          strategy: response.strategy,
          label: response.strategy ? `Fine-tuned (${strategyLabel(response.strategy)})` : 'Fine-tuned',
          available: response.finetuned.available,
          url: response.finetuned.url,
        },
        note: response.finetuned.note,
        linearProbeInvolved: response.strategy === 'linear_probe',
      };
    },
    enabled: Boolean(imageId && model && !modelsLoading),
  });

  const leftUrl = data?.left.url ?? null;
  const rightUrl = data?.right.url ?? null;
  const labels = Array.from(new Set(bboxes.map((bbox) => bbox.label_name || `Feature ${bbox.label}`)));
  const sliderAvailable =
    Boolean(data?.left.available) &&
    Boolean(data?.right.available) &&
    typeof leftUrl === 'string' &&
    typeof rightUrl === 'string';
  const selectedBbox = selectedBboxIndex !== null ? bboxes[selectedBboxIndex] : null;
  const originalUrl = imagesAPI.getImageUrl(imageId, 224);
  const legendUrl = useMemo(() => renderHeatmapLegend(200, 16), []);

  const featureOptions = useMemo(
    () =>
      bboxes.map((bbox, index) => ({
        index,
        label: bbox.label_name || `Feature ${bbox.label}`,
      })),
    [bboxes]
  );

  const {
    data: bboxMetrics,
    isLoading: bboxMetricsLoading,
    error: bboxMetricsError,
  } = useQuery({
    queryKey: [
      'variant-compare-bbox-metrics',
      imageId,
      effectiveLayer,
      selectedBboxIndex,
      data?.left.model_key,
      data?.right.model_key,
    ],
    queryFn: async () => {
      if (selectedBboxIndex === null || !data) {
        return null;
      }

      const [left, right] = await Promise.all([
        metricsAPI.getBboxMetrics(imageId, data.left.model_key, effectiveLayer, selectedBboxIndex),
        metricsAPI.getBboxMetrics(imageId, data.right.model_key, effectiveLayer, selectedBboxIndex),
      ]);

      return { left, right };
    },
    enabled: sliderAvailable && selectedBboxIndex !== null && Boolean(data),
  });

  const {
    data: similarityData,
    isLoading: similarityLoading,
    error: similarityError,
  } = useQuery({
    queryKey: [
      'variant-compare-similarity',
      imageId,
      data?.left.model_key,
      data?.right.model_key,
      effectiveLayer,
      selectedBbox,
    ],
    queryFn: async () => {
      if (!selectedBbox || !data) {
        return null;
      }

      const bboxPayload = {
        left: selectedBbox.left,
        top: selectedBbox.top,
        width: selectedBbox.width,
        height: selectedBbox.height,
        label: selectedBbox.label_name || undefined,
      };

      const [left, right] = await Promise.all([
        attentionAPI.getSimilarity(imageId, bboxPayload, data.left.model_key, effectiveLayer),
        attentionAPI.getSimilarity(imageId, bboxPayload, data.right.model_key, effectiveLayer),
      ]);

      return { left, right };
    },
    enabled: sliderAvailable && Boolean(selectedBbox) && Boolean(data),
    retry: false,
  });

  const leftSimilarity = similarityData?.left?.similarity;
  const leftPatchGrid = similarityData?.left?.patch_grid as [number, number] | undefined;
  const rightSimilarity = similarityData?.right?.similarity;
  const rightPatchGrid = similarityData?.right?.patch_grid as [number, number] | undefined;

  const leftSimilarityHeatmapUrl = useMemo(() => {
    if (!leftSimilarity || !leftPatchGrid) {
      return null;
    }
    try {
      return renderHeatmap({
        similarity: leftSimilarity,
        patchGrid: leftPatchGrid,
        opacity: heatmapOpacity,
        style: heatmapStyle,
      });
    } catch {
      return null;
    }
  }, [heatmapOpacity, heatmapStyle, leftPatchGrid, leftSimilarity]);

  const rightSimilarityHeatmapUrl = useMemo(() => {
    if (!rightSimilarity || !rightPatchGrid) {
      return null;
    }
    try {
      return renderHeatmap({
        similarity: rightSimilarity,
        patchGrid: rightPatchGrid,
        opacity: heatmapOpacity,
        style: heatmapStyle,
      });
    } catch {
      return null;
    }
  }, [heatmapOpacity, heatmapStyle, rightPatchGrid, rightSimilarity]);

  const leftSimilarityStats = useMemo(() => {
    if (!leftSimilarity) {
      return null;
    }
    return computeSimilarityStats(leftSimilarity);
  }, [leftSimilarity]);

  const rightSimilarityStats = useMemo(() => {
    if (!rightSimilarity) {
      return null;
    }
    return computeSimilarityStats(rightSimilarity);
  }, [rightSimilarity]);

  const showSimilarityHeatmaps =
    Boolean(selectedBbox) &&
    Boolean(leftSimilarityHeatmapUrl) &&
    Boolean(rightSimilarityHeatmapUrl) &&
    !similarityLoading &&
    !similarityError;

  if (modelsLoading || isLoading) {
    return (
      <div className="rounded-lg border border-gray-200 bg-gray-50 p-4 text-sm text-gray-600">
        Loading comparison availability...
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-700">
        Failed to load comparison data.
      </div>
    );
  }

  if (!data) {
    return null;
  }

  if (!sliderAvailable) {
    return (
      <div className="space-y-4">
        {leftUrl && (
          <div className="relative">
            <img
              src={leftUrl}
              alt={`${data.left.label} attention`}
              className="h-auto w-full rounded-lg"
            />
            <div className="absolute bottom-2 left-2 rounded bg-black/50 px-2 py-1 text-xs text-white">
              {data.left.label}
            </div>
          </div>
        )}

        <div className="rounded-lg border border-yellow-200 bg-yellow-50 p-4">
          <h4 className="font-medium text-yellow-800">Comparison Overlay Unavailable</h4>
          <p className="mt-1 text-sm text-yellow-700">{data.note}</p>
          <p className="mt-1 text-sm text-yellow-700">
            The slider comparison appears automatically when both variant overlays are cached.
          </p>
        </div>
      </div>
    );
  }

  const deltaIoU =
    bboxMetrics && typeof bboxMetrics.right.iou === 'number' && typeof bboxMetrics.left.iou === 'number'
      ? bboxMetrics.right.iou - bboxMetrics.left.iou
      : null;
  const deltaCoverage =
    bboxMetrics &&
    typeof bboxMetrics.right.coverage === 'number' &&
    typeof bboxMetrics.left.coverage === 'number'
      ? bboxMetrics.right.coverage - bboxMetrics.left.coverage
      : null;

  const deltaTone =
    deltaIoU === null ? 'border-gray-200 bg-gray-100 text-gray-700'
    : deltaIoU > 0 ? 'border-green-200 bg-green-50 text-green-700'
    : deltaIoU < 0 ? 'border-red-200 bg-red-50 text-red-700'
    : 'border-gray-200 bg-gray-100 text-gray-700';

  const setBboxIndex = (index: number | null) => {
    setSelectedBboxIndex(index);
  };

  return (
    <div className="space-y-4">
      {/* View toggle: side-by-side (default) vs slider */}
      <div className="flex items-center gap-2">
        <span className="text-sm font-medium text-gray-700">View:</span>
        <div className="inline-flex rounded-lg border border-gray-200 bg-gray-50 p-0.5">
          <button
            type="button"
            onClick={() => setViewMode('side-by-side')}
            className={`rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
              viewMode === 'side-by-side'
                ? 'bg-white text-gray-900 shadow-sm'
                : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            Side by side
          </button>
          <button
            type="button"
            onClick={() => setViewMode('slider')}
            className={`rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
              viewMode === 'slider'
                ? 'bg-white text-gray-900 shadow-sm'
                : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            Slider
          </button>
        </div>
      </div>

      {viewMode === 'side-by-side' && (
        <>
          <div className="grid grid-cols-2 gap-4">
            {/* Left variant */}
            <Card>
              <CardContent className="p-0">
                <div className="px-3 py-2 text-sm font-medium text-gray-900 border-b border-gray-200">
                  {data.left.label}
                </div>
                <div className="relative aspect-square w-full overflow-hidden bg-gray-950">
                  <CompareCanvas
                    imageSrc={showSimilarityHeatmaps ? originalUrl : leftUrl!}
                    imageAlt={showSimilarityHeatmaps ? `${data.left.label} similarity heatmap` : `${data.left.label} attention`}
                    overlaySrc={showSimilarityHeatmaps ? leftSimilarityHeatmapUrl : null}
                    overlayAlt={`${data.left.label} similarity overlay`}
                  />
                  {similarityLoading && selectedBbox && (
                    <div className="pointer-events-none absolute inset-0 flex items-center justify-center bg-black/20">
                      <div className="h-8 w-8 animate-spin rounded-full border-2 border-white border-t-transparent" />
                    </div>
                  )}
                  {showBboxes && bboxes.length > 0 && (
                    <InteractiveBboxOverlay
                      bboxes={bboxes}
                      selectedIndex={selectedBboxIndex}
                      onBboxClick={(_bbox, index) => setBboxIndex(selectedBboxIndex === index ? null : index)}
                    />
                  )}
                </div>
                <div className="px-3 py-2 text-sm text-gray-600 border-t border-gray-200">
                  {selectedBbox && bboxMetrics && !bboxMetricsLoading ? (
                    <span>IoU: {bboxMetrics.left.iou.toFixed(3)} · Coverage: {bboxMetrics.left.coverage.toFixed(3)}</span>
                  ) : selectedBbox && bboxMetricsLoading ? (
                    <span>Loading metrics…</span>
                  ) : (
                    <span>Click a feature to see metrics</span>
                  )}
                </div>
              </CardContent>
            </Card>

            {/* Right variant */}
            <Card>
              <CardContent className="p-0">
                <div className="px-3 py-2 text-sm font-medium text-gray-900 border-b border-gray-200">
                  {data.right.label}
                </div>
                <div className="relative aspect-square w-full overflow-hidden bg-gray-950">
                  <CompareCanvas
                    imageSrc={showSimilarityHeatmaps ? originalUrl : rightUrl!}
                    imageAlt={showSimilarityHeatmaps ? `${data.right.label} similarity heatmap` : `${data.right.label} attention`}
                    overlaySrc={showSimilarityHeatmaps ? rightSimilarityHeatmapUrl : null}
                    overlayAlt={`${data.right.label} similarity overlay`}
                  />
                  {similarityLoading && selectedBbox && (
                    <div className="pointer-events-none absolute inset-0 flex items-center justify-center bg-black/20">
                      <div className="h-8 w-8 animate-spin rounded-full border-2 border-white border-t-transparent" />
                    </div>
                  )}
                  {showBboxes && bboxes.length > 0 && (
                    <InteractiveBboxOverlay
                      bboxes={bboxes}
                      selectedIndex={selectedBboxIndex}
                      onBboxClick={(_bbox, index) => setBboxIndex(selectedBboxIndex === index ? null : index)}
                    />
                  )}
                </div>
                <div className="px-3 py-2 text-sm text-gray-600 border-t border-gray-200">
                  {selectedBbox && bboxMetrics && !bboxMetricsLoading ? (
                    <span>IoU: {bboxMetrics.right.iou.toFixed(3)} · Coverage: {bboxMetrics.right.coverage.toFixed(3)}</span>
                  ) : selectedBbox && bboxMetricsLoading ? (
                    <span>Loading metrics…</span>
                  ) : (
                    <span>Click a feature to see metrics</span>
                  )}
                </div>
              </CardContent>
            </Card>
          </div>
          {data.linearProbeInvolved && (
            <p className="text-center text-xs text-amber-700">
              Linear probe trains only the classifier head, so attention maps can look identical.
            </p>
          )}
          {similarityError && selectedBbox && (
            <p className="text-center text-xs text-amber-700">
              Similarity heatmaps are unavailable for this selection. Run feature-cache generation for both compared variants to enable bbox-local inspection.
            </p>
          )}
        </>
      )}

      {viewMode === 'slider' && (
        <>
          <div className="flex justify-between text-sm text-gray-600">
            <span>{data.left.label}</span>
            <span>{data.right.label}</span>
          </div>
          <div className="rounded-lg border border-sky-100 bg-sky-50/70 px-4 py-3 text-sm text-sky-900">
            Clicking a bounding box switches the slider from cached global overlays to bbox-conditioned
            similarity heatmaps, using the same architectural feature as the query for both compared variants.
          </div>
          <div className="relative mx-auto w-full max-w-3xl">
            <ReactCompareSlider
              itemOne={
                <CompareCanvas
                  imageSrc={showSimilarityHeatmaps ? originalUrl : leftUrl!}
                  imageAlt={showSimilarityHeatmaps ? `${data.left.label} similarity heatmap` : `${data.left.label} attention`}
                  overlaySrc={showSimilarityHeatmaps ? leftSimilarityHeatmapUrl : null}
                  overlayAlt={`${data.left.label} similarity overlay`}
                />
              }
              itemTwo={
                <CompareCanvas
                  imageSrc={showSimilarityHeatmaps ? originalUrl : rightUrl!}
                  imageAlt={showSimilarityHeatmaps ? `${data.right.label} similarity heatmap` : `${data.right.label} attention`}
                  overlaySrc={showSimilarityHeatmaps ? rightSimilarityHeatmapUrl : null}
                  overlayAlt={`${data.right.label} similarity overlay`}
                />
              }
              className="aspect-square overflow-hidden rounded-lg"
              position={50}
            />
            {similarityLoading && selectedBbox && (
              <div className="pointer-events-none absolute inset-0 flex items-center justify-center bg-black/20">
                <div className="h-8 w-8 animate-spin rounded-full border-2 border-white border-t-transparent" />
              </div>
            )}
            {showBboxes && bboxes.length > 0 && (
              <InteractiveBboxOverlay
                bboxes={bboxes}
                selectedIndex={selectedBboxIndex}
                onBboxClick={(_bbox, index) => {
                  setSelectedBboxIndex((current) => (current === index ? null : index));
                }}
              />
            )}
          </div>
          <p className="text-center text-xs text-gray-500">
            {showSimilarityHeatmaps
              ? 'Drag slider to compare bbox-conditioned similarity heatmaps'
              : `Drag slider to compare ${data.left.label.toLowerCase()} vs ${data.right.label.toLowerCase()} attention${showBboxes ? ' with annotated boxes' : ''}`}
          </p>
          {data.linearProbeInvolved && (
            <p className="text-center text-xs text-amber-700">
              Linear probe trains only the classifier head, so attention maps can look identical.
            </p>
          )}
          {similarityError && selectedBbox && (
            <p className="text-center text-xs text-amber-700">
              Similarity heatmaps are unavailable for this selection, so the view stays on the global overlays. Run
              feature-cache generation for both compared variants to enable bbox-local inspection.
            </p>
          )}
        </>
      )}

      {labels.length > 0 && (
        <div className="space-y-3 rounded-lg border border-gray-200 bg-white p-4">
          <div className="flex items-center justify-between gap-3">
            <div>
              <p className="text-sm font-medium text-gray-900">Feature-focused inspection</p>
              <p className="text-xs text-gray-600">
                Click a box on the image or choose a feature below to compare local alignment.
              </p>
            </div>
            {selectedBboxIndex !== null && (
              <button
                type="button"
                onClick={() => setSelectedBboxIndex(null)}
                className="text-xs font-medium text-primary-600 hover:underline"
              >
                Clear selection
              </button>
            )}
          </div>

          <div className="flex flex-wrap gap-2">
            {featureOptions.map((feature) => {
              const selected = selectedBboxIndex === feature.index;
              return (
                <button
                  key={`${feature.label}-${feature.index}`}
                  type="button"
                  onClick={() => {
                    setSelectedBboxIndex((current) => (current === feature.index ? null : feature.index));
                  }}
                  className={`rounded-full border px-3 py-1 text-xs transition-colors ${
                    selected
                      ? 'border-primary-600 bg-primary-50 text-primary-700'
                      : 'border-gray-200 bg-gray-50 text-gray-700 hover:border-gray-300'
                  }`}
                >
                  {feature.label}
                </button>
              );
            })}
          </div>

          {selectedBbox && (
            <div className="grid gap-3 md:grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)]">
              <div className="rounded-lg border border-gray-200 bg-gray-50 p-3">
                <p className="text-sm font-medium text-gray-900">
                  Selected feature: {selectedBbox.label_name || `Feature ${selectedBbox.label}`}
                </p>
                <p className="mt-1 text-xs text-gray-600">
                  Bounding box #{selectedBboxIndex! + 1} at layer {effectiveLayer}. The slider now uses this expert
                  region as the similarity query, and the metrics below are computed inside the same bbox.
                </p>
              </div>

              <div className={`rounded-lg border p-3 ${deltaTone}`}>
                <p className="text-xs font-semibold uppercase tracking-wide">Feature-local delta</p>
                {bboxMetricsLoading ? (
                  <p className="mt-2 text-sm">Loading feature metrics...</p>
                ) : bboxMetricsError ? (
                  <p className="mt-2 text-sm">
                    Failed to load bbox metrics for this comparison. Check that both compared metrics are cached.
                  </p>
                ) : bboxMetrics ? (
                  <>
                    <p className="mt-2 text-2xl font-semibold">
                      {deltaIoU === null ? 'n/a' : `${deltaIoU >= 0 ? '+' : ''}${deltaIoU.toFixed(3)} IoU`}
                    </p>
                    <p className="mt-1 text-xs">
                      {deltaCoverage === null
                        ? 'Coverage delta unavailable'
                        : `${deltaCoverage >= 0 ? '+' : ''}${deltaCoverage.toFixed(3)} coverage`}
                    </p>
                  </>
                ) : (
                  <p className="mt-2 text-sm">Select a feature to see local change.</p>
                )}
              </div>
            </div>
          )}

          {selectedBbox && showSimilarityHeatmaps && (
            <div className="flex items-center justify-between gap-4 rounded-lg border border-gray-200 bg-gray-50 p-3 text-xs text-gray-600">
              <div className="space-y-1">
                <p className="font-medium text-gray-900">{data.left.label} similarity</p>
                <p>
                  {leftSimilarityStats
                    ? `min ${leftSimilarityStats.min.toFixed(2)} | max ${leftSimilarityStats.max.toFixed(2)}`
                    : 'No stats available'}
                </p>
              </div>
              <div className="flex items-center gap-2">
                <span>Low similarity</span>
                <img src={legendUrl} alt="Similarity scale" className="h-4 rounded" />
                <span>High similarity</span>
              </div>
              <div className="space-y-1 text-right">
                <p className="font-medium text-gray-900">{data.right.label} similarity</p>
                <p>
                  {rightSimilarityStats
                    ? `min ${rightSimilarityStats.min.toFixed(2)} | max ${rightSimilarityStats.max.toFixed(2)}`
                    : 'No stats available'}
                </p>
              </div>
            </div>
          )}

          {selectedBbox && bboxMetrics && !bboxMetricsLoading && !bboxMetricsError && (
            <div className="grid gap-3 md:grid-cols-2">
              <div className="rounded-lg border border-gray-200 p-3">
                <p className="text-xs font-semibold uppercase tracking-wide text-gray-500">{data.left.label}</p>
                <p className="mt-2 text-sm text-gray-900">IoU: {bboxMetrics.left.iou.toFixed(3)}</p>
                <p className="text-sm text-gray-700">Coverage: {bboxMetrics.left.coverage.toFixed(3)}</p>
              </div>
              <div className="rounded-lg border border-gray-200 p-3">
                <p className="text-xs font-semibold uppercase tracking-wide text-gray-500">{data.right.label}</p>
                <p className="mt-2 text-sm text-gray-900">IoU: {bboxMetrics.right.iou.toFixed(3)}</p>
                <p className="text-sm text-gray-700">Coverage: {bboxMetrics.right.coverage.toFixed(3)}</p>
              </div>
            </div>
          )}

          {!selectedBbox && (
            <div className="rounded-lg border border-dashed border-gray-300 bg-gray-50 p-3 text-sm text-gray-600">
              Global overlays often look similar. Selecting a feature switches the view to bbox-conditioned
              similarity heatmaps so you can compare the same architectural cue across the chosen variants.
            </div>
          )}
        </div>
      )}
    </div>
  );
}
