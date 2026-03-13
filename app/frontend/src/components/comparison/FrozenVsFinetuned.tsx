/**
 * Frozen vs Fine-tuned comparison using react-compare-slider.
 */

import { useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { ReactCompareSlider } from 'react-compare-slider';
import { attentionAPI, comparisonAPI, imagesAPI, metricsAPI } from '../../api/client';
import { InteractiveBboxOverlay } from '../attention/InteractiveBboxOverlay';
import { useModels } from '../../hooks/useAttention';
import { useHeatmapOpacity, useHeatmapStyle } from '../../store/viewStore';
import { computeSimilarityStats, renderHeatmap, renderHeatmapLegend } from '../../utils/renderHeatmap';
import type { BoundingBox } from '../../types';

interface FrozenVsFinetunedProps {
  imageId: string;
  model: string;
  layer: number;
  strategy?: string;
  bboxes?: BoundingBox[];
  showBboxes?: boolean;
}

interface CompareCanvasProps {
  imageSrc: string;
  imageAlt: string;
  overlaySrc?: string | null;
  overlayAlt?: string;
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
          className="absolute inset-0 h-full w-full object-cover pointer-events-none"
        />
      )}
    </div>
  );
}

export function FrozenVsFinetuned({
  imageId,
  model,
  layer,
  strategy,
  bboxes = [],
  showBboxes = true,
}: FrozenVsFinetunedProps) {
  const [selectedBboxIndex, setSelectedBboxIndex] = useState<number | null>(null);
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
    queryKey: ['frozen-vs-finetuned', imageId, model, effectiveLayer, strategy, showBboxes],
    queryFn: () =>
      comparisonAPI.compareFrozenVsFinetuned(imageId, model, effectiveLayer, strategy, showBboxes),
    enabled: Boolean(imageId && model && !modelsLoading),
  });

  const frozenUrl = data?.frozen.url ?? null;
  const finetunedUrl = data?.finetuned.url ?? null;
  const labels = Array.from(new Set(bboxes.map((bbox) => bbox.label_name || `Feature ${bbox.label}`)));
  const sliderAvailable =
    Boolean(data?.frozen.available) &&
    Boolean(data?.finetuned.available) &&
    typeof frozenUrl === 'string' &&
    typeof finetunedUrl === 'string';
  const resolvedStrategy = data?.strategy || strategy || null;
  const finetunedModelKey = resolvedStrategy
    ? `${model}_finetuned_${resolvedStrategy}`
    : `${model}_finetuned`;
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
      'frozen-vs-finetuned-bbox-metrics',
      imageId,
      model,
      effectiveLayer,
      selectedBboxIndex,
      finetunedModelKey,
    ],
    queryFn: async () => {
      if (selectedBboxIndex === null) {
        return null;
      }

      const [frozen, finetuned] = await Promise.all([
        metricsAPI.getBboxMetrics(imageId, model, effectiveLayer, selectedBboxIndex),
        metricsAPI.getBboxMetrics(imageId, finetunedModelKey, effectiveLayer, selectedBboxIndex),
      ]);

      return { frozen, finetuned };
    },
    enabled: sliderAvailable && selectedBboxIndex !== null,
  });

  const {
    data: similarityData,
    isLoading: similarityLoading,
    error: similarityError,
  } = useQuery({
    queryKey: [
      'frozen-vs-finetuned-similarity',
      imageId,
      model,
      finetunedModelKey,
      effectiveLayer,
      selectedBbox,
    ],
    queryFn: async () => {
      if (!selectedBbox) {
        return null;
      }

      const bboxPayload = {
        left: selectedBbox.left,
        top: selectedBbox.top,
        width: selectedBbox.width,
        height: selectedBbox.height,
        label: selectedBbox.label_name || undefined,
      };

      const [frozen, finetuned] = await Promise.all([
        attentionAPI.getSimilarity(imageId, bboxPayload, model, effectiveLayer),
        attentionAPI.getSimilarity(imageId, bboxPayload, finetunedModelKey, effectiveLayer),
      ]);

      return { frozen, finetuned };
    },
    enabled: sliderAvailable && Boolean(selectedBbox),
    retry: false,
  });

  const frozenSimilarity = similarityData?.frozen?.similarity;
  const frozenPatchGrid = similarityData?.frozen?.patch_grid as [number, number] | undefined;
  const finetunedSimilarity = similarityData?.finetuned?.similarity;
  const finetunedPatchGrid = similarityData?.finetuned?.patch_grid as [number, number] | undefined;

  const frozenSimilarityHeatmapUrl = useMemo(() => {
    if (!frozenSimilarity || !frozenPatchGrid) {
      return null;
    }
    try {
      return renderHeatmap({
        similarity: frozenSimilarity,
        patchGrid: frozenPatchGrid,
        opacity: heatmapOpacity,
        style: heatmapStyle,
      });
    } catch {
      return null;
    }
  }, [frozenPatchGrid, frozenSimilarity, heatmapOpacity, heatmapStyle]);

  const finetunedSimilarityHeatmapUrl = useMemo(() => {
    if (!finetunedSimilarity || !finetunedPatchGrid) {
      return null;
    }
    try {
      return renderHeatmap({
        similarity: finetunedSimilarity,
        patchGrid: finetunedPatchGrid,
        opacity: heatmapOpacity,
        style: heatmapStyle,
      });
    } catch {
      return null;
    }
  }, [finetunedPatchGrid, finetunedSimilarity, heatmapOpacity, heatmapStyle]);

  const frozenSimilarityStats = useMemo(() => {
    if (!frozenSimilarity) {
      return null;
    }
    return computeSimilarityStats(frozenSimilarity);
  }, [frozenSimilarity]);

  const finetunedSimilarityStats = useMemo(() => {
    if (!finetunedSimilarity) {
      return null;
    }
    return computeSimilarityStats(finetunedSimilarity);
  }, [finetunedSimilarity]);

  const showSimilarityHeatmaps =
    Boolean(selectedBbox) &&
    Boolean(frozenSimilarityHeatmapUrl) &&
    Boolean(finetunedSimilarityHeatmapUrl) &&
    !similarityLoading &&
    !similarityError;

  if (modelsLoading || isLoading) {
    return (
      <div className="bg-gray-50 border border-gray-200 rounded-lg p-4 text-sm text-gray-600">
        Loading frozen vs fine-tuned availability...
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-sm text-red-700">
        Failed to load frozen vs fine-tuned comparison data.
      </div>
    );
  }

  if (!data) {
    return null;
  }

  if (!sliderAvailable) {
    return (
      <div className="space-y-4">
        {frozenUrl && (
          <div className="relative">
            <img
              src={frozenUrl}
              alt={`${model} frozen attention`}
              className="w-full h-auto rounded-lg"
            />
            <div className="absolute bottom-2 left-2 px-2 py-1 bg-black/50 text-white text-xs rounded">
              {model} (Frozen/Pretrained)
            </div>
          </div>
        )}

        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
          <h4 className="font-medium text-yellow-800">Fine-tuned Overlay Unavailable</h4>
          <p className="text-sm text-yellow-700 mt-1">
            {data.finetuned.note}
          </p>
          <p className="text-sm text-yellow-700 mt-1">
            The slider comparison appears automatically when both frozen and fine-tuned overlays are cached.
          </p>
        </div>
      </div>
    );
  }

  const deltaIoU =
    bboxMetrics && typeof bboxMetrics.finetuned.iou === 'number' && typeof bboxMetrics.frozen.iou === 'number'
      ? bboxMetrics.finetuned.iou - bboxMetrics.frozen.iou
      : null;
  const deltaCoverage =
    bboxMetrics &&
    typeof bboxMetrics.finetuned.coverage === 'number' &&
    typeof bboxMetrics.frozen.coverage === 'number'
      ? bboxMetrics.finetuned.coverage - bboxMetrics.frozen.coverage
      : null;

  const deltaTone =
    deltaIoU === null ? 'text-gray-700 bg-gray-100 border-gray-200'
    : deltaIoU > 0 ? 'text-green-700 bg-green-50 border-green-200'
    : deltaIoU < 0 ? 'text-red-700 bg-red-50 border-red-200'
    : 'text-gray-700 bg-gray-100 border-gray-200';

  return (
    <div className="space-y-4">
      <div className="flex justify-between text-sm text-gray-600">
        <span>Frozen (Pretrained)</span>
        <span>
          {resolvedStrategy ? `Fine-tuned (${resolvedStrategy})` : 'Fine-tuned'}
        </span>
      </div>

      <div className="rounded-lg border border-sky-100 bg-sky-50/70 px-4 py-3 text-sm text-sky-900">
        Clicking a bounding box switches the slider from the cached global overlays to bbox-conditioned
        similarity heatmaps, using the selected architectural feature as the query for both frozen and
        fine-tuned variants.
      </div>

      <div className="relative mx-auto w-full max-w-3xl">
        <ReactCompareSlider
          itemOne={
            <CompareCanvas
              imageSrc={showSimilarityHeatmaps ? originalUrl : frozenUrl!}
              imageAlt={showSimilarityHeatmaps ? 'Frozen similarity heatmap' : 'Frozen model attention'}
              overlaySrc={showSimilarityHeatmaps ? frozenSimilarityHeatmapUrl : null}
              overlayAlt="Frozen similarity overlay"
            />
          }
          itemTwo={
            <CompareCanvas
              imageSrc={showSimilarityHeatmaps ? originalUrl : finetunedUrl!}
              imageAlt={showSimilarityHeatmaps ? 'Fine-tuned similarity heatmap' : 'Fine-tuned model attention'}
              overlaySrc={showSimilarityHeatmaps ? finetunedSimilarityHeatmapUrl : null}
              overlayAlt="Fine-tuned similarity overlay"
            />
          }
          className="aspect-square overflow-hidden rounded-lg"
          position={50}
        />
        {similarityLoading && selectedBbox && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/20 pointer-events-none">
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

      <p className="text-xs text-gray-500 text-center">
        {showSimilarityHeatmaps
          ? 'Drag slider to compare bbox-conditioned similarity heatmaps'
          : `Drag slider to compare frozen vs fine-tuned attention${showBboxes ? ' with annotated boxes' : ''}`}
      </p>
      {data.strategy === 'linear_probe' && (
        <p className="text-xs text-amber-700 text-center">
          Linear probe trains only the classifier head, so attention maps can look identical.
        </p>
      )}
      {similarityError && selectedBbox && (
        <p className="text-xs text-amber-700 text-center">
          Similarity heatmaps are unavailable for this selection, so the view stays on the global overlays. Run
          feature-cache generation for both frozen and fine-tuned variants to enable gallery-style bbox comparison.
        </p>
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
                    Failed to load bbox metrics for this comparison. Check that both frozen and fine-tuned metrics
                    are cached.
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
                <p className="font-medium text-gray-900">Frozen similarity</p>
                <p>
                  {frozenSimilarityStats
                    ? `min ${frozenSimilarityStats.min.toFixed(2)} | max ${frozenSimilarityStats.max.toFixed(2)}`
                    : 'No stats available'}
                </p>
              </div>
              <div className="flex items-center gap-2">
                <span>Low similarity</span>
                <img src={legendUrl} alt="Similarity scale" className="h-4 rounded" />
                <span>High similarity</span>
              </div>
              <div className="space-y-1 text-right">
                <p className="font-medium text-gray-900">Fine-tuned similarity</p>
                <p>
                  {finetunedSimilarityStats
                    ? `min ${finetunedSimilarityStats.min.toFixed(2)} | max ${finetunedSimilarityStats.max.toFixed(2)}`
                    : 'No stats available'}
                </p>
              </div>
            </div>
          )}

          {selectedBbox && bboxMetrics && !bboxMetricsLoading && !bboxMetricsError && (
            <div className="grid gap-3 md:grid-cols-2">
              <div className="rounded-lg border border-gray-200 p-3">
                <p className="text-xs font-semibold uppercase tracking-wide text-gray-500">Frozen</p>
                <p className="mt-2 text-sm text-gray-900">IoU: {bboxMetrics.frozen.iou.toFixed(3)}</p>
                <p className="text-sm text-gray-700">Coverage: {bboxMetrics.frozen.coverage.toFixed(3)}</p>
              </div>
              <div className="rounded-lg border border-gray-200 p-3">
                <p className="text-xs font-semibold uppercase tracking-wide text-gray-500">
                  Fine-tuned{resolvedStrategy ? ` (${resolvedStrategy})` : ''}
                </p>
                <p className="mt-2 text-sm text-gray-900">IoU: {bboxMetrics.finetuned.iou.toFixed(3)}</p>
                <p className="text-sm text-gray-700">Coverage: {bboxMetrics.finetuned.coverage.toFixed(3)}</p>
              </div>
            </div>
          )}

          {!selectedBbox && (
            <div className="rounded-lg border border-dashed border-gray-300 bg-gray-50 p-3 text-sm text-gray-600">
              Global overlays often look similar. Selecting a feature switches the view to the same bbox-conditioned
              similarity treatment used in the gallery, then asks the more useful question: did fine-tuning increase
              or decrease attention on this architectural cue?
            </div>
          )}
        </div>
      )}
    </div>
  );
}
