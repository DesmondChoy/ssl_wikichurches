/**
 * Variant comparison slider for frozen-vs-fine-tuned and strategy-vs-strategy views.
 */

import { useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { ReactCompareSlider } from 'react-compare-slider';
import { attentionAPI, comparisonAPI, imagesAPI, metricsAPI } from '../../api/client';
import { useQ2Summary } from '../../hooks/useMetrics';
import { Card, CardContent } from '../ui/Card';
import { InteractiveBboxOverlay } from '../attention/InteractiveBboxOverlay';
import { useModels } from '../../hooks/useAttention';
import { useHeatmapOpacity, useHeatmapStyle } from '../../store/viewStore';
import { computeSimilarityStats, renderHeatmap, renderHeatmapLegend } from '../../utils/renderHeatmap';
import type { BoundingBox, ComparisonVariant, IoUResult, Q2StrategyComparison, Q2StrategyResult } from '../../types';

type ComparisonMode = 'frozen' | 'methods';
type ViewMode = 'side-by-side' | 'slider';

interface FrozenVsFinetunedProps {
  imageId: string;
  model: string;
  layer: number;
  percentile: number;
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

function formatSigned(value: number | null | undefined, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return 'n/a';
  }
  return `${value >= 0 ? '+' : ''}${value.toFixed(digits)}`;
}

function formatMetric(value: number | null | undefined, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return 'n/a';
  }
  return value.toFixed(digits);
}

function findStrategyPair(
  comparisons: Q2StrategyComparison[],
  leftStrategy: string,
  rightStrategy: string
) {
  return comparisons.find(
    (comparison) =>
      (comparison.strategy_a === leftStrategy && comparison.strategy_b === rightStrategy) ||
      (comparison.strategy_a === rightStrategy && comparison.strategy_b === leftStrategy)
  );
}

function VariantMetricSummary({ metrics }: { metrics: IoUResult }) {
  return (
    <div className="space-y-1 text-sm">
      <p className="text-gray-900">IoU: {formatMetric(metrics.iou)}</p>
      <p className="text-gray-700">Coverage: {formatMetric(metrics.coverage)}</p>
      <p className="text-gray-700">MSE: {formatMetric(metrics.mse, 4)}</p>
      <p className="text-gray-700">KL: {formatMetric(metrics.kl, 4)}</p>
      <p className="text-gray-700">EMD: {formatMetric(metrics.emd, 4)}</p>
    </div>
  );
}

function ExperimentSummaryCard({
  title,
  strategyId,
  summary,
}: {
  title: string;
  strategyId?: string | null;
  summary: Q2StrategyResult;
}) {
  return (
    <Card>
      <CardContent className="space-y-3">
        <div>
          <p className="text-xs font-semibold uppercase tracking-wide text-gray-500">Experiment summary</p>
          <p className="text-sm font-medium text-gray-900">{title}</p>
          {strategyId && (
            <p className="text-xs text-gray-600">{strategyLabel(strategyId)} across 139 annotated images</p>
          )}
        </div>
        <div className="grid gap-3 sm:grid-cols-2">
          <div>
            <p className="text-xs uppercase tracking-wide text-gray-500">Mean ΔIoU</p>
            <p className="text-2xl font-semibold text-gray-900">{formatSigned(summary.mean_delta_iou)}</p>
            <p className="text-xs text-gray-600">
              CI [{summary.delta_ci_lower.toFixed(3)}, {summary.delta_ci_upper.toFixed(3)}]
            </p>
          </div>
          <div className="space-y-1 text-sm">
            <p>
              <span className="font-medium">Frozen IoU:</span> {formatMetric(summary.frozen_mean_iou)}
            </p>
            <p>
              <span className="font-medium">Fine-tuned IoU:</span> {formatMetric(summary.finetuned_mean_iou)}
            </p>
            <p>
              <span className="font-medium">Retention:</span> {formatMetric(summary.iou_retention_ratio)}
            </p>
            <p>
              <span className="font-medium">Effect size:</span> {formatMetric(summary.cohens_d)}
            </p>
            <p>
              <span className="font-medium">Significance:</span>{' '}
              {summary.significant ? 'Holm-corrected significant' : 'Not significant'}
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function PairwiseSummaryCard({ comparison }: { comparison: Q2StrategyComparison }) {
  return (
    <Card>
      <CardContent className="space-y-3">
        <div>
          <p className="text-xs font-semibold uppercase tracking-wide text-gray-500">Experiment summary</p>
          <p className="text-sm font-medium text-gray-900">Cross-strategy comparison</p>
          <p className="text-xs text-gray-600">
            {strategyLabel(comparison.strategy_a)} vs {strategyLabel(comparison.strategy_b)}
          </p>
        </div>
        <div className="grid gap-3 sm:grid-cols-2">
          <div>
            <p className="text-xs uppercase tracking-wide text-gray-500">ΔΔIoU</p>
            <p className="text-2xl font-semibold text-gray-900">
              {formatSigned(comparison.mean_delta_difference)}
            </p>
            <p className="text-xs text-gray-600">
              Positive means {strategyLabel(comparison.strategy_a)} shifts more than {strategyLabel(comparison.strategy_b)}.
            </p>
          </div>
          <div className="space-y-1 text-sm">
            <p>
              <span className="font-medium">Effect size:</span> {formatMetric(comparison.cohens_d)}
            </p>
            <p>
              <span className="font-medium">p-value:</span> {formatMetric(comparison.corrected_p_value ?? comparison.p_value, 4)}
            </p>
            <p>
              <span className="font-medium">Significance:</span>{' '}
              {comparison.significant ? 'Holm-corrected significant' : 'Not significant'}
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export function FrozenVsFinetuned({
  imageId,
  model,
  layer,
  percentile,
  strategy,
  strategyA = 'linear_probe',
  strategyB = 'full',
  mode = 'frozen',
  bboxes = [],
  showBboxes = true,
}: FrozenVsFinetunedProps) {
  const [selectedBboxIndex, setSelectedBboxIndex] = useState<number | null>(null);
  const [viewMode, setViewMode] = useState<ViewMode>('side-by-side');
  const [q2SummaryExpanded, setQ2SummaryExpanded] = useState(false);
  const { data: modelsData, isLoading: modelsLoading } = useModels();
  const heatmapOpacity = useHeatmapOpacity();
  const heatmapStyle = useHeatmapStyle();
  const maxLayer = modelsData?.num_layers_per_model?.[model];
  const effectiveLayer = typeof maxLayer === 'number' ? Math.min(layer, maxLayer - 1) : layer;
  const percentileKey = String(percentile);
  const percentileCopy = `Top ${100 - percentile}% attention`;

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

  const { data: q2Summary, isLoading: q2SummaryLoading } = useQ2Summary(percentile, model);

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
      percentile,
      selectedBboxIndex,
      data?.left.model_key,
      data?.right.model_key,
    ],
    queryFn: async () => {
      if (selectedBboxIndex === null || !data) {
        return null;
      }

      const [left, right] = await Promise.all([
        metricsAPI.getBboxMetrics(imageId, data.left.model_key, effectiveLayer, selectedBboxIndex, percentile),
        metricsAPI.getBboxMetrics(imageId, data.right.model_key, effectiveLayer, selectedBboxIndex, percentile),
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

  const selectedFrozenStrategy = mode === 'frozen' ? data?.right.strategy ?? null : null;
  const q2ModelSummary = q2Summary?.models?.[model] ?? {};
  const q2StrategyComparisons = q2Summary?.strategy_comparisons?.[model]?.[percentileKey] ?? [];
  const frozenSummary =
    selectedFrozenStrategy && q2ModelSummary[selectedFrozenStrategy]
      ? q2ModelSummary[selectedFrozenStrategy][percentileKey]
      : null;
  const leftStrategySummary =
    mode === 'methods' && data?.left.strategy && q2ModelSummary[data.left.strategy]
      ? q2ModelSummary[data.left.strategy][percentileKey]
      : null;
  const rightStrategySummary =
    mode === 'methods' && data?.right.strategy && q2ModelSummary[data.right.strategy]
      ? q2ModelSummary[data.right.strategy][percentileKey]
      : null;
  const methodPairSummary =
    mode === 'methods' && data?.left.strategy && data?.right.strategy
      ? findStrategyPair(q2StrategyComparisons, data.left.strategy, data.right.strategy)
      : null;
  const setBboxIndex = (index: number | null) => {
    setSelectedBboxIndex(index);
  };
  const hasQ2Content =
    (!q2SummaryLoading && mode === 'frozen' && frozenSummary) ||
    (!q2SummaryLoading &&
      mode === 'methods' &&
      (leftStrategySummary || rightStrategySummary || methodPairSummary));

  const experimentSummary = (
    <div className="space-y-3">
      {q2SummaryLoading && (
        <div className="rounded-lg border border-gray-200 bg-gray-50 p-4 text-sm text-gray-600">
          Loading experiment summary...
        </div>
      )}

      {hasQ2Content && (
        <Card>
          <button
            type="button"
            onClick={() => setQ2SummaryExpanded((prev) => !prev)}
            className="flex w-full items-center justify-between px-4 py-3 text-left hover:bg-gray-50"
            aria-expanded={q2SummaryExpanded}
          >
            <span className="text-sm font-medium text-gray-900">Experiment summary (aggregate)</span>
            <span className="text-gray-500" aria-hidden>
              {q2SummaryExpanded ? '▼' : '▶'}
            </span>
          </button>
          {q2SummaryExpanded && (
            <CardContent className="border-t border-gray-200 pt-3">
              <div className="space-y-3">
                {mode === 'frozen' && frozenSummary && (
                  <ExperimentSummaryCard
                    title="Frozen vs fine-tuned shift"
                    strategyId={selectedFrozenStrategy}
                    summary={frozenSummary}
                  />
                )}

                {mode === 'methods' && (leftStrategySummary || rightStrategySummary || methodPairSummary) && (
                  <div className="grid gap-4 lg:grid-cols-3">
                    {leftStrategySummary && (
                      <ExperimentSummaryCard
                        title={`${data?.left.label ?? 'Left strategy'} vs frozen baseline`}
                        strategyId={data?.left.strategy}
                        summary={leftStrategySummary}
                      />
                    )}
                    {rightStrategySummary && (
                      <ExperimentSummaryCard
                        title={`${data?.right.label ?? 'Right strategy'} vs frozen baseline`}
                        strategyId={data?.right.strategy}
                        summary={rightStrategySummary}
                      />
                    )}
                    {methodPairSummary && <PairwiseSummaryCard comparison={methodPairSummary} />}
                  </div>
                )}
              </div>
            </CardContent>
          )}
        </Card>
      )}
    </div>
  );

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
        {experimentSummary}
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
            Side-by-side and slider comparison both require cached overlays for the compared variants.
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
  const deltaMse =
    bboxMetrics &&
    typeof bboxMetrics.right.mse === 'number' &&
    typeof bboxMetrics.left.mse === 'number'
      ? bboxMetrics.right.mse - bboxMetrics.left.mse
      : null;
  const deltaKl =
    bboxMetrics &&
    typeof bboxMetrics.right.kl === 'number' &&
    typeof bboxMetrics.left.kl === 'number'
      ? bboxMetrics.right.kl - bboxMetrics.left.kl
      : null;
  const deltaEmd =
    bboxMetrics &&
    typeof bboxMetrics.right.emd === 'number' &&
    typeof bboxMetrics.left.emd === 'number'
      ? bboxMetrics.right.emd - bboxMetrics.left.emd
      : null;

  const deltaTone =
    deltaIoU === null ? 'border-gray-200 bg-gray-100 text-gray-700'
    : deltaIoU > 0 ? 'border-green-200 bg-green-50 text-green-700'
    : deltaIoU < 0 ? 'border-red-200 bg-red-50 text-red-700'
    : 'border-gray-200 bg-gray-100 text-gray-700';

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

      {experimentSummary}

      {viewMode === 'side-by-side' && (
        <>
          <div className="grid grid-cols-2 gap-4">
            {/* Left variant */}
            <Card>
              <CardContent className="p-0">
                <div className="border-b border-gray-200 px-3 py-2">
                  <p className="text-sm font-medium text-gray-900">{data.left.label}</p>
                  <p className="text-xs text-gray-500">
                    {showSimilarityHeatmaps ? 'Feature-local similarity' : 'Global overlay'} · {percentileCopy}
                  </p>
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
                <div className="border-t border-gray-200 px-3 py-2 text-sm text-gray-600">
                  {selectedBbox && bboxMetrics && !bboxMetricsLoading ? (
                    <VariantMetricSummary metrics={bboxMetrics.left} />
                  ) : selectedBbox && bboxMetricsLoading ? (
                    <span>Loading metrics…</span>
                  ) : (
                    <span>Click a feature to see local metrics for {percentileCopy.toLowerCase()}.</span>
                  )}
                </div>
              </CardContent>
            </Card>

            {/* Right variant */}
            <Card>
              <CardContent className="p-0">
                <div className="border-b border-gray-200 px-3 py-2">
                  <p className="text-sm font-medium text-gray-900">{data.right.label}</p>
                  <p className="text-xs text-gray-500">
                    {showSimilarityHeatmaps ? 'Feature-local similarity' : 'Global overlay'} · {percentileCopy}
                  </p>
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
                <div className="border-t border-gray-200 px-3 py-2 text-sm text-gray-600">
                  {selectedBbox && bboxMetrics && !bboxMetricsLoading ? (
                    <VariantMetricSummary metrics={bboxMetrics.right} />
                  ) : selectedBbox && bboxMetricsLoading ? (
                    <span>Loading metrics…</span>
                  ) : (
                    <span>Click a feature to see local metrics for {percentileCopy.toLowerCase()}.</span>
                  )}
                </div>
              </CardContent>
            </Card>
          </div>
          {data.linearProbeInvolved && (
            <p className="text-center text-xs text-amber-700">
              Linear probe is the no-backbone-change baseline, so attention differences are expected to stay small.
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
            similarity heatmaps. Local IoU uses the active threshold ({percentileCopy.toLowerCase()}), while
            MSE/KL/EMD compare dense heatmaps directly.
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
              Linear probe is the no-backbone-change baseline, so attention differences are expected to stay small.
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
              <p className="text-sm font-medium text-gray-900">Feature-local metrics</p>
              <p className="text-xs text-gray-600">
                Click a box on the image or choose a feature below to compare local alignment at {percentileCopy.toLowerCase()}.
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
                  region as the similarity query. IoU and coverage use {percentileCopy.toLowerCase()}, while
                  MSE/KL/EMD compare dense heatmaps inside the same bbox.
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
                    <div className="mt-3 grid gap-2 text-xs sm:grid-cols-3">
                      <p>MSE Δ: {formatSigned(deltaMse, 4)}</p>
                      <p>KL Δ: {formatSigned(deltaKl, 4)}</p>
                      <p>EMD Δ: {formatSigned(deltaEmd, 4)}</p>
                    </div>
                    <p className="mt-2 text-xs">
                      Lower is better for MSE, KL, and EMD.
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
                <div className="mt-2">
                  <VariantMetricSummary metrics={bboxMetrics.left} />
                </div>
              </div>
              <div className="rounded-lg border border-gray-200 p-3">
                <p className="text-xs font-semibold uppercase tracking-wide text-gray-500">{data.right.label}</p>
                <div className="mt-2">
                  <VariantMetricSummary metrics={bboxMetrics.right} />
                </div>
              </div>
            </div>
          )}

          {!selectedBbox && (
            <div className="rounded-lg border border-dashed border-gray-300 bg-gray-50 p-3 text-sm text-gray-600">
              Global overlays often look similar. Selecting a feature switches the view to bbox-conditioned
              similarity heatmaps and feature-local metrics so you can compare the same architectural cue across
              the chosen variants.
            </div>
          )}
        </div>
      )}
    </div>
  );
}
