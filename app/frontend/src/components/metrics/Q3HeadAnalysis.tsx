import { useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';

import { useModels } from '../../hooks/useAttention';
import { useHeadExemplars, useHeadFeatureMatrix, useHeadRanking } from '../../hooks/useMetrics';
import { ANALYSIS_METRIC_METADATA, ANALYSIS_METRIC_OPTIONS, formatMetricValue } from '../../constants/metricMetadata';
import { PERCENTILE_OPTIONS } from '../../constants/percentiles';
import { buildImageDetailQ3Href } from '../../constants/q3Routing';
import {
  Q3_DEFAULTS,
  Q3_PRIMARY_MODELS,
  Q3_VARIANT_OPTIONS,
  formatQ3ScopeOptionLabel,
  getQ3SelectionHelperText,
  getQ3VariantScopeStatus,
} from '../../constants/q3Scope';
import { Card, CardContent, CardHeader } from '../ui/Card';
import { Select } from '../ui/Select';
import { Q3ExemplarPicker, type Q3ExemplarPickerRequest } from './Q3ExemplarPicker';
import { Q3ScopeChip, Q3StudyScopeCallout } from './Q3ScopeFraming';
import type { AnalysisMetric, CompareVariantId, HeadExemplarCandidate, MetricDirection } from '../../types';

const ITEMS_PER_PAGE = 20;
const EXEMPLAR_LIMIT = 12;

interface HeatmapRange {
  min: number;
  max: number;
}

interface HeatmapCellPreview {
  head: number;
  featureLabel: number;
  featureName: string;
  score: number;
}

function getMetricTone(metric: AnalysisMetric, value: number | null): string {
  if (value === null || value === undefined) {
    return 'text-gray-400';
  }

  const direction = ANALYSIS_METRIC_METADATA[metric].direction;
  if (direction === 'higher') {
    if (value >= 0.6) return 'text-green-700 bg-green-50';
    if (value >= 0.4) return 'text-amber-700 bg-amber-50';
    return 'text-rose-700 bg-rose-50';
  }

  if (value <= 0.05) return 'text-green-700 bg-green-50';
  if (value <= 0.15) return 'text-amber-700 bg-amber-50';
  return 'text-rose-700 bg-rose-50';
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function interpolateColor(start: number, end: number, ratio: number): number {
  return Math.round(start + (end - start) * ratio);
}

function getHeatmapIntensity(score: number, range: HeatmapRange | null, direction: MetricDirection): number {
  if (!range || range.max === range.min) {
    return 1;
  }

  const normalized = clamp01((score - range.min) / (range.max - range.min));
  return direction === 'higher' ? normalized : 1 - normalized;
}

function getHeatmapCellStyle(score: number | null, range: HeatmapRange | null, direction: MetricDirection) {
  if (score === null) {
    return {
      backgroundColor: '#f8fafc',
      borderColor: '#e2e8f0',
    };
  }

  const intensity = getHeatmapIntensity(score, range, direction);
  return {
    backgroundColor: `rgb(${interpolateColor(240, 12, intensity)}, ${interpolateColor(249, 74, intensity)}, ${interpolateColor(255, 110, intensity)})`,
    borderColor: `rgb(${interpolateColor(186, 8, intensity)}, ${interpolateColor(230, 145, intensity)}, ${interpolateColor(253, 178, intensity)})`,
  };
}

function getHeatmapDotClass(score: number | null, range: HeatmapRange | null, direction: MetricDirection): string {
  if (score === null) {
    return 'border-slate-300 bg-slate-300';
  }

  const intensity = getHeatmapIntensity(score, range, direction);
  return intensity >= 0.55
    ? 'border-white/80 bg-white/80'
    : 'border-slate-500/30 bg-slate-700/20';
}

function formatHoverReadout(metric: AnalysisMetric, preview: HeatmapCellPreview): string {
  const metricLabel = ANALYSIS_METRIC_METADATA[metric].shortLabel;
  return `${preview.featureName} · H${preview.head} · ${metricLabel}: ${formatMetricValue(metric, preview.score)}`;
}

export function Q3HeadAnalysis() {
  const navigate = useNavigate();
  const { data: modelsData } = useModels();
  const [model, setModel] = useState(Q3_DEFAULTS.model);
  const [variant, setVariant] = useState<CompareVariantId>(Q3_DEFAULTS.variant);
  const [layer, setLayer] = useState(Q3_DEFAULTS.layer);
  const [metric, setMetric] = useState<AnalysisMetric>(Q3_DEFAULTS.metric);
  const [percentile, setPercentile] = useState(Q3_DEFAULTS.percentile);
  const [searchQuery, setSearchQuery] = useState('');
  const [showCount, setShowCount] = useState(ITEMS_PER_PAGE);
  const [drilldownRequest, setDrilldownRequest] = useState<Q3ExemplarPickerRequest | null>(null);
  const [hoveredCell, setHoveredCell] = useState<HeatmapCellPreview | null>(null);

  const availableQ3Models = useMemo(() => {
    const visibleModels = (modelsData?.models ?? []).filter((value) =>
      Q3_PRIMARY_MODELS.includes(value as (typeof Q3_PRIMARY_MODELS)[number]),
    );
    return visibleModels.length > 0 ? visibleModels : [...Q3_PRIMARY_MODELS];
  }, [modelsData?.models]);

  const resolvedModel = availableQ3Models.includes(model) ? model : availableQ3Models[0];
  const metricMetadata = ANALYSIS_METRIC_METADATA[metric];
  const effectivePercentile = metricMetadata.thresholdFree ? 90 : percentile;
  const maxLayer = modelsData?.num_layers_per_model?.[resolvedModel]
    ? modelsData.num_layers_per_model[resolvedModel] - 1
    : Q3_DEFAULTS.layer;
  const resolvedLayer = Math.min(layer, maxLayer);
  const variantScopeStatus = getQ3VariantScopeStatus(variant);
  const selectionHelperText = getQ3SelectionHelperText('primary', variantScopeStatus);

  const rankingQuery = useHeadRanking(resolvedModel, resolvedLayer, effectivePercentile, metric, variant);
  const matrixQuery = useHeadFeatureMatrix(resolvedModel, resolvedLayer, effectivePercentile, metric, variant);
  const activeDrilldownRequest = useMemo(() => {
    if (!drilldownRequest) {
      return null;
    }
    if (
      drilldownRequest.model !== resolvedModel
      || drilldownRequest.variant !== variant
      || drilldownRequest.layer !== resolvedLayer
      || drilldownRequest.metric !== metric
      || drilldownRequest.percentile !== effectivePercentile
    ) {
      return null;
    }
    return drilldownRequest;
  }, [drilldownRequest, resolvedModel, variant, resolvedLayer, metric, effectivePercentile]);

  const activeHoveredCell = useMemo(() => {
    if (!hoveredCell) {
      return null;
    }
    const feature = (matrixQuery.data?.features ?? []).find((entry) => entry.feature_label === hoveredCell.featureLabel);
    const headIndex = (matrixQuery.data?.heads ?? []).indexOf(hoveredCell.head);
    if (!feature || headIndex < 0) {
      return null;
    }
    const score = feature.scores[headIndex];
    if (score === null) {
      return null;
    }
    return {
      ...hoveredCell,
      featureName: feature.feature_name,
      score,
    };
  }, [hoveredCell, matrixQuery.data?.features, matrixQuery.data?.heads]);

  const exemplarQuery = useHeadExemplars(
    activeDrilldownRequest?.model ?? Q3_DEFAULTS.model,
    activeDrilldownRequest?.head ?? null,
    activeDrilldownRequest?.layer ?? Q3_DEFAULTS.layer,
    activeDrilldownRequest?.percentile ?? effectivePercentile,
    activeDrilldownRequest?.metric ?? metric,
    activeDrilldownRequest?.variant ?? variant,
    {
      featureLabel: activeDrilldownRequest?.featureLabel,
      limit: EXEMPLAR_LIMIT,
      enabled: activeDrilldownRequest !== null,
    },
  );

  const filteredFeatures = useMemo(() => {
    const features = matrixQuery.data?.features ?? [];
    if (!searchQuery.trim()) {
      return features;
    }
    const query = searchQuery.toLowerCase();
    return features.filter((feature) => feature.feature_name.toLowerCase().includes(query));
  }, [matrixQuery.data?.features, searchQuery]);

  const heatmapRange = useMemo<HeatmapRange | null>(() => {
    const scores = (matrixQuery.data?.features ?? []).flatMap((feature) =>
      feature.scores.filter((score): score is number => score !== null),
    );
    if (scores.length === 0) {
      return null;
    }
    return {
      min: Math.min(...scores),
      max: Math.max(...scores),
    };
  }, [matrixQuery.data?.features]);

  const visibleFeatures = filteredFeatures.slice(0, showCount);
  const hasMore = showCount < filteredFeatures.length;
  const heatmapDirection = matrixQuery.data?.direction ?? metricMetadata.direction;
  const selectedHead = activeDrilldownRequest?.head ?? null;
  const selectedFeatureLabel = activeDrilldownRequest?.origin === 'feature' ? (activeDrilldownRequest.featureLabel ?? null) : null;

  const modelOptions = availableQ3Models.map((value) => ({
    value,
    label: value,
  }));
  const variantOptions = Q3_VARIANT_OPTIONS.map((option) => ({
    value: option.value,
    label: formatQ3ScopeOptionLabel(option.label, getQ3VariantScopeStatus(option.value)),
  }));
  const exemplarError = exemplarQuery.error instanceof Error ? exemplarQuery.error.message : null;

  const openRankingDrilldown = (head: number, score: number | null) => {
    setDrilldownRequest({
      origin: 'ranking',
      model: resolvedModel,
      variant,
      layer: resolvedLayer,
      head,
      metric,
      percentile: effectivePercentile,
      score,
    });
  };

  const openFeatureDrilldown = (head: number, featureLabel: number, featureName: string, score: number) => {
    setDrilldownRequest({
      origin: 'feature',
      model: resolvedModel,
      variant,
      layer: resolvedLayer,
      head,
      metric,
      percentile: effectivePercentile,
      featureLabel,
      featureName,
      score,
    });
  };

  const handleSelectCandidate = (candidate: HeadExemplarCandidate) => {
    if (!activeDrilldownRequest) {
      return;
    }

    navigate(
      buildImageDetailQ3Href(candidate.image_id, {
        model: activeDrilldownRequest.model,
        variant: activeDrilldownRequest.variant,
        layer: activeDrilldownRequest.layer,
        head: activeDrilldownRequest.head,
        metric: activeDrilldownRequest.metric,
        mode: 'head_attention',
        showBboxes: true,
        bboxIndex: activeDrilldownRequest.origin === 'feature' ? candidate.default_bbox_index : null,
        featureLabel: activeDrilldownRequest.featureLabel ?? null,
        featureName: activeDrilldownRequest.featureName ?? null,
      }),
    );
  };

  const hoverReadout = activeHoveredCell
    ? `Hover: ${formatHoverReadout(metric, activeHoveredCell)}`
    : 'Hover or focus a heatmap cell to inspect the exact score before loading representative images.';

  const selectionSummary = (() => {
    if (!activeDrilldownRequest) {
      return 'Select a head or click a heatmap cell to load representative images inline below the matrix.';
    }
    if (activeDrilldownRequest.origin === 'feature') {
      const featureLabel = activeDrilldownRequest.featureName ?? `feature ${activeDrilldownRequest.featureLabel}`;
      const scoreText = activeDrilldownRequest.score === null || activeDrilldownRequest.score === undefined
        ? ''
        : ` · ${metricMetadata.shortLabel} ${formatMetricValue(metric, activeDrilldownRequest.score)}`;
      return `Selected cell: ${featureLabel} · H${activeDrilldownRequest.head}${scoreText}.`;
    }
    const scoreText = activeDrilldownRequest.score === null || activeDrilldownRequest.score === undefined
      ? ''
      : ` · ${metricMetadata.shortLabel} ${formatMetricValue(metric, activeDrilldownRequest.score)}`;
    return `Selected head: Head ${activeDrilldownRequest.head}${scoreText}.`;
  })();

  return (
    <Card>
      <CardHeader>
        <div className="flex flex-col gap-2 lg:flex-row lg:items-start lg:justify-between">
          <div>
            <h3 className="font-semibold">Q3 Per-Head Specialization</h3>
            <p className="mt-1 text-sm text-gray-600">
              Use the heatmap to spot head-feature specialization patterns, then inspect representative images without leaving Dashboard Q3.
            </p>
          </div>
          <div className="text-xs text-gray-500">
            {rankingQuery.data?.method
              ? `Automatic method: ${rankingQuery.data.method}`
              : 'Automatic method selection'}
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-6">
        <Q3StudyScopeCallout
          context="dashboard"
          dataTestId="q3-scope-callout"
        />

        <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-5">
          <Select
            value={resolvedModel}
            onChange={setModel}
            options={modelOptions}
            label="Model"
          />
          <Select
            value={variant}
            onChange={(value) => setVariant(value as CompareVariantId)}
            options={variantOptions}
            label="Variant"
          />
          <Select
            value={resolvedLayer}
            onChange={(value) => setLayer(Number(value))}
            options={Array.from({ length: maxLayer + 1 }, (_, index) => ({
              value: index,
              label: `Layer ${index}`,
            }))}
            label="Layer"
          />
          <Select
            value={metric}
            onChange={(value) => setMetric(value as AnalysisMetric)}
            options={ANALYSIS_METRIC_OPTIONS}
            label="Metric"
          />
          <Select
            value={percentile}
            onChange={(value) => setPercentile(Number(value))}
            options={PERCENTILE_OPTIONS}
            label="Percentile"
            disabled={metricMetadata.thresholdFree}
          />
        </div>

        <div
          className="rounded-lg border border-slate-200 bg-white px-4 py-3"
          data-testid="q3-selection-scope"
        >
          <div className="text-xs font-semibold uppercase tracking-[0.12em] text-slate-500">
            Current Q3 workflow context
          </div>
          <div className="mt-2 flex flex-wrap gap-2">
            <span className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-sm text-slate-700">
              <span className="font-medium text-slate-900">Model</span>
              <span>{resolvedModel}</span>
              <Q3ScopeChip status="primary" dataTestId="q3-model-scope-chip" />
            </span>
            <span className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-sm text-slate-700">
              <span className="font-medium text-slate-900">Variant</span>
              <span>{Q3_VARIANT_OPTIONS.find((option) => option.value === variant)?.label ?? variant}</span>
              <Q3ScopeChip status={variantScopeStatus} dataTestId="q3-variant-scope-chip" />
            </span>
          </div>
          <p className="mt-2 text-xs text-slate-600" data-testid="q3-selection-helper">
            {selectionHelperText}
          </p>
        </div>

        {metricMetadata.thresholdFree && (
          <div className="rounded-lg border border-blue-100 bg-blue-50 px-4 py-3 text-sm text-blue-900">
            {metricMetadata.infoBanner}
          </div>
        )}

        {!rankingQuery.data?.supported || !matrixQuery.data?.supported ? (
          <div className="rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900">
            {rankingQuery.data?.reason || matrixQuery.data?.reason || 'Per-head analysis is not available for this selection.'}
          </div>
        ) : (rankingQuery.data?.heads.length ?? 0) === 0 && (matrixQuery.data?.features.length ?? 0) === 0 ? (
          <div className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-700">
            No Q3 per-head rows are available for this selection yet. Run the per-head precompute commands first.
          </div>
        ) : (
          <>
            <div className="rounded-lg border border-sky-100 bg-sky-50 px-4 py-3 text-sm text-sky-900">
              Darker cells indicate better-performing head-feature pairs for the current metric. Hover for the exact value, then click to load representative images inline.
            </div>

            <div className="grid grid-cols-1 gap-6 xl:grid-cols-[24rem_minmax(0,1fr)]">
              <div className="rounded-lg border border-gray-200">
                <div className="border-b border-gray-100 px-4 py-3">
                  <div className="font-medium text-gray-900">Head Ranking</div>
                  <div className="text-xs text-gray-500">
                    {metricMetadata.direction === 'higher' ? 'Higher scores rank better.' : 'Lower scores rank better.'}
                  </div>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead className="border-b border-gray-200 text-left text-xs text-gray-500">
                      <tr>
                        <th className="px-4 py-2 font-medium">Head</th>
                        <th className="px-4 py-2 font-medium text-right">{metricMetadata.shortLabel}</th>
                        <th className="px-4 py-2 font-medium text-right">Mean Rank</th>
                        <th className="px-4 py-2 font-medium text-right">Top-1</th>
                        <th className="px-4 py-2 font-medium text-right">Inspect</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-100">
                      {(rankingQuery.data?.heads ?? []).map((entry) => {
                        const isSelectedRankingHead = activeDrilldownRequest?.origin === 'ranking' && activeDrilldownRequest.head === entry.head;
                        return (
                          <tr key={entry.head} className={isSelectedRankingHead ? 'bg-primary-50/50' : undefined}>
                            <td className="px-4 py-2 font-medium text-gray-900">Head {entry.head}</td>
                            <td className="px-4 py-2 text-right">
                              <span className={`inline-block rounded px-2 py-0.5 text-xs font-medium ${getMetricTone(metric, entry.mean_score)}`}>
                                {formatMetricValue(metric, entry.mean_score)}
                              </span>
                            </td>
                            <td className="px-4 py-2 text-right text-gray-600">{entry.mean_rank.toFixed(2)}</td>
                            <td className="px-4 py-2 text-right text-gray-600">{entry.top1_count}</td>
                            <td className="px-4 py-2 text-right">
                              <button
                                type="button"
                                onClick={() => openRankingDrilldown(entry.head, entry.mean_score)}
                                className={`rounded-md border px-3 py-1.5 text-xs font-medium transition-colors ${
                                  isSelectedRankingHead
                                    ? 'border-primary-500 bg-primary-50 text-primary-700'
                                    : 'border-primary-200 bg-white text-primary-700 hover:border-primary-300 hover:bg-primary-50'
                                }`}
                              >
                                Inspect exemplar
                              </button>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>

              <div className="rounded-lg border border-gray-200">
                <div className="border-b border-gray-100 px-4 py-3">
                  <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
                    <div>
                      <div className="font-medium text-gray-900">Head × Feature Heatmap</div>
                      <div className="text-xs text-gray-500">
                        Showing {visibleFeatures.length} of {filteredFeatures.length} feature types
                      </div>
                    </div>
                    <input
                      type="text"
                      placeholder="Search features..."
                      value={searchQuery}
                      onChange={(event) => {
                        setSearchQuery(event.target.value);
                        setShowCount(ITEMS_PER_PAGE);
                      }}
                      className="w-full rounded-md border border-gray-200 px-3 py-2 text-sm focus:border-primary-500 focus:outline-none focus:ring-2 focus:ring-primary-500 lg:max-w-xs"
                    />
                  </div>

                  <div className="mt-4 grid gap-3">
                    <div
                      className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-700"
                      data-testid="q3-heatmap-hover-readout"
                    >
                      {hoverReadout}
                    </div>
                    <div
                      className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-700"
                      data-testid="q3-heatmap-selection-summary"
                    >
                      {selectionSummary}
                    </div>
                    <div className="flex flex-wrap items-center gap-3 text-xs text-slate-500">
                      <div
                        className="h-2.5 w-40 rounded-full border border-slate-200"
                        data-testid="q3-heatmap-legend"
                        style={{ background: 'linear-gradient(90deg, rgb(240, 249, 255) 0%, rgb(12, 74, 110) 100%)' }}
                      />
                      <span>Darker cells rank better for the selected metric.</span>
                    </div>
                  </div>
                </div>

                <div className="overflow-auto">
                  <table className="min-w-full border-separate border-spacing-0 text-sm">
                    <thead className="border-b border-gray-200 text-left text-xs text-gray-500">
                      <tr>
                        <th className="sticky left-0 z-20 bg-white px-4 py-3 font-medium">Feature</th>
                        {(matrixQuery.data?.heads ?? []).map((headIdx) => (
                          <th
                            key={headIdx}
                            data-testid={`q3-heatmap-head-${headIdx}`}
                            className={`px-2 py-3 text-center font-medium ${
                              selectedHead === headIdx ? 'bg-slate-100 text-slate-900' : 'text-slate-500'
                            }`}
                          >
                            H{headIdx}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {visibleFeatures.map((feature) => {
                        const isSelectedRow = selectedFeatureLabel === feature.feature_label;
                        return (
                          <tr key={feature.feature_label} className="border-b border-gray-100">
                            <td
                              data-testid={`q3-heatmap-feature-${feature.feature_label}`}
                              className={`sticky left-0 z-10 border-b border-gray-100 px-4 py-3 ${
                                isSelectedRow ? 'bg-sky-50' : 'bg-white'
                              }`}
                            >
                              <div className="font-medium text-gray-900">{feature.feature_name}</div>
                              <div className="text-xs text-gray-400">ID {feature.feature_label} • {feature.bbox_count} boxes</div>
                            </td>
                            {feature.scores.map((score, index) => {
                              const headIdx = matrixQuery.data?.heads[index] ?? index;
                              const isSelectedCell = isSelectedRow && selectedHead === headIdx;

                              return (
                                <td key={`${feature.feature_label}-${headIdx}`} className="border-b border-gray-100 px-2 py-2 text-center">
                                  {score === null ? (
                                    <div className="flex h-11 w-11 items-center justify-center rounded-md border border-dashed border-slate-200 bg-slate-50 text-[11px] text-slate-400">
                                      n/a
                                    </div>
                                  ) : (() => {
                                    const preview: HeatmapCellPreview = {
                                      head: headIdx,
                                      featureLabel: feature.feature_label,
                                      featureName: feature.feature_name,
                                      score,
                                    };

                                    return (
                                      <button
                                        type="button"
                                        data-testid={`q3-heatmap-cell-${feature.feature_label}-${headIdx}`}
                                        onClick={() => openFeatureDrilldown(headIdx, feature.feature_label, feature.feature_name, score)}
                                        onMouseEnter={() => setHoveredCell(preview)}
                                        onMouseLeave={() => setHoveredCell((current) => (
                                          current?.head === headIdx && current.featureLabel === feature.feature_label ? null : current
                                        ))}
                                        onFocus={() => setHoveredCell(preview)}
                                        onBlur={() => setHoveredCell((current) => (
                                          current?.head === headIdx && current.featureLabel === feature.feature_label ? null : current
                                        ))}
                                        title={formatHoverReadout(metric, preview)}
                                        aria-label={formatHoverReadout(metric, preview)}
                                        className={`relative flex h-11 w-11 items-center justify-center rounded-md border transition-all focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-1 ${
                                          isSelectedCell ? 'ring-2 ring-slate-900 ring-offset-1' : 'hover:-translate-y-0.5 hover:shadow-sm'
                                        }`}
                                        style={getHeatmapCellStyle(score, heatmapRange, heatmapDirection)}
                                      >
                                        <span className={`h-2.5 w-2.5 rounded-full border ${getHeatmapDotClass(score, heatmapRange, heatmapDirection)}`} />
                                        <span className="sr-only">{formatHoverReadout(metric, preview)}</span>
                                      </button>
                                    );
                                  })()}
                                </td>
                              );
                            })}
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>

                {hasMore && (
                  <div className="border-t border-gray-100 px-4 py-3 text-center">
                    <button
                      type="button"
                      onClick={() => setShowCount((current) => current + ITEMS_PER_PAGE)}
                      className="text-sm font-medium text-primary-600 hover:text-primary-700"
                    >
                      Show more ({filteredFeatures.length - showCount} remaining)
                    </button>
                  </div>
                )}
              </div>
            </div>

            <Q3ExemplarPicker
              open={activeDrilldownRequest !== null}
              request={activeDrilldownRequest}
              data={exemplarQuery.data}
              isLoading={exemplarQuery.isLoading}
              error={exemplarError}
              onClose={() => setDrilldownRequest(null)}
              onSelectCandidate={handleSelectCandidate}
            />
          </>
        )}
      </CardContent>
    </Card>
  );
}
