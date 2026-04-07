import { useMemo, useState } from 'react';

import { useModels } from '../../hooks/useAttention';
import { useHeadFeatureMatrix, useHeadRanking } from '../../hooks/useMetrics';
import { ANALYSIS_METRIC_METADATA, ANALYSIS_METRIC_OPTIONS, COMPARE_VARIANT_OPTIONS, formatMetricValue } from '../../constants/metricMetadata';
import { PERCENTILE_OPTIONS } from '../../constants/percentiles';
import {
  Q3_DEFAULTS,
  formatQ3ScopeOptionLabel,
  getQ3ModelScopeStatus,
  getQ3SelectionHelperText,
  getQ3VariantScopeStatus,
} from '../../constants/q3Scope';
import { Card, CardContent, CardHeader } from '../ui/Card';
import { Select } from '../ui/Select';
import { Q3ScopeChip, Q3StudyScopeCallout } from './Q3ScopeFraming';
import type { AnalysisMetric, CompareVariantId } from '../../types';

const ITEMS_PER_PAGE = 20;

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

export function Q3HeadAnalysis() {
  const { data: modelsData } = useModels();
  const [model, setModel] = useState(Q3_DEFAULTS.model);
  const [variant, setVariant] = useState<CompareVariantId>(Q3_DEFAULTS.variant);
  const [layer, setLayer] = useState(Q3_DEFAULTS.layer);
  const [metric, setMetric] = useState<AnalysisMetric>(Q3_DEFAULTS.metric);
  const [percentile, setPercentile] = useState(Q3_DEFAULTS.percentile);
  const [searchQuery, setSearchQuery] = useState('');
  const [showCount, setShowCount] = useState(ITEMS_PER_PAGE);

  const resolvedModel = useMemo(() => {
    if (!modelsData?.models.length) {
      return model;
    }
    return modelsData.models.includes(model) ? model : modelsData.models[0];
  }, [model, modelsData]);

  const metricMetadata = ANALYSIS_METRIC_METADATA[metric];
  const effectivePercentile = metricMetadata.thresholdFree ? 90 : percentile;
  const maxLayer = modelsData?.num_layers_per_model?.[resolvedModel]
    ? modelsData.num_layers_per_model[resolvedModel] - 1
    : 11;
  const resolvedLayer = Math.min(layer, maxLayer);
  const modelScopeStatus = getQ3ModelScopeStatus(resolvedModel);
  const variantScopeStatus = getQ3VariantScopeStatus(variant);
  const selectionHelperText = getQ3SelectionHelperText(modelScopeStatus, variantScopeStatus);

  const rankingQuery = useHeadRanking(resolvedModel, resolvedLayer, effectivePercentile, metric, variant);
  const matrixQuery = useHeadFeatureMatrix(resolvedModel, resolvedLayer, effectivePercentile, metric, variant);

  const filteredFeatures = useMemo(() => {
    const features = matrixQuery.data?.features ?? [];
    if (!searchQuery.trim()) {
      return features;
    }
    const query = searchQuery.toLowerCase();
    return features.filter((feature) => feature.feature_name.toLowerCase().includes(query));
  }, [matrixQuery.data?.features, searchQuery]);

  const visibleFeatures = filteredFeatures.slice(0, showCount);
  const hasMore = showCount < filteredFeatures.length;
  const modelOptions = (modelsData?.models ?? []).map((value) => ({
    value,
    label: formatQ3ScopeOptionLabel(value, getQ3ModelScopeStatus(value)),
  }));
  const variantOptions = COMPARE_VARIANT_OPTIONS.map((option) => ({
    ...option,
    label: formatQ3ScopeOptionLabel(option.label, getQ3VariantScopeStatus(option.value)),
  }));
  const unsupportedHelperText = modelScopeStatus === 'outside'
    ? 'Outside-primary-scope selections remain explorable, but they are not part of the headline Q3 claim.'
    : selectionHelperText;
  const emptyStateHelperText = modelScopeStatus === 'outside'
    ? 'This out-of-scope selection stays available for exploration even when per-head rows are not yet populated.'
    : selectionHelperText;

  return (
    <Card>
      <CardHeader>
        <div className="flex flex-col gap-2 lg:flex-row lg:items-start lg:justify-between">
          <div>
            <h3 className="font-semibold">Q3 Per-Head Specialization</h3>
            <p className="mt-1 text-sm text-gray-600">
              Compare which individual heads align best with expert annotations for each metric.
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
            options={Array.from({ length: maxLayer + 1 }, (_, idx) => ({
              value: idx,
              label: `Layer ${idx}`,
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
            Current Q3 framing
          </div>
          <div className="mt-2 flex flex-wrap gap-2">
            <span className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-sm text-slate-700">
              <span className="font-medium text-slate-900">Model</span>
              <span>{resolvedModel}</span>
              <Q3ScopeChip status={modelScopeStatus} dataTestId="q3-model-scope-chip" />
            </span>
            <span className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-sm text-slate-700">
              <span className="font-medium text-slate-900">Variant</span>
              <span>{COMPARE_VARIANT_OPTIONS.find((option) => option.value === variant)?.label ?? variant}</span>
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
            {rankingQuery.data?.reason || matrixQuery.data?.reason || 'Per-head analysis is not available for this selection.'}{' '}
            {unsupportedHelperText}
          </div>
        ) : (rankingQuery.data?.heads.length ?? 0) === 0 && (matrixQuery.data?.features.length ?? 0) === 0 ? (
          <div className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-700">
            No Q3 per-head rows are available for this selection yet. Run the per-head precompute commands first.{' '}
            {emptyStateHelperText}
          </div>
        ) : (
          <>
            <div className="grid grid-cols-1 gap-6 xl:grid-cols-[20rem_minmax(0,1fr)]">
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
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-100">
                      {(rankingQuery.data?.heads ?? []).map((entry) => (
                        <tr key={entry.head}>
                          <td className="px-4 py-2 font-medium text-gray-900">Head {entry.head}</td>
                          <td className="px-4 py-2 text-right">
                            <span className={`inline-block rounded px-2 py-0.5 text-xs font-medium ${getMetricTone(metric, entry.mean_score)}`}>
                              {formatMetricValue(metric, entry.mean_score)}
                            </span>
                          </td>
                          <td className="px-4 py-2 text-right text-gray-600">{entry.mean_rank.toFixed(2)}</td>
                          <td className="px-4 py-2 text-right text-gray-600">{entry.top1_count}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              <div className="rounded-lg border border-gray-200">
                <div className="border-b border-gray-100 px-4 py-3">
                  <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
                    <div>
                      <div className="font-medium text-gray-900">Head × Feature Matrix</div>
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
                </div>

                <div className="overflow-x-auto">
                  <table className="min-w-full text-sm">
                    <thead className="border-b border-gray-200 text-left text-xs text-gray-500">
                      <tr>
                        <th className="sticky left-0 bg-white px-4 py-2 font-medium">Feature</th>
                        {(matrixQuery.data?.heads ?? []).map((headIdx) => (
                          <th key={headIdx} className="px-3 py-2 text-right font-medium">
                            H{headIdx}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-100">
                      {visibleFeatures.map((feature) => (
                        <tr key={feature.feature_label}>
                          <td className="sticky left-0 bg-white px-4 py-2">
                            <div className="font-medium text-gray-900">{feature.feature_name}</div>
                            <div className="text-xs text-gray-400">ID {feature.feature_label} • {feature.bbox_count} boxes</div>
                          </td>
                          {feature.scores.map((score, index) => (
                            <td key={`${feature.feature_label}-${index}`} className="px-3 py-2 text-right">
                              <span className={`inline-block rounded px-2 py-0.5 text-xs font-medium ${getMetricTone(metric, score)}`}>
                                {formatMetricValue(metric, score)}
                              </span>
                            </td>
                          ))}
                        </tr>
                      ))}
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
          </>
        )}
      </CardContent>
    </Card>
  );
}
