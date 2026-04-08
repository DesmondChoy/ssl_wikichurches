import { useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';

import { useModels } from '../../hooks/useAttention';
import { ANALYSIS_METRIC_OPTIONS } from '../../constants/metricMetadata';
import { PERCENTILE_OPTIONS } from '../../constants/percentiles';
import { buildAdvancedQ3WorkspaceHref } from '../../constants/q3Routing';
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
import { Q3DeltaPanel } from './Q3DeltaPanel';
import { Q3ScopeChip, Q3StudyScopeCallout } from './Q3ScopeFraming';
import { Q3SingleVariantExplorer, type Q3ExplorerFocus } from './Q3SingleVariantExplorer';
import type { AnalysisMetric, CompareVariantId } from '../../types';

export function Q3HeadAnalysis() {
  const navigate = useNavigate();
  const { data: modelsData } = useModels();
  const [model, setModel] = useState(Q3_DEFAULTS.model);
  const [variant, setVariant] = useState<CompareVariantId>(Q3_DEFAULTS.variant);
  const [layer, setLayer] = useState(Q3_DEFAULTS.layer);
  const [metric, setMetric] = useState<AnalysisMetric>(Q3_DEFAULTS.metric);
  const [percentile, setPercentile] = useState(Q3_DEFAULTS.percentile);
  const [activeFocus, setActiveFocus] = useState<Q3ExplorerFocus>({ head: null, featureLabel: null });

  const availableQ3Models = useMemo(() => {
    const visibleModels = (modelsData?.models ?? []).filter((value) =>
      Q3_PRIMARY_MODELS.includes(value as (typeof Q3_PRIMARY_MODELS)[number]),
    );
    return visibleModels.length > 0 ? visibleModels : [...Q3_PRIMARY_MODELS];
  }, [modelsData?.models]);

  const resolvedModel = availableQ3Models.includes(model) ? model : availableQ3Models[0];
  const maxLayer = modelsData?.num_layers_per_model?.[resolvedModel]
    ? modelsData.num_layers_per_model[resolvedModel] - 1
    : Q3_DEFAULTS.layer;
  const resolvedLayer = Math.min(layer, maxLayer);
  const variantScopeStatus = getQ3VariantScopeStatus(variant);
  const selectionHelperText = getQ3SelectionHelperText('primary', variantScopeStatus);

  const modelOptions = availableQ3Models.map((value) => ({
    value,
    label: value,
  }));
  const variantOptions = Q3_VARIANT_OPTIONS.map((option) => ({
    value: option.value,
    label: formatQ3ScopeOptionLabel(option.label, getQ3VariantScopeStatus(option.value)),
  }));

  const handleOpenAdvancedWorkspace = () => {
    navigate(buildAdvancedQ3WorkspaceHref({
      primaryModel: resolvedModel,
      secondaryModel: availableQ3Models.find((value) => value !== resolvedModel) ?? resolvedModel,
      variant,
      layer: resolvedLayer,
      metric,
      percentile,
      head: activeFocus.head,
      featureLabel: activeFocus.featureLabel,
    }));
  };

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
            Advanced comparison lives on <code>/q3</code>; Dashboard stays the main discovery surface.
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-6">
        <Q3StudyScopeCallout
          context="dashboard"
          dataTestId="q3-scope-callout"
          action={{
            label: 'Open advanced Q3 workspace',
            onClick: handleOpenAdvancedWorkspace,
            dataTestId: 'dashboard-q3-open-advanced-workspace',
          }}
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

        <Q3DeltaPanel
          model={resolvedModel}
          layer={resolvedLayer}
          metric={metric}
          percentile={percentile}
        />

        <Q3SingleVariantExplorer
          model={resolvedModel}
          variant={variant}
          layer={resolvedLayer}
          metric={metric}
          percentile={percentile}
          onActiveFocusChange={setActiveFocus}
        />
      </CardContent>
    </Card>
  );
}
