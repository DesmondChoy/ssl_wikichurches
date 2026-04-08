import { useEffect, useMemo } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';

import { useModels } from '../hooks/useAttention';
import { Card, CardContent, CardHeader } from '../components/ui/Card';
import { Select } from '../components/ui/Select';
import { Q3DeltaPanel } from '../components/metrics/Q3DeltaPanel';
import { Q3ScopeChip, Q3StudyScopeCallout } from '../components/metrics/Q3ScopeFraming';
import { Q3SingleVariantExplorer, type Q3ExplorerFocus } from '../components/metrics/Q3SingleVariantExplorer';
import { ANALYSIS_METRIC_OPTIONS } from '../constants/metricMetadata';
import { PERCENTILE_OPTIONS } from '../constants/percentiles';
import {
  createAdvancedQ3WorkspaceSearchParams,
  parseAdvancedQ3WorkspaceState,
} from '../constants/q3Routing';
import {
  Q3_PRIMARY_MODELS,
  Q3_VARIANT_OPTIONS,
  formatQ3ScopeOptionLabel,
  getQ3SelectionHelperText,
  getQ3VariantScopeStatus,
} from '../constants/q3Scope';
import type { AnalysisMetric, CompareVariantId } from '../types';

export function Q3Page() {
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const { data: modelsData } = useModels();

  const availableQ3Models = useMemo(() => {
    const visibleModels = (modelsData?.models ?? []).filter((value) =>
      Q3_PRIMARY_MODELS.includes(value as (typeof Q3_PRIMARY_MODELS)[number]),
    );
    return visibleModels.length > 0 ? visibleModels : [...Q3_PRIMARY_MODELS];
  }, [modelsData?.models]);

  const provisionalState = parseAdvancedQ3WorkspaceState(searchParams, {
    availableModels: availableQ3Models,
  });

  const primaryMaxLayer = modelsData?.num_layers_per_model?.[provisionalState.primaryModel]
    ? modelsData.num_layers_per_model[provisionalState.primaryModel] - 1
    : 11;
  const secondaryMaxLayer = modelsData?.num_layers_per_model?.[provisionalState.secondaryModel]
    ? modelsData.num_layers_per_model[provisionalState.secondaryModel] - 1
    : 11;
  const sharedMaxLayer = Math.max(0, Math.min(primaryMaxLayer, secondaryMaxLayer));

  const workspaceState = parseAdvancedQ3WorkspaceState(searchParams, {
    availableModels: availableQ3Models,
    maxLayer: sharedMaxLayer,
  });
  const searchParamsString = searchParams.toString();
  const variantScopeStatus = getQ3VariantScopeStatus(workspaceState.variant);
  const selectionHelperText = getQ3SelectionHelperText('primary', variantScopeStatus);
  const variantOptions = Q3_VARIANT_OPTIONS.map((option) => ({
    value: option.value,
    label: formatQ3ScopeOptionLabel(option.label, getQ3VariantScopeStatus(option.value)),
  }));

  useEffect(() => {
    const normalized = createAdvancedQ3WorkspaceSearchParams(workspaceState);
    if (normalized.toString() !== searchParamsString) {
      setSearchParams(normalized, { replace: true });
    }
  }, [searchParamsString, setSearchParams, workspaceState]);

  const updateWorkspaceState = (
    patch: Partial<typeof workspaceState>,
    replace = false,
  ) => {
    const nextState = {
      ...workspaceState,
      ...patch,
    };
    const normalized = createAdvancedQ3WorkspaceSearchParams(nextState);
    if (normalized.toString() === searchParamsString) {
      return;
    }
    setSearchParams(normalized, { replace });
  };

  const handlePrimaryModelChange = (value: string) => {
    updateWorkspaceState({
      primaryModel: value,
      secondaryModel: value === workspaceState.secondaryModel
        ? workspaceState.primaryModel
        : workspaceState.secondaryModel,
    });
  };

  const handleSecondaryModelChange = (value: string) => {
    updateWorkspaceState({
      secondaryModel: value,
      primaryModel: value === workspaceState.primaryModel
        ? workspaceState.secondaryModel
        : workspaceState.primaryModel,
    });
  };

  const handleSwapModels = () => {
    updateWorkspaceState({
      primaryModel: workspaceState.secondaryModel,
      secondaryModel: workspaceState.primaryModel,
    });
  };

  const handleSharedFocusChange = (focus: Q3ExplorerFocus) => {
    updateWorkspaceState({
      head: focus.head,
      featureLabel: focus.featureLabel,
    });
  };

  return (
    <div className="space-y-6" data-testid="advanced-q3-page">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <h1 className="text-3xl font-bold text-slate-900">Q3 Advanced Workspace</h1>
          <p className="mt-2 max-w-3xl text-sm text-slate-600">
            Compare two primary-study models side by side without changing the Dashboard-first Q3 workflow.
            Shared controls keep both panes in the same layer, metric, and variant context so H2 differences are easier to inspect.
          </p>
        </div>
      </div>

      <Q3StudyScopeCallout
        context="workspace"
        dataTestId="advanced-q3-scope-callout"
        action={{
          label: 'Return to Dashboard Q3',
          onClick: () => navigate('/dashboard?tab=q3'),
          dataTestId: 'advanced-q3-return-dashboard',
        }}
      />

      <Card>
        <CardHeader>
          <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
            <div>
              <h2 className="text-lg font-semibold text-slate-900">Shared comparison controls</h2>
              <p className="mt-1 text-sm text-slate-600">
                Keep the experiment context aligned while you compare one model pair at a time.
              </p>
            </div>
            <button
              type="button"
              onClick={handleSwapModels}
              className="rounded-md border border-slate-200 bg-white px-3 py-2 text-sm font-medium text-slate-700 transition-colors hover:bg-slate-50"
            >
              Swap models
            </button>
          </div>
        </CardHeader>
        <CardContent className="space-y-4" data-testid="advanced-q3-controls">
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-6">
            <Select
              value={workspaceState.primaryModel}
              onChange={handlePrimaryModelChange}
              options={availableQ3Models.map((value) => ({ value, label: value }))}
              label="Primary model"
            />
            <Select
              value={workspaceState.secondaryModel}
              onChange={handleSecondaryModelChange}
              options={availableQ3Models.map((value) => ({ value, label: value }))}
              label="Comparison model"
            />
            <Select
              value={workspaceState.variant}
              onChange={(value) => updateWorkspaceState({ variant: value as CompareVariantId })}
              options={variantOptions}
              label="Variant"
            />
            <Select
              value={workspaceState.layer}
              onChange={(value) => updateWorkspaceState({ layer: Number(value) })}
              options={Array.from({ length: sharedMaxLayer + 1 }, (_, index) => ({
                value: index,
                label: `Layer ${index}`,
              }))}
              label="Layer"
            />
            <Select
              value={workspaceState.metric}
              onChange={(value) => updateWorkspaceState({ metric: value as AnalysisMetric })}
              options={ANALYSIS_METRIC_OPTIONS}
              label="Metric"
            />
            <Select
              value={workspaceState.percentile}
              onChange={(value) => updateWorkspaceState({ percentile: Number(value) })}
              options={PERCENTILE_OPTIONS}
              label="Percentile"
            />
          </div>

          <div className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-3">
            <div className="flex flex-wrap gap-2">
              <Q3ScopeChip status="primary">Scoped Q3 models</Q3ScopeChip>
              <Q3ScopeChip status={variantScopeStatus} dataTestId="advanced-q3-variant-scope-chip" />
            </div>
            <p className="mt-2 text-sm text-slate-700" data-testid="advanced-q3-selection-helper">
              {selectionHelperText}
            </p>
            <p className="mt-2 text-xs text-slate-600" data-testid="advanced-q3-focus-helper">
              Ranking-row and heatmap-cell selections sync through the shared URL state, so you can compare the same head or head-feature context across both panes.
            </p>
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 gap-6 2xl:grid-cols-2">
        <div data-testid="advanced-q3-pane-primary">
          <Card>
            <CardHeader>
              <div className="space-y-2">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-xs font-semibold uppercase tracking-[0.12em] text-slate-600">
                    Primary pane
                  </span>
                  <Q3ScopeChip status="primary" />
                </div>
                <div>
                  <h2 className="text-lg font-semibold text-slate-900">{workspaceState.primaryModel}</h2>
                  <p className="mt-1 text-sm text-slate-600">
                    Use this pane as the anchor comparison for the current Q3 context.
                  </p>
                </div>
              </div>
            </CardHeader>
            <CardContent className="space-y-6">
              <Q3SingleVariantExplorer
                model={workspaceState.primaryModel}
                variant={workspaceState.variant}
                layer={workspaceState.layer}
                metric={workspaceState.metric}
                percentile={workspaceState.percentile}
                focus={{
                  head: workspaceState.head,
                  featureLabel: workspaceState.featureLabel,
                }}
                onActiveFocusChange={handleSharedFocusChange}
                autoScrollSelectionIntoView={false}
              />
            </CardContent>
          </Card>
        </div>

        <div data-testid="advanced-q3-pane-secondary">
          <Card>
            <CardHeader>
              <div className="space-y-2">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-xs font-semibold uppercase tracking-[0.12em] text-slate-600">
                    Comparison pane
                  </span>
                  <Q3ScopeChip status="primary" />
                </div>
                <div>
                  <h2 className="text-lg font-semibold text-slate-900">{workspaceState.secondaryModel}</h2>
                  <p className="mt-1 text-sm text-slate-600">
                    Compare the same Q3 setting against a second model without manually re-entering filters.
                  </p>
                </div>
              </div>
            </CardHeader>
            <CardContent className="space-y-6">
              <Q3SingleVariantExplorer
                model={workspaceState.secondaryModel}
                variant={workspaceState.variant}
                layer={workspaceState.layer}
                metric={workspaceState.metric}
                percentile={workspaceState.percentile}
                focus={{
                  head: workspaceState.head,
                  featureLabel: workspaceState.featureLabel,
                }}
                onActiveFocusChange={handleSharedFocusChange}
                autoScrollSelectionIntoView={false}
              />
            </CardContent>
          </Card>
        </div>
      </div>

      <Q3DeltaPanel
        model={workspaceState.primaryModel}
        layer={workspaceState.layer}
        metric={workspaceState.metric}
        percentile={workspaceState.percentile}
        title={`Adaptation shifts for ${workspaceState.primaryModel}`}
        description={`Keep the pairwise comparison above as the main workspace, and use this supporting panel to see how ${workspaceState.primaryModel} shifts from Frozen into the adapted variants.`}
        helperText="This supporting panel reuses the existing Dashboard Q3 delta view for the primary model only."
        testIdPrefix="advanced-q3-primary"
      />
    </div>
  );
}
