import { COMPARE_VARIANT_OPTIONS } from '../../constants/metricMetadata';
import {
  Q3_PRIMARY_MODELS,
  Q3_VARIANT_OPTIONS,
  formatQ3ScopeOptionLabel,
  getQ3ScopeChipClassName,
  getQ3ScopeLabel,
  getQ3VariantScopeStatus,
} from '../../constants/q3Scope';
import { Card, CardContent, CardHeader } from '../ui/Card';
import { Select } from '../ui/Select';
import { Slider } from '../ui/Slider';
import { Toggle } from '../ui/Toggle';
import type { CompareVariantId } from '../../types';

interface Q3ImageDetailControlsProps {
  model: string;
  variant: CompareVariantId;
  layer: number;
  head: number | null;
  maxLayer: number;
  numHeads: number;
  showBboxes: boolean;
  featureName: string | null;
  onModelChange: (model: string) => void;
  onVariantChange: (variant: CompareVariantId) => void;
  onLayerChange: (layer: number) => void;
  onHeadChange: (head: number | null) => void;
  onShowBboxesChange: (show: boolean) => void;
}

export function Q3ImageDetailControls({
  model,
  variant,
  layer,
  head,
  maxLayer,
  numHeads,
  showBboxes,
  featureName,
  onModelChange,
  onVariantChange,
  onLayerChange,
  onHeadChange,
  onShowBboxesChange,
}: Q3ImageDetailControlsProps) {
  const variantStatus = getQ3VariantScopeStatus(variant);
  const controlVariantLabel = COMPARE_VARIANT_OPTIONS.find((option) => option.value === variant)?.label ?? variant;

  return (
    <Card>
      <CardHeader>
        <div className="space-y-1">
          <h3 className="font-semibold text-gray-900">Q3 Controls</h3>
          <p className="text-sm text-gray-600">
            Stay inside the selected Q3 workflow context while you inspect one exemplar image.
          </p>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-3">
          <div className="text-xs font-semibold uppercase tracking-[0.12em] text-slate-500">
            Current drill-down
          </div>
          <div className="mt-2 flex flex-wrap gap-2 text-sm text-slate-700">
            <span className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-3 py-1">
              <span className="font-medium text-slate-900">Model</span>
              <span>{model}</span>
            </span>
            <span className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-3 py-1">
              <span className="font-medium text-slate-900">Variant</span>
              <span>{controlVariantLabel}</span>
              <span className={`inline-flex items-center rounded-full border px-2 py-0.5 text-xs font-semibold ${getQ3ScopeChipClassName(variantStatus)}`}>
                {getQ3ScopeLabel(variantStatus)}
              </span>
            </span>
          </div>
          {featureName && (
            <p className="mt-2 text-xs text-slate-600">
              Feature context from Dashboard Q3: {featureName}
            </p>
          )}
        </div>

        <Select
          value={model}
          onChange={onModelChange}
          options={Q3_PRIMARY_MODELS.map((value) => ({
            value,
            label: value,
          }))}
          label="Model"
        />

        <Select
          value={variant}
          onChange={(value) => onVariantChange(value as CompareVariantId)}
          options={Q3_VARIANT_OPTIONS.map((option) => ({
            value: option.value,
            label: formatQ3ScopeOptionLabel(option.label, getQ3VariantScopeStatus(option.value)),
          }))}
          label="Variant"
        />

        <Select
          value={head === null ? 'all' : String(head)}
          onChange={(value) => onHeadChange(value === 'all' ? null : Number(value))}
          options={[
            { value: 'all', label: 'All (Fused)' },
            ...Array.from({ length: numHeads }, (_, index) => ({
              value: String(index),
              label: `Head ${index}`,
            })),
          ]}
          label="Attention Head"
        />

        <Slider
          value={layer}
          onChange={onLayerChange}
          min={0}
          max={maxLayer}
          label={`Layer ${layer}`}
          showValue={false}
        />

        <Toggle
          checked={showBboxes}
          onChange={onShowBboxesChange}
          label="Show Bounding Boxes"
        />

        <div className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-600">
          Head context stays loaded even in Feature Similarity mode so you can switch back to attention inspection without losing the selected Q3 finding.
        </div>
      </CardContent>
    </Card>
  );
}
