/**
 * Control panel for selecting model, layer, attention method, and display options.
 */

import { useEffect } from 'react';
import { useViewStore } from '../../store/viewStore';
import { useModels } from '../../hooks/useAttention';
import { Select } from '../ui/Select';
import { Slider } from '../ui/Slider';
import { Toggle } from '../ui/Toggle';

// Human-readable labels for attention methods
const METHOD_LABELS: Record<string, string> = {
  cls: 'CLS Attention',
  rollout: 'Attention Rollout',
  gradcam: 'Grad-CAM',
  mean: 'Mean Attention',
};

interface ControlPanelProps {
  className?: string;
}

export function ControlPanel({ className = '' }: ControlPanelProps) {
  const {
    model,
    layer,
    method,
    percentile,
    showBboxes,
    heatmapOpacity,
    availableMethods,
    setModel,
    setLayer,
    setMethod,
    setPercentile,
    setShowBboxes,
    setHeatmapOpacity,
    setMethodsConfig,
  } = useViewStore();

  const { data: modelsData, isLoading } = useModels();

  // Update store with methods config when API data loads
  useEffect(() => {
    if (modelsData?.methods && modelsData?.default_methods) {
      setMethodsConfig(modelsData.methods, modelsData.default_methods);
    }
  }, [modelsData, setMethodsConfig]);

  if (isLoading) {
    return (
      <div className={`p-4 bg-white rounded-lg shadow ${className}`}>
        <div className="animate-pulse space-y-4">
          <div className="h-8 bg-gray-200 rounded" />
          <div className="h-8 bg-gray-200 rounded" />
          <div className="h-8 bg-gray-200 rounded" />
        </div>
      </div>
    );
  }

  const modelOptions =
    modelsData?.models.map((m) => ({
      value: m,
      label: m.charAt(0).toUpperCase() + m.slice(1),
    })) || [];

  // Get available methods for current model
  const currentModelMethods = availableMethods[model] || modelsData?.methods?.[model] || ['cls'];
  const methodOptions = currentModelMethods.map((m) => ({
    value: m,
    label: METHOD_LABELS[m] || m,
  }));

  const percentileOptions = [
    { value: 90, label: 'Top 10%' },
    { value: 85, label: 'Top 15%' },
    { value: 80, label: 'Top 20%' },
    { value: 70, label: 'Top 30%' },
    { value: 60, label: 'Top 40%' },
    { value: 50, label: 'Top 50%' },
  ];

  return (
    <div className={`p-4 bg-white rounded-lg shadow space-y-4 ${className}`}>
      <h3 className="font-semibold text-gray-900">View Settings</h3>

      <Select
        value={model}
        onChange={setModel}
        options={modelOptions}
        label="Model"
      />

      {/* Only show method selector if model has multiple methods */}
      {currentModelMethods.length > 1 && (
        <Select
          value={method}
          onChange={setMethod}
          options={methodOptions}
          label="Attention Method"
        />
      )}

      <Slider
        value={layer}
        onChange={setLayer}
        min={0}
        max={(modelsData?.num_layers || 12) - 1}
        label={`Layer ${layer}`}
      />

      <Select
        value={percentile}
        onChange={(v) => setPercentile(Number(v))}
        options={percentileOptions}
        label="Attention Threshold"
      />

      <Toggle
        checked={showBboxes}
        onChange={setShowBboxes}
        label="Show Bounding Boxes"
      />

      <div className="border-t pt-4 mt-2">
        <h4 className="text-sm font-medium text-gray-700 mb-3">Similarity Heatmap</h4>

        <Slider
          value={heatmapOpacity}
          onChange={setHeatmapOpacity}
          min={0.2}
          max={0.9}
          step={0.1}
          label={`Opacity ${Math.round(heatmapOpacity * 100)}%`}
          showValue={false}
        />
      </div>
    </div>
  );
}
