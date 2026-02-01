/**
 * Control panel for selecting model, layer, attention method, and display options.
 */

import { useEffect } from 'react';
import { useViewStore } from '../../store/viewStore';
import { useModels } from '../../hooks/useAttention';
import { Select } from '../ui/Select';
import { Slider } from '../ui/Slider';
import { Toggle } from '../ui/Toggle';
import { GLOSSARY } from '../../constants/glossary';
import type { HeatmapStyle } from '../../types';

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
    heatmapStyle,
    availableMethods,
    setModel,
    setLayer,
    setMethod,
    setPercentile,
    setShowBboxes,
    setHeatmapOpacity,
    setHeatmapStyle,
    setMethodsConfig,
    setNumLayersPerModel,
  } = useViewStore();

  const { data: modelsData, isLoading } = useModels();

  // Update store with methods config when API data loads
  useEffect(() => {
    if (modelsData?.methods && modelsData?.default_methods) {
      setMethodsConfig(modelsData.methods, modelsData.default_methods);
    }
  }, [modelsData, setMethodsConfig]);

  // Update store with num_layers_per_model when API data loads
  useEffect(() => {
    if (modelsData?.num_layers_per_model) {
      setNumLayersPerModel(modelsData.num_layers_per_model);
    }
  }, [modelsData, setNumLayersPerModel]);

  // Get max layer for current model (0-indexed, so subtract 1)
  const maxLayer = modelsData?.num_layers_per_model?.[model]
    ? modelsData.num_layers_per_model[model] - 1
    : (modelsData?.num_layers || 12) - 1;

  // Note: Layer clamping now happens synchronously in viewStore.setModel()
  // This prevents race conditions where API calls fire before the useEffect runs

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

  const heatmapStyleOptions = [
    { value: 'smooth', label: 'Smooth Gradient' },
    { value: 'squares', label: 'Squares' },
    { value: 'circles', label: 'Circles' },
  ];

  return (
    <div className={`p-4 bg-white rounded-lg shadow space-y-4 ${className}`}>
      <h3 className="font-semibold text-gray-900">View Settings</h3>

      <Select
        value={model}
        onChange={setModel}
        options={modelOptions}
        label="Model"
        tooltip={GLOSSARY['Model']}
      />

      {/* Only show method selector if model has multiple methods */}
      {currentModelMethods.length > 1 && (
        <Select
          value={method}
          onChange={setMethod}
          options={methodOptions}
          label="Attention Method"
          tooltip={GLOSSARY['Attention Method']}
        />
      )}

      <Slider
        value={layer}
        onChange={setLayer}
        min={0}
        max={maxLayer}
        label={`Layer ${layer}`}
        tooltip={GLOSSARY['Layer']}
      />

      <Select
        value={percentile}
        onChange={(v) => setPercentile(Number(v))}
        options={percentileOptions}
        label="Attention Threshold"
        tooltip={GLOSSARY['Attention Threshold']}
      />

      <Toggle
        checked={showBboxes}
        onChange={setShowBboxes}
        label="Show Bounding Boxes"
      />

      <div className="border-t pt-4 mt-2 space-y-3">
        <h4 className="text-sm font-medium text-gray-700">Similarity Heatmap</h4>

        <Select
          value={heatmapStyle}
          onChange={(v) => setHeatmapStyle(v as HeatmapStyle)}
          options={heatmapStyleOptions}
          label="Heatmap Style"
          tooltip={GLOSSARY['Heatmap Style']}
        />

        <Slider
          value={heatmapOpacity}
          onChange={setHeatmapOpacity}
          min={0.2}
          max={0.9}
          step={0.1}
          label={`Opacity ${Math.round(heatmapOpacity * 100)}%`}
          tooltip={GLOSSARY['Heatmap Opacity']}
          showValue={false}
        />
      </div>
    </div>
  );
}
