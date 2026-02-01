/**
 * Control panel for selecting model, layer, and display options.
 */

import { useViewStore } from '../../store/viewStore';
import { useModels } from '../../hooks/useAttention';
import { Select } from '../ui/Select';
import { Slider } from '../ui/Slider';
import { Toggle } from '../ui/Toggle';

interface ControlPanelProps {
  className?: string;
}

export function ControlPanel({ className = '' }: ControlPanelProps) {
  const {
    model,
    layer,
    percentile,
    showBboxes,
    heatmapOpacity,
    setModel,
    setLayer,
    setPercentile,
    setShowBboxes,
    setHeatmapOpacity,
  } = useViewStore();

  const { data: modelsData, isLoading } = useModels();

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
