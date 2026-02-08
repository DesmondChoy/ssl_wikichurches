/**
 * Dashboard page with overall metrics and leaderboard.
 */

import { useEffect } from 'react';
import { Link } from 'react-router-dom';
// TODO: recharts has SVG animation bugs with current React version
// Re-enable when recharts releases a fix
// import {
//   LineChart, Line, XAxis, YAxis, CartesianGrid,
//   Tooltip, Legend, ResponsiveContainer, BarChart, Bar,
// } from 'recharts';
import { useViewStore } from '../store/viewStore';
import { useAllModelsSummary, useStyleBreakdown } from '../hooks/useMetrics';
import { useModels } from '../hooks/useAttention';
import { ModelLeaderboard } from '../components/metrics/ModelLeaderboard';
import { FeatureBreakdown } from '../components/metrics/FeatureBreakdown';
import { Card, CardHeader, CardContent } from '../components/ui/Card';
import { Select } from '../components/ui/Select';
import { PERCENTILE_OPTIONS } from '../constants/percentiles';

export function DashboardPage() {
  const { model, layer, method, percentile, setModel, setPercentile, setMethodsConfig, setNumLayersPerModel } = useViewStore();
  const { data: modelsData } = useModels();

  // Populate store with model config (methods, layer counts) so setModel
  // resolves the correct default method (e.g. gradcam for ResNet-50)
  useEffect(() => {
    if (modelsData?.methods && modelsData?.default_methods) {
      setMethodsConfig(modelsData.methods, modelsData.default_methods);
    }
  }, [modelsData, setMethodsConfig]);

  useEffect(() => {
    if (modelsData?.num_layers_per_model) {
      setNumLayersPerModel(modelsData.num_layers_per_model);
    }
  }, [modelsData, setNumLayersPerModel]);

  const { data: summary, isLoading: summaryLoading } = useAllModelsSummary(percentile);
  const { data: styleBreakdown, isLoading: styleLoading } = useStyleBreakdown(model, layer, percentile, method);

  // Collect all unique layers from API response (handles models with different layer counts)
  const allLayers = new Set<string>();
  if (summary?.models) {
    for (const modelData of Object.values(summary.models)) {
      for (const layerKey of Object.keys(modelData.layer_progression)) {
        allLayers.add(layerKey);
      }
    }
  }
  // Sort layers numerically (layer0, layer1, ..., layer11)
  const sortedLayers = Array.from(allLayers).sort((a, b) =>
    parseInt(a.replace('layer', '')) - parseInt(b.replace('layer', ''))
  );

  // Merge into single array by layer
  const chartData: Record<string, number | string>[] = sortedLayers.map((layerKey) => {
    const layerNum = parseInt(layerKey.replace('layer', ''));
    const layerData: Record<string, number | string> = { layer: `L${layerNum}` };
    if (summary?.models) {
      for (const [modelName, modelData] of Object.entries(summary.models)) {
        if (modelData.layer_progression[layerKey] !== undefined) {
          layerData[modelName] = modelData.layer_progression[layerKey];
        }
      }
    }
    return layerData;
  });

  // Style breakdown data
  const styleData = styleBreakdown
    ? Object.entries(styleBreakdown.styles).map(([style, iou]) => ({
        style,
        iou,
        count: styleBreakdown.style_counts[style] || 0,
      }))
    : [];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
          <p className="text-gray-600 mt-1">
            Attention-annotation alignment metrics across models
          </p>
        </div>

        <Select
          value={percentile}
          onChange={(v) => setPercentile(Number(v))}
          options={PERCENTILE_OPTIONS}
          label="Threshold"
        />
      </div>

      {/* Main grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Leaderboard */}
        <div>
          <ModelLeaderboard
            percentile={percentile}
            onModelSelect={setModel}
          />
        </div>

        {/* Layer progression chart */}
        <div className="lg:col-span-2">
          <Card>
            <CardHeader>
              <h3 className="font-semibold">Layer Progression (All Models)</h3>
            </CardHeader>
            <CardContent>
              {summaryLoading ? (
                <div className="h-64 animate-pulse bg-gray-100 rounded" />
              ) : (
                <div className="h-[300px] flex items-center justify-center bg-gray-50 rounded border-2 border-dashed border-gray-200">
                  <div className="text-center text-gray-500">
                    <p className="font-medium">Layer Progression Chart</p>
                    <p className="text-sm">{chartData.length} layers × {summary?.leaderboard.length || 0} models</p>
                    <p className="text-xs mt-2 text-gray-400">Charts temporarily disabled</p>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Style breakdown and Feature breakdown */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <div className="flex justify-between items-center">
              <h3 className="font-semibold">IoU by Architectural Style</h3>
              <span className="text-xs text-gray-500 capitalize">{model}</span>
            </div>
          </CardHeader>
          <CardContent>
            {styleLoading ? (
              <div className="h-48 animate-pulse bg-gray-100 rounded" />
            ) : styleData.length > 0 ? (
              <div className="h-[200px] flex items-center justify-center bg-gray-50 rounded border-2 border-dashed border-gray-200">
                <div className="text-center text-gray-500">
                  <p className="font-medium">Style Breakdown Chart</p>
                  <p className="text-sm">{styleData.length} architectural styles</p>
                  <p className="text-xs mt-2 text-gray-400">Charts temporarily disabled</p>
                </div>
              </div>
            ) : (
              <div className="h-48 flex items-center justify-center text-gray-500">
                No style data available
              </div>
            )}
          </CardContent>
        </Card>

        {/* Feature breakdown */}
        <FeatureBreakdown model={model} layer={layer} percentile={percentile} method={method} />
      </div>

      {/* Quick links */}
      <Card>
        <CardHeader>
          <h3 className="font-semibold">Quick Actions</h3>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            <Link
              to="/"
              className="block p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
            >
              <div className="font-medium">Browse Images</div>
              <div className="text-sm text-gray-500">
                View all 139 annotated church images
              </div>
            </Link>

            <Link
              to="/compare"
              className="block p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
            >
              <div className="font-medium">Compare Models</div>
              <div className="text-sm text-gray-500">
                Side-by-side attention comparison
              </div>
            </Link>

            <div className="p-3 bg-yellow-50 rounded-lg">
              <div className="font-medium text-yellow-800">Pre-computation Required</div>
              <div className="text-sm text-yellow-700">
                Run the pre-computation scripts to generate heatmaps and metrics
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
