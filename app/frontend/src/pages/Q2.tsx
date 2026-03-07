import { useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { useModels } from '../hooks/useAttention';
import { useQ2Summary } from '../hooks/useMetrics';
import { Card, CardContent, CardHeader } from '../components/ui/Card';
import { Select } from '../components/ui/Select';

const STRATEGY_OPTIONS = [
  { value: 'all', label: 'All Strategies' },
  { value: 'linear_probe', label: 'Linear Probe' },
  { value: 'lora', label: 'LoRA' },
  { value: 'full', label: 'Full Fine-tune' },
];

const PERCENTILES = [90, 80, 70, 60, 50];

export function Q2Page() {
  const [percentile, setPercentile] = useState(90);
  const [selectedModel, setSelectedModel] = useState('all');
  const [selectedStrategy, setSelectedStrategy] = useState('all');
  const [layer, setLayer] = useState('11');

  const { data: modelsData } = useModels();
  const modelOptions = useMemo(() => {
    const models = (modelsData?.models || []).filter((m) => m !== 'resnet50');
    return [{ value: 'all', label: 'All Models' }, ...models.map((m) => ({ value: m, label: m }))];
  }, [modelsData]);

  const { data, isLoading, error } = useQ2Summary(
    percentile,
    selectedModel === 'all' ? undefined : selectedModel,
    selectedStrategy === 'all' ? undefined : selectedStrategy,
  );

  const rows = useMemo(() => {
    if (!data?.models) return [];
    const p = String(percentile);
    const acc: Array<{
      model: string;
      strategy: string;
      delta: number;
      ci: string;
      significant: boolean;
      retention: string;
    }> = [];

    Object.entries(data.models).forEach(([model, strategies]) => {
      Object.entries(strategies).forEach(([strategy, percentileMap]) => {
        const point = percentileMap[p];
        if (!point) return;
        acc.push({
          model,
          strategy,
          delta: point.mean_delta_iou,
          ci: `[${point.delta_ci_lower.toFixed(3)}, ${point.delta_ci_upper.toFixed(3)}]`,
          significant: Boolean(point.significant),
          retention:
            typeof point.iou_retention_ratio === 'number'
              ? point.iou_retention_ratio.toFixed(3)
              : 'n/a',
        });
      });
    });

    return acc.sort((a, b) => b.delta - a.delta);
  }, [data, percentile]);

  const pairwise = useMemo(() => {
    if (!data?.strategy_comparisons) return [];
    const p = String(percentile);
    const output: Array<{ model: string; pair: string; delta: number; significant: boolean }> = [];
    Object.entries(data.strategy_comparisons).forEach(([model, byPercentile]) => {
      const comps = byPercentile[p] || [];
      comps.forEach((comp) => {
        output.push({
          model,
          pair: `${comp.strategy_a} vs ${comp.strategy_b}`,
          delta: comp.mean_delta_difference,
          significant: Boolean(comp.significant),
        });
      });
    });
    return output;
  }, [data, percentile]);

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-2 text-sm">
        <Link to="/dashboard" className="text-primary-600 hover:underline">Dashboard</Link>
        <span className="text-gray-400">/</span>
        <span className="text-gray-600">Q2 Investigation</span>
      </div>

      <h1 className="text-2xl font-bold text-gray-900">Q2 Strategy-Aware Attention Shift</h1>

      <Card>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <Select
              value={String(percentile)}
              onChange={(v) => setPercentile(Number(v))}
              options={PERCENTILES.map((p) => ({ value: String(p), label: `Top ${100 - p}%` }))}
              label="Percentile"
            />
            <Select
              value={selectedModel}
              onChange={setSelectedModel}
              options={modelOptions}
              label="Model"
            />
            <Select
              value={selectedStrategy}
              onChange={setSelectedStrategy}
              options={STRATEGY_OPTIONS}
              label="Strategy"
            />
            <Select
              value={layer}
              onChange={setLayer}
              options={Array.from({ length: 12 }, (_, i) => ({ value: String(i), label: `Layer ${i}` }))}
              label="Layer (Compare Link)"
            />
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <h3 className="font-semibold">Model × Strategy ΔIoU</h3>
            <Link
              to={`/compare?type=frozen&model=${selectedModel === 'all' ? 'dinov2' : selectedModel}&strategy=${selectedStrategy === 'all' ? '' : selectedStrategy}&layer=${layer}`}
              className="text-sm text-primary-600 hover:underline"
            >
              Open Frozen vs Fine-tuned Compare
            </Link>
          </div>
        </CardHeader>
        <CardContent>
          {isLoading && <div className="text-sm text-gray-500">Loading Q2 summary...</div>}
          {error && <div className="text-sm text-red-600">Failed to load Q2 summary.</div>}
          {!isLoading && !error && rows.length === 0 && (
            <div className="text-sm text-gray-500">No Q2 rows available for current filters.</div>
          )}
          {rows.length > 0 && (
            <div className="overflow-x-auto">
              <table className="min-w-full text-sm">
                <thead>
                  <tr className="text-left border-b">
                    <th className="py-2 pr-3">Model</th>
                    <th className="py-2 pr-3">Strategy</th>
                    <th className="py-2 pr-3">ΔIoU</th>
                    <th className="py-2 pr-3">95% CI</th>
                    <th className="py-2 pr-3">Retention</th>
                    <th className="py-2">Significant</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((row) => (
                    <tr key={`${row.model}-${row.strategy}`} className="border-b last:border-0">
                      <td className="py-2 pr-3">{row.model}</td>
                      <td className="py-2 pr-3">{row.strategy}</td>
                      <td className="py-2 pr-3 font-medium">{row.delta.toFixed(3)}</td>
                      <td className="py-2 pr-3">{row.ci}</td>
                      <td className="py-2 pr-3">{row.retention}</td>
                      <td className="py-2">{row.significant ? 'Yes' : 'No'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <h3 className="font-semibold">Cross-Strategy Paired Comparisons</h3>
        </CardHeader>
        <CardContent>
          {pairwise.length === 0 && (
            <div className="text-sm text-gray-500">No cross-strategy comparisons available.</div>
          )}
          {pairwise.length > 0 && (
            <div className="space-y-2 text-sm">
              {pairwise.map((row, index) => (
                <div key={`${row.model}-${row.pair}-${index}`} className="flex justify-between border-b pb-1">
                  <span>{row.model} - {row.pair}</span>
                  <span className="font-medium">
                    ΔΔIoU {row.delta.toFixed(3)} ({row.significant ? 'sig' : 'ns'})
                  </span>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
