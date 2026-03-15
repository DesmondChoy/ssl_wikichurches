/**
 * Model comparison page.
 */

import { useSearchParams, Link } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { imagesAPI } from '../api/client';
import { useViewStore } from '../store/viewStore';
import { ModelCompare } from '../components/comparison/ModelCompare';
import { FrozenVsFinetuned } from '../components/comparison/FrozenVsFinetuned';
import { Card, CardContent } from '../components/ui/Card';
import { Select } from '../components/ui/Select';

const PERCENTILE_OPTIONS = [90, 80, 70, 60, 50].map((value) => ({
  value: String(value),
  label: `Top ${100 - value}%`,
}));

export function ComparePage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const imageId = searchParams.get('image') || '';
  const comparisonType = searchParams.get('type') || 'models';
  const strategy = searchParams.get('strategy') || '';
  const strategyA = searchParams.get('strategyA') || 'linear_probe';
  const strategyB = searchParams.get('strategyB') || 'full';
  const modelParam = searchParams.get('model') || '';
  const layerParam = searchParams.get('layer') || '';
  const percentileParam = searchParams.get('percentile') || '';

  const { model, layer, method, percentile, setModel, setLayer, setPercentile } = useViewStore();
  const compareModel = modelParam || model;
  const compareLayer = layerParam ? Number(layerParam) : layer;
  const comparePercentile = percentileParam ? Number(percentileParam) : percentile;

  // Fetch image list for selection
  const { data: images } = useQuery({
    queryKey: ['images'],
    queryFn: () => imagesAPI.list({ limit: 139 }),
  });

  // Fetch image details for available models
  const { data: imageDetail } = useQuery({
    queryKey: ['imageDetail', imageId],
    queryFn: () => imagesAPI.getDetail(imageId),
    enabled: !!imageId,
  });

  const imageOptions = images?.map((img) => ({
    value: img.image_id,
    label: `${img.image_id.split('_')[0]} (${img.style_names.join(', ')})`,
  })) || [];

  const comparisonTypes = [
    { value: 'models', label: 'Model vs Model' },
    { value: 'frozen', label: 'Frozen vs Fine-tuned' },
    { value: 'methods', label: 'Fine-tuning Method vs Method' },
  ];
  const strategyOptions = [
    { value: '', label: 'Auto (legacy)' },
    { value: 'linear_probe', label: 'Linear Probe' },
    { value: 'lora', label: 'LoRA' },
    { value: 'full', label: 'Full Fine-tune' },
  ];

  const buildSearchParams = (overrides?: Record<string, string>) => ({
    image: imageId,
    type: comparisonType,
    strategy,
    strategyA,
    strategyB,
    model: compareModel,
    layer: String(compareLayer),
    percentile: String(comparePercentile),
    ...overrides,
  });

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-2 text-sm">
        <Link to="/" className="text-primary-600 hover:underline">
          Gallery
        </Link>
        <span className="text-gray-400">/</span>
        <span className="text-gray-600">Compare</span>
      </div>

      <h1 className="text-2xl font-bold text-gray-900">Model Comparison</h1>

      {/* Controls */}
      <Card>
        <CardContent>
          <div className="grid grid-cols-1 gap-4 md:grid-cols-4">
            <Select
              value={imageId}
              onChange={(v) => setSearchParams(buildSearchParams({ image: v }))}
              options={[{ value: '', label: 'Select an image...' }, ...imageOptions]}
              label="Image"
            />

            <Select
              value={comparisonType}
              onChange={(v) => setSearchParams(buildSearchParams({ type: v }))}
              options={comparisonTypes}
              label="Comparison Type"
            />

            {comparisonType === 'frozen' || comparisonType === 'methods' ? (
              <Select
                value={compareModel}
                onChange={(v) => {
                  setModel(v);
                  setSearchParams(buildSearchParams({ model: v }));
                }}
                options={(imageDetail?.available_models || ['dinov2']).filter((m) => m !== 'resnet50').map((m) => ({
                  value: m,
                  label: m,
                }))}
                label="Model"
              />
            ) : (
              <div className="flex items-end">
                {imageId && (
                  <Link
                    to={`/image/${encodeURIComponent(imageId)}`}
                    className="px-4 py-2 text-sm text-primary-600 hover:underline"
                  >
                    View Details &rarr;
                  </Link>
                )}
              </div>
            )}

            <Select
              value={String(comparePercentile)}
              onChange={(value) => {
                const nextPercentile = Number(value);
                setPercentile(nextPercentile);
                setSearchParams(buildSearchParams({ percentile: value }));
              }}
              options={PERCENTILE_OPTIONS}
              label="Percentile"
            />

            {comparisonType === 'frozen' && (
              <Select
                value={strategy}
                onChange={(v) => setSearchParams(buildSearchParams({ strategy: v }))}
                options={strategyOptions}
                label="Strategy"
              />
            )}
            {comparisonType === 'methods' && (
              <Select
                value={strategyA}
                onChange={(v) => setSearchParams(buildSearchParams({ strategyA: v }))}
                options={strategyOptions.filter((option) => option.value)}
                label="Left Method"
              />
            )}
            {comparisonType === 'methods' && (
              <Select
                value={strategyB}
                onChange={(v) => setSearchParams(buildSearchParams({ strategyB: v }))}
                options={strategyOptions.filter((option) => option.value)}
                label="Right Method"
              />
            )}
            {(comparisonType === 'frozen' || comparisonType === 'methods') && (
              <Select
                value={String(compareLayer)}
                onChange={(v) => {
                  const nextLayer = Number(v);
                  setLayer(nextLayer);
                  setSearchParams(buildSearchParams({ layer: String(nextLayer) }));
                }}
                options={Array.from({ length: 12 }, (_, i) => ({ value: String(i), label: `Layer ${i}` }))}
                label="Layer"
              />
            )}
          </div>
        </CardContent>
      </Card>

      {/* Comparison content */}
      {!imageId && (
        <div className="bg-gray-50 rounded-lg p-8 text-center text-gray-500">
          Select an image above to start comparing
        </div>
      )}

      {imageId && comparisonType === 'models' && imageDetail && (
        <ModelCompare
          imageId={imageId}
          layer={layer}
          percentile={comparePercentile}
          method={method}
          availableModels={imageDetail.available_models}
          bboxes={imageDetail.annotation.bboxes}
        />
      )}

      {imageId && comparisonType === 'frozen' && (
        <FrozenVsFinetuned
          imageId={imageId}
          model={compareModel}
          layer={compareLayer}
          percentile={comparePercentile}
          mode="frozen"
          strategy={strategy || undefined}
          bboxes={imageDetail?.annotation.bboxes || []}
          showBboxes
        />
      )}

      {imageId && comparisonType === 'methods' && (
        <FrozenVsFinetuned
          imageId={imageId}
          model={compareModel}
          layer={compareLayer}
          percentile={comparePercentile}
          mode="methods"
          strategyA={strategyA}
          strategyB={strategyB}
          bboxes={imageDetail?.annotation.bboxes || []}
          showBboxes
        />
      )}
    </div>
  );
}
