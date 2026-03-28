/**
 * Model and variant comparison page.
 */

import { useEffect, useState } from 'react';
import { useSearchParams, Link } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { imagesAPI } from '../api/client';
import { useModels } from '../hooks/useAttention';
import { useViewStore } from '../store/viewStore';
import { ModelCompare } from '../components/comparison/ModelCompare';
import { VariantCompare } from '../components/comparison/FrozenVsFinetuned';
import { Card, CardContent } from '../components/ui/Card';
import { Select } from '../components/ui/Select';
import {
  ANALYSIS_METRIC_METADATA,
  ANALYSIS_METRIC_OPTIONS,
  COMPARE_VARIANT_OPTIONS,
  isAnalysisMetric,
  isCompareVariantId,
} from '../constants/metricMetadata';
import type { AnalysisMetric, CompareVariantId } from '../types';

const PERCENTILE_OPTIONS = [90, 80, 70, 60, 50].map((value) => ({
  value: String(value),
  label: `Top ${100 - value}%`,
}));

type ComparisonType = 'models' | 'variants';

function clampLayer(value: string | null, fallback: number) {
  if (!value) return fallback;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

export function ComparePage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const [isPlaying, setIsPlaying] = useState(false);
  const imageId = searchParams.get('image') || '';
  const comparisonTypeParam = searchParams.get('type');
  const comparisonType: ComparisonType = comparisonTypeParam === 'variants' ? 'variants' : 'models';
  const legacyStrategy = searchParams.get('strategy') || '';
  const strategyA = searchParams.get('strategyA') || 'linear_probe';
  const strategyB = searchParams.get('strategyB') || 'full';
  const modelParam = searchParams.get('model') || '';
  const layerParam = searchParams.get('layer');
  const percentileParam = searchParams.get('percentile');
  const metricParam = searchParams.get('metric');
  const leftVariantParam = searchParams.get('left_variant');
  const rightVariantParam = searchParams.get('right_variant');

  const {
    model,
    layer,
    method,
    percentile,
    setModel,
    setLayer,
    setPercentile,
  } = useViewStore();
  const compareModel = modelParam || model;
  const compareLayer = clampLayer(layerParam, layer);
  const comparePercentile = percentileParam ? Number(percentileParam) : percentile;
  const compareMetric: AnalysisMetric = isAnalysisMetric(metricParam) ? metricParam : 'iou';
  const metricMetadata = ANALYSIS_METRIC_METADATA[compareMetric];
  const thresholdFree = metricMetadata.thresholdFree;
  const leftVariant: CompareVariantId = isCompareVariantId(leftVariantParam) ? leftVariantParam : 'frozen';
  const rightVariant: CompareVariantId = isCompareVariantId(rightVariantParam) ? rightVariantParam : 'full';

  useEffect(() => {
    if (comparisonTypeParam === 'frozen') {
      const nextRightVariant: CompareVariantId =
        legacyStrategy === 'linear_probe' || legacyStrategy === 'lora' || legacyStrategy === 'full'
          ? legacyStrategy
          : 'full';
      setSearchParams(
        {
          image: imageId,
          type: 'variants',
          model: compareModel,
          layer: String(compareLayer),
          percentile: String(comparePercentile),
          metric: compareMetric,
          left_variant: 'frozen',
          right_variant: nextRightVariant,
        },
        { replace: true },
      );
      return;
    }

    if (comparisonTypeParam === 'methods') {
      setSearchParams(
        {
          image: imageId,
          type: 'variants',
          model: compareModel,
          layer: String(compareLayer),
          percentile: String(comparePercentile),
          metric: compareMetric,
          left_variant: isCompareVariantId(strategyA) ? strategyA : 'linear_probe',
          right_variant: isCompareVariantId(strategyB) ? strategyB : 'full',
        },
        { replace: true },
      );
    }
  }, [
    compareLayer,
    compareMetric,
    compareModel,
    comparePercentile,
    comparisonTypeParam,
    imageId,
    legacyStrategy,
    setSearchParams,
    strategyA,
    strategyB,
  ]);

  const { data: images } = useQuery({
    queryKey: ['images'],
    queryFn: () => imagesAPI.list({ limit: 139 }),
  });
  const { data: modelsData } = useModels();

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
    { value: 'variants', label: 'Variant vs Variant' },
  ];
  const availableVariantModels = imageDetail?.available_models?.length
    ? imageDetail.available_models
    : modelsData?.models || [];
  const filteredVariantModels = availableVariantModels.filter((entry) => entry !== 'resnet50');
  const variantModelValues =
    compareModel && !filteredVariantModels.includes(compareModel)
      ? [compareModel, ...filteredVariantModels]
      : filteredVariantModels;
  const variantModelOptions = Array.from(new Set(variantModelValues)).map((entry) => ({
    value: entry,
    label: entry,
  }));
  const headerControlsClassName = comparisonType === 'models'
    ? 'grid grid-cols-1 gap-4 md:grid-cols-3'
    : 'grid grid-cols-1 gap-4 md:grid-cols-4';

  const buildSearchParams = (overrides?: Record<string, string>) => ({
    image: imageId,
    type: comparisonType,
    model: compareModel,
    layer: String(compareLayer),
    percentile: String(comparePercentile),
    metric: compareMetric,
    left_variant: leftVariant,
    right_variant: rightVariant,
    ...overrides,
  });

  const updateLayer = (nextLayer: number, options?: { replace?: boolean }) => {
    setLayer(nextLayer);
    setSearchParams(buildSearchParams({ layer: String(nextLayer) }), { replace: options?.replace ?? false });
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-2 text-sm">
        <Link to="/" className="text-primary-600 hover:underline">
          Gallery
        </Link>
        <span className="text-gray-400">/</span>
        <span className="text-gray-600">Compare</span>
      </div>

      <h1 className="text-2xl font-bold text-gray-900">Model Comparison</h1>

      <Card>
        <CardContent>
          <div className={headerControlsClassName} data-testid="compare-header-controls">
            <Select
              value={imageId}
              onChange={(value) => setSearchParams(buildSearchParams({ image: value }))}
              options={[{ value: '', label: 'Select an image...' }, ...imageOptions]}
              label="Image"
            />

            <Select
              value={comparisonType}
              onChange={(value) => setSearchParams(buildSearchParams({ type: value }))}
              options={comparisonTypes}
              label="Comparison Type"
            />

            <Select
              value={String(comparePercentile)}
              onChange={(value) => {
                const nextPercentile = Number(value);
                setPercentile(nextPercentile);
                setSearchParams(buildSearchParams({ percentile: value }));
              }}
              options={PERCENTILE_OPTIONS}
              label="Percentile"
              disabled={comparisonType === 'variants' && thresholdFree}
            />

            {comparisonType === 'variants' && (
              <>
                <Select
                  value={compareModel}
                  onChange={(value) => {
                    setModel(value);
                    setSearchParams(buildSearchParams({ model: value }));
                  }}
                  options={variantModelOptions}
                  label="Model"
                />
                <Select
                  value={compareMetric}
                  onChange={(value) => setSearchParams(buildSearchParams({ metric: value }))}
                  options={ANALYSIS_METRIC_OPTIONS}
                  label="Metric"
                />
                <Select
                  value={leftVariant}
                  onChange={(value) => setSearchParams(buildSearchParams({ left_variant: value }))}
                  options={COMPARE_VARIANT_OPTIONS}
                  label="Left Variant"
                />
                <Select
                  value={rightVariant}
                  onChange={(value) => setSearchParams(buildSearchParams({ right_variant: value }))}
                  options={COMPARE_VARIANT_OPTIONS}
                  label="Right Variant"
                />
              </>
            )}
          </div>
          {comparisonType === 'variants' && thresholdFree && (
            <p className="mt-3 text-sm text-slate-600">
              {metricMetadata.optionLabel} is threshold-free, so percentile stays visible for consistency but only affects IoU-based analysis.
            </p>
          )}
        </CardContent>
      </Card>

      {!imageId && (
        <div className="rounded-lg bg-gray-50 p-8 text-center text-gray-500">
          Select an image above to start comparing
        </div>
      )}

      {imageId && comparisonType === 'models' && imageDetail && (
        <ModelCompare
          imageId={imageId}
          layer={compareLayer}
          percentile={comparePercentile}
          method={method}
          availableModels={imageDetail.available_models}
          bboxes={imageDetail.annotation.bboxes}
        />
      )}

      {imageId && comparisonType === 'variants' && (
        <VariantCompare
          imageId={imageId}
          model={compareModel}
          layer={compareLayer}
          percentile={comparePercentile}
          metric={compareMetric}
          leftVariant={leftVariant}
          rightVariant={rightVariant}
          isPlaying={isPlaying}
          onPlayingChange={setIsPlaying}
          onLayerChange={updateLayer}
          bboxes={imageDetail?.annotation.bboxes || []}
          showBboxes
        />
      )}
    </div>
  );
}
