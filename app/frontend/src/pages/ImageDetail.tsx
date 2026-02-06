/**
 * Image detail page with attention viewer and metrics.
 */

import { useCallback } from 'react';
import { useParams, Link } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { imagesAPI } from '../api/client';
import { useViewStore } from '../store/viewStore';
import { useImageMetrics, useBboxMetrics, useModels } from '../hooks/useAttention';
import { AttentionViewer } from '../components/attention/AttentionViewer';
import { ControlPanel } from '../components/attention/ControlPanel';
import { LayerSlider } from '../components/attention/LayerSlider';
import { IoUDisplay } from '../components/metrics/IoUDisplay';
import { Card, CardHeader, CardContent } from '../components/ui/Card';

export function ImageDetailPage() {
  const { imageId } = useParams<{ imageId: string }>();
  const decodedId = imageId ? decodeURIComponent(imageId) : '';

  const { model, layer, method, percentile, showBboxes, selectedBboxIndex, setLayer, setSelectedBboxIndex } = useViewStore();

  // Get models config for per-model layer counts
  const { data: modelsData } = useModels();
  const maxLayers = modelsData?.num_layers_per_model?.[model] ?? modelsData?.num_layers ?? 12;

  const handleBboxSelect = useCallback((index: number | null) => {
    setSelectedBboxIndex(index);
  }, [setSelectedBboxIndex]);

  // Fetch image details
  const { data: imageDetail, isLoading: detailLoading, error } = useQuery({
    queryKey: ['imageDetail', decodedId],
    queryFn: () => imagesAPI.getDetail(decodedId),
    enabled: !!decodedId,
  });

  // Fetch metrics (union of all bboxes — always running for instant fallback)
  const { data: metrics, isLoading: metricsLoading } = useImageMetrics(
    decodedId,
    model,
    layer,
    percentile,
    method
  );

  // Fetch per-bbox metrics (only when a bbox is selected)
  const { data: bboxMetrics, isLoading: bboxMetricsLoading } = useBboxMetrics(
    decodedId, model, layer, selectedBboxIndex, percentile, method
  );

  // Choose which metrics to display based on bbox selection
  const effectiveMetrics = selectedBboxIndex !== null ? bboxMetrics : metrics;
  const effectiveLoading = selectedBboxIndex !== null ? bboxMetricsLoading : metricsLoading;
  const metricsContext = selectedBboxIndex !== null && imageDetail
    ? (imageDetail.annotation.bboxes[selectedBboxIndex]?.label_name || `Bbox #${selectedBboxIndex + 1}`)
    : null;

  if (!decodedId) {
    return <div>Invalid image ID</div>;
  }

  if (error) {
    return (
      <div className="space-y-4">
        <Link to="/" className="text-primary-600 hover:underline">
          &larr; Back to gallery
        </Link>
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-700">
          Failed to load image: {decodedId}
        </div>
      </div>
    );
  }

  if (detailLoading) {
    return (
      <div className="animate-pulse space-y-4">
        <div className="h-8 bg-gray-200 rounded w-1/4" />
        <div className="grid grid-cols-1 lg:grid-cols-10 gap-6">
          <div className="lg:col-span-2">
            <div className="h-64 bg-gray-200 rounded-lg" />
          </div>
          <div className="lg:col-span-5 aspect-square bg-gray-200 rounded-lg" />
          <div className="lg:col-span-3 space-y-4">
            <div className="h-40 bg-gray-200 rounded-lg" />
            <div className="h-32 bg-gray-200 rounded-lg" />
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Breadcrumb */}
      <div className="flex items-center gap-2 text-sm">
        <Link to="/" className="text-primary-600 hover:underline">
          Gallery
        </Link>
        <span className="text-gray-400">/</span>
        <span className="text-gray-600">{decodedId}</span>
      </div>

      {/* Main content */}
      <div className="grid grid-cols-1 lg:grid-cols-10 gap-6">
        {/* Left: Annotations */}
        <div className="lg:col-span-2">
          {imageDetail && (
            <Card>
              <CardHeader>
                <h3 className="font-semibold">Annotations</h3>
              </CardHeader>
              <CardContent className="space-y-3">
                <div>
                  <span className="text-gray-500 text-sm">Styles:</span>
                  <div className="flex gap-1 mt-1 flex-wrap">
                    {imageDetail.annotation.style_names.map((style) => (
                      <span
                        key={style}
                        className="px-2 py-0.5 bg-primary-100 text-primary-700 text-sm rounded"
                      >
                        {style}
                      </span>
                    ))}
                  </div>
                </div>

                <div>
                  <span className="text-gray-500 text-sm">Bounding Boxes:</span>
                  <span className="ml-2 font-medium">
                    {imageDetail.annotation.num_bboxes}
                  </span>
                </div>

                {showBboxes && (
                  <div className="text-xs text-green-600 bg-green-50 px-2 py-1 rounded">
                    Click a bounding box to see feature similarity heatmap
                  </div>
                )}

                <div className="text-xs text-gray-500 space-y-1 max-h-48 overflow-y-auto">
                  {imageDetail.annotation.bboxes.map((bbox, i) => (
                    <div
                      key={i}
                      className={`flex justify-between cursor-pointer hover:bg-gray-100 px-1 rounded ${
                        selectedBboxIndex === i ? 'bg-green-100 text-green-700' : ''
                      }`}
                      onClick={() => {
                        if (showBboxes) {
                          handleBboxSelect(selectedBboxIndex === i ? null : i);
                        }
                      }}
                    >
                      <span>{bbox.label_name || `Label ${bbox.label}`}</span>
                      <span className="text-gray-400">
                        {(bbox.width * 100).toFixed(0)}% x {(bbox.height * 100).toFixed(0)}%
                      </span>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </div>

        {/* Center: Attention viewer */}
        <div className="lg:col-span-5 space-y-4">
          <AttentionViewer
            imageId={decodedId}
            model={model}
            layer={layer}
            method={method}
            percentile={percentile}
            showBboxes={showBboxes}
            bboxes={imageDetail?.annotation.bboxes}
            selectedBboxIndex={selectedBboxIndex}
            onBboxSelect={handleBboxSelect}
            className="aspect-square"
          />

          {/* Layer slider */}
          <Card>
            <CardContent>
              <LayerSlider
                currentLayer={layer}
                maxLayers={maxLayers}
                onChange={setLayer}
                playSpeed={400}
              />
            </CardContent>
          </Card>
        </div>

        {/* Right: Controls and metrics */}
        <div className="lg:col-span-3 space-y-4">
          <ControlPanel />

          {/* Metrics */}
          <Card>
            <CardHeader>
              <h3 className="font-semibold">Metrics</h3>
            </CardHeader>
            <CardContent>
              <IoUDisplay metrics={effectiveMetrics} isLoading={effectiveLoading} bboxLabel={metricsContext} />
            </CardContent>
          </Card>

          {/* Navigation */}
          <div className="flex gap-2">
            <Link
              to={`/compare?image=${encodeURIComponent(decodedId)}`}
              className="flex-1 py-2 text-center bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors"
            >
              Compare Models
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}
