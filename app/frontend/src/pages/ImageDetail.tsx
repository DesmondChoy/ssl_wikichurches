/**
 * Image detail page with attention viewer and metrics.
 */

import { useCallback, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { imagesAPI } from '../api/client';
import { useViewStore } from '../store/viewStore';
import { useModels } from '../hooks/useAttention';
import { AttentionViewer } from '../components/attention/AttentionViewer';
import { ControlPanel } from '../components/attention/ControlPanel';
import { LayerSlider } from '../components/attention/LayerSlider';
import { ImageDetailMetricsPanel } from '../components/metrics/ImageDetailMetricsPanel';
import { Card, CardContent } from '../components/ui/Card';
import { ErrorBoundary } from '../components/ui/ErrorBoundary';
import { AnnotationsCard } from '../components/image-detail/AnnotationsCard';

export function ImageDetailPage() {
  const { imageId } = useParams<{ imageId: string }>();
  const decodedId = imageId ? decodeURIComponent(imageId) : '';
  const [isPlaying, setIsPlaying] = useState(false);

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
  const canQueryProgression = !!decodedId && !!imageDetail;

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

  if (!detailLoading && !imageDetail) {
    return (
      <div className="space-y-4">
        <Link to="/" className="text-primary-600 hover:underline">
          &larr; Back to gallery
        </Link>
        <div className="rounded-lg border border-amber-200 bg-amber-50 p-4 text-amber-900">
          Image not found: {decodedId}
        </div>
      </div>
    );
  }

  if (detailLoading) {
    return (
      <div className="animate-pulse space-y-4">
        <div className="h-8 bg-gray-200 rounded w-1/4" />
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-12 xl:grid-cols-[22rem_minmax(0,1fr)_minmax(0,1fr)] xl:gap-8">
          <div className="space-y-4 lg:col-span-3 xl:col-span-1">
            <div className="h-80 rounded-lg bg-gray-200" />
            <div className="h-64 rounded-lg bg-gray-200" />
          </div>
          <div className="min-w-0 space-y-4 lg:col-span-5 xl:col-span-1">
            <div className="aspect-square rounded-lg bg-gray-200" />
            <div className="h-24 rounded-lg bg-gray-200" />
          </div>
          <div className="min-w-0 lg:col-span-4 xl:col-span-1">
            <div className="h-[32rem] rounded-lg bg-gray-200" />
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
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-12 xl:grid-cols-[22rem_minmax(0,1fr)_minmax(0,1fr)] xl:gap-8">
        <div className="space-y-4 lg:col-span-3 xl:col-span-1" data-testid="image-detail-left-column">
          <div data-testid="view-settings-panel">
            <ControlPanel />
          </div>
          {imageDetail && (
            <AnnotationsCard
              annotation={imageDetail.annotation}
              showBboxes={showBboxes}
              selectedBboxIndex={selectedBboxIndex}
              onBboxSelect={handleBboxSelect}
            />
          )}
        </div>

        {/* Center: Attention viewer */}
        <div className="min-w-0 space-y-4 lg:col-span-5 xl:col-span-1" data-testid="image-detail-center-column">
          <ErrorBoundary resetKeys={[model, layer, method, percentile]}>
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
          </ErrorBoundary>

          {/* Layer slider */}
          <Card>
            <CardContent>
              <LayerSlider
                currentLayer={layer}
                maxLayers={maxLayers}
                onChange={setLayer}
                isPlaying={isPlaying}
                onPlayingChange={setIsPlaying}
                playSpeed={400}
              />
            </CardContent>
          </Card>
        </div>

        <div className="min-w-0 lg:col-span-4 xl:col-span-1" data-testid="image-detail-right-column">
          <ErrorBoundary resetKeys={[model, layer, percentile, method, selectedBboxIndex, isPlaying]}>
            <ImageDetailMetricsPanel
              imageId={decodedId}
              model={model}
              percentile={percentile}
              method={method}
              selectedBboxIndex={selectedBboxIndex}
              currentLayer={layer}
              isPlaying={isPlaying}
              enabled={canQueryProgression}
            />
          </ErrorBoundary>
        </div>
      </div>
    </div>
  );
}
