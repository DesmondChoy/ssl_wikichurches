/**
 * Image detail page with attention viewer and metrics.
 */

import { useParams, Link } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { imagesAPI } from '../api/client';
import { useViewStore } from '../store/viewStore';
import { useImageMetrics } from '../hooks/useAttention';
import { AttentionViewer } from '../components/attention/AttentionViewer';
import { ControlPanel } from '../components/attention/ControlPanel';
import { LayerSlider } from '../components/attention/LayerSlider';
import { IoUDisplay } from '../components/metrics/IoUDisplay';
import { Card, CardHeader, CardContent } from '../components/ui/Card';

export function ImageDetailPage() {
  const { imageId } = useParams<{ imageId: string }>();
  const decodedId = imageId ? decodeURIComponent(imageId) : '';

  const { model, layer, percentile, showBboxes, setLayer } = useViewStore();

  // Fetch image details
  const { data: imageDetail, isLoading: detailLoading, error } = useQuery({
    queryKey: ['imageDetail', decodedId],
    queryFn: () => imagesAPI.getDetail(decodedId),
    enabled: !!decodedId,
  });

  // Fetch metrics
  const { data: metrics, isLoading: metricsLoading } = useImageMetrics(
    decodedId,
    model,
    layer,
    percentile
  );

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
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 aspect-square bg-gray-200 rounded-lg" />
          <div className="space-y-4">
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
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: Attention viewer */}
        <div className="lg:col-span-2 space-y-4">
          <AttentionViewer
            imageId={decodedId}
            model={model}
            layer={layer}
            showBboxes={showBboxes}
            className="aspect-square"
          />

          {/* Layer slider */}
          <Card>
            <CardContent>
              <LayerSlider
                currentLayer={layer}
                onChange={setLayer}
                playSpeed={400}
              />
            </CardContent>
          </Card>
        </div>

        {/* Right: Controls and info */}
        <div className="space-y-4">
          <ControlPanel />

          {/* Metrics */}
          <Card>
            <CardHeader>
              <h3 className="font-semibold">Metrics</h3>
            </CardHeader>
            <CardContent>
              <IoUDisplay metrics={metrics} isLoading={metricsLoading} />
            </CardContent>
          </Card>

          {/* Annotation info */}
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

                <div className="text-xs text-gray-500 space-y-1 max-h-48 overflow-y-auto">
                  {imageDetail.annotation.bboxes.map((bbox, i) => (
                    <div key={i} className="flex justify-between">
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

          {/* Navigation */}
          <div className="flex gap-2">
            <Link
              to={`/compare?image=${encodeURIComponent(decodedId)}`}
              className="flex-1 py-2 text-center bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors"
            >
              Compare Models
            </Link>
            <Link
              to={`/layers?image=${encodeURIComponent(decodedId)}&model=${model}`}
              className="flex-1 py-2 text-center bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors"
            >
              Layer Analysis
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}
