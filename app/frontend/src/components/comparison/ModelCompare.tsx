/**
 * Side-by-side model comparison component with interactive similarity heatmaps.
 *
 * Displays ground truth bounding boxes on both panels. When a bbox is clicked,
 * similarity heatmaps are shown for both models simultaneously, allowing users
 * to compare how different models "see" the same architectural feature.
 */

import { useState, useMemo } from 'react';
import { useModelComparison } from '../../hooks/useAttention';
import { Select } from '../ui/Select';
import { Card, CardContent } from '../ui/Card';
import { SimilarityViewer } from './SimilarityViewer';
import { ErrorBoundary } from '../ui/ErrorBoundary';
import { renderHeatmapLegend } from '../../utils/renderHeatmap';
import type { BoundingBox } from '../../types';

interface ModelCompareProps {
  imageId: string;
  layer: number;
  percentile: number;
  availableModels: string[];
  bboxes: BoundingBox[];
}

export function ModelCompare({
  imageId,
  layer,
  percentile,
  availableModels,
  bboxes,
}: ModelCompareProps) {
  const [leftModel, setLeftModel] = useState(availableModels[0] || 'dinov2');
  const [rightModel, setRightModel] = useState(availableModels[1] || 'clip');
  const [selectedBboxIndex, setSelectedBboxIndex] = useState<number | null>(null);

  const { data: comparison, isLoading, error } = useModelComparison(
    imageId,
    [leftModel, rightModel],
    layer,
    percentile
  );

  const modelOptions = availableModels.map((m) => ({
    value: m,
    label: m.charAt(0).toUpperCase() + m.slice(1),
  }));

  // Generate legend URL (static, computed once)
  const legendUrl = useMemo(() => renderHeatmapLegend(200, 16), []);

  // Get selected bbox for display
  const selectedBbox = selectedBboxIndex !== null ? bboxes[selectedBboxIndex] : null;

  return (
    <div className="space-y-4">
      <div className="flex gap-4">
        <Select
          value={leftModel}
          onChange={setLeftModel}
          options={modelOptions}
          label="Left Model"
          className="flex-1"
        />
        <Select
          value={rightModel}
          onChange={setRightModel}
          options={modelOptions}
          label="Right Model"
          className="flex-1"
        />
      </div>

      <div className="grid grid-cols-2 gap-4">
        {/* Left model */}
        <Card>
          <ErrorBoundary resetKeys={[imageId, leftModel, layer]}>
            <SimilarityViewer
              imageId={imageId}
              model={leftModel}
              layer={layer}
              bboxes={bboxes}
              selectedBboxIndex={selectedBboxIndex}
              onBboxSelect={setSelectedBboxIndex}
            />
          </ErrorBoundary>
          <CardContent>
            {comparison?.results.find((r) => r.model === leftModel) && (
              <div className="text-sm">
                <span className="font-medium">IoU:</span>{' '}
                {comparison.results
                  .find((r) => r.model === leftModel)
                  ?.iou.toFixed(3)}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Right model */}
        <Card>
          <ErrorBoundary resetKeys={[imageId, rightModel, layer]}>
            <SimilarityViewer
              imageId={imageId}
              model={rightModel}
              layer={layer}
              bboxes={bboxes}
              selectedBboxIndex={selectedBboxIndex}
              onBboxSelect={setSelectedBboxIndex}
            />
          </ErrorBoundary>
          <CardContent>
            {comparison?.results.find((r) => r.model === rightModel) && (
              <div className="text-sm">
                <span className="font-medium">IoU:</span>{' '}
                {comparison.results
                  .find((r) => r.model === rightModel)
                  ?.iou.toFixed(3)}
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Selection info and controls */}
      {selectedBbox && (
        <div className="flex items-center justify-between bg-gray-50 rounded-lg px-4 py-2">
          <span className="text-sm text-gray-700">
            <span className="font-medium">Selected:</span>{' '}
            {selectedBbox.label_name || `Feature ${selectedBbox.label}`}
          </span>
          <button
            onClick={() => setSelectedBboxIndex(null)}
            className="px-3 py-1 text-sm bg-gray-200 hover:bg-gray-300 rounded transition-colors"
          >
            Clear Selection
          </button>
        </div>
      )}

      {/* Colormap legend (shown when bbox is selected) */}
      {selectedBbox && (
        <div className="flex items-center justify-center gap-2 text-sm text-gray-600">
          <span>Low similarity</span>
          <img src={legendUrl} alt="Similarity scale" className="h-4 rounded" />
          <span>High similarity</span>
        </div>
      )}

      {/* Hint for users when no bbox is selected */}
      {bboxes.length > 0 && !selectedBbox && (
        <p className="text-center text-sm text-gray-500">
          Click on a bounding box to compare similarity heatmaps between models
        </p>
      )}

      {isLoading && (
        <div className="text-center text-gray-500">Loading comparison...</div>
      )}
      {error && (
        <div className="text-center text-red-500">Failed to load comparison</div>
      )}
    </div>
  );
}
