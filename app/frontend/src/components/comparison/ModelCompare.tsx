/**
 * Side-by-side model comparison component.
 */

import { useState } from 'react';
import { attentionAPI } from '../../api/client';
import { useModelComparison } from '../../hooks/useAttention';
import { Select } from '../ui/Select';
import { Card, CardContent } from '../ui/Card';

interface ModelCompareProps {
  imageId: string;
  layer: number;
  percentile: number;
  availableModels: string[];
}

export function ModelCompare({
  imageId,
  layer,
  percentile,
  availableModels,
}: ModelCompareProps) {
  const [leftModel, setLeftModel] = useState(availableModels[0] || 'dinov2');
  const [rightModel, setRightModel] = useState(availableModels[1] || 'clip');

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

  const leftUrl = attentionAPI.getOverlayUrl(imageId, leftModel, layer, false);
  const rightUrl = attentionAPI.getOverlayUrl(imageId, rightModel, layer, false);

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
          <div className="relative">
            <img
              src={leftUrl}
              alt={`${leftModel} attention`}
              className="w-full h-auto"
            />
            <div className="absolute bottom-2 left-2 px-2 py-1 bg-black/50 text-white text-xs rounded">
              {leftModel}
            </div>
          </div>
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
          <div className="relative">
            <img
              src={rightUrl}
              alt={`${rightModel} attention`}
              className="w-full h-auto"
            />
            <div className="absolute bottom-2 left-2 px-2 py-1 bg-black/50 text-white text-xs rounded">
              {rightModel}
            </div>
          </div>
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

      {isLoading && (
        <div className="text-center text-gray-500">Loading comparison...</div>
      )}
      {error && (
        <div className="text-center text-red-500">Failed to load comparison</div>
      )}
    </div>
  );
}
