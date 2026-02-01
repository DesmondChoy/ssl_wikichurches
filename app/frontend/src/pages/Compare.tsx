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

export function ComparePage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const imageId = searchParams.get('image') || '';
  const comparisonType = searchParams.get('type') || 'models';

  const { model, layer, percentile } = useViewStore();

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
  ];

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
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Select
              value={imageId}
              onChange={(v) => setSearchParams({ image: v, type: comparisonType })}
              options={[{ value: '', label: 'Select an image...' }, ...imageOptions]}
              label="Image"
            />

            <Select
              value={comparisonType}
              onChange={(v) => setSearchParams({ image: imageId, type: v })}
              options={comparisonTypes}
              label="Comparison Type"
            />

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
          percentile={percentile}
          availableModels={imageDetail.available_models}
          bboxes={imageDetail.annotation.bboxes}
        />
      )}

      {imageId && comparisonType === 'frozen' && (
        <FrozenVsFinetuned
          imageId={imageId}
          model={model}
          layer={layer}
        />
      )}
    </div>
  );
}
