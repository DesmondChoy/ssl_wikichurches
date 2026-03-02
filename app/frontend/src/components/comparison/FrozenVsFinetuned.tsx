/**
 * Frozen vs Fine-tuned comparison using react-compare-slider.
 */

import { useQuery } from '@tanstack/react-query';
import { ReactCompareSlider, ReactCompareSliderImage } from 'react-compare-slider';
import { comparisonAPI } from '../../api/client';

interface FrozenVsFinetunedProps {
  imageId: string;
  model: string;
  layer: number;
}

export function FrozenVsFinetuned({ imageId, model, layer }: FrozenVsFinetunedProps) {
  const {
    data,
    isLoading,
    error,
  } = useQuery({
    queryKey: ['frozen-vs-finetuned', imageId, model, layer],
    queryFn: () => comparisonAPI.compareFrozenVsFinetuned(imageId, model, layer),
    enabled: Boolean(imageId && model),
  });

  if (isLoading) {
    return (
      <div className="bg-gray-50 border border-gray-200 rounded-lg p-4 text-sm text-gray-600">
        Loading frozen vs fine-tuned availability...
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-sm text-red-700">
        Failed to load frozen vs fine-tuned comparison data.
      </div>
    );
  }

  if (!data) {
    return null;
  }

  const frozenUrl = data.frozen.url;
  const finetunedUrl = data.finetuned.url;
  const sliderAvailable =
    data.frozen.available &&
    data.finetuned.available &&
    typeof frozenUrl === 'string' &&
    typeof finetunedUrl === 'string';

  if (!sliderAvailable) {
    return (
      <div className="space-y-4">
        {frozenUrl && (
          <div className="relative">
            <img
              src={frozenUrl}
              alt={`${model} frozen attention`}
              className="w-full h-auto rounded-lg"
            />
            <div className="absolute bottom-2 left-2 px-2 py-1 bg-black/50 text-white text-xs rounded">
              {model} (Frozen/Pretrained)
            </div>
          </div>
        )}

        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
          <h4 className="font-medium text-yellow-800">Fine-tuned Overlay Unavailable</h4>
          <p className="text-sm text-yellow-700 mt-1">
            {data.finetuned.note}
          </p>
          <p className="text-sm text-yellow-700 mt-1">
            The slider comparison appears automatically when both frozen and fine-tuned overlays are cached.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <div className="flex justify-between text-sm text-gray-600">
        <span>Frozen (Pretrained)</span>
        <span>Fine-tuned</span>
      </div>

      <ReactCompareSlider
        itemOne={
          <ReactCompareSliderImage
            src={frozenUrl!}
            alt="Frozen model attention"
          />
        }
        itemTwo={
          <ReactCompareSliderImage
            src={finetunedUrl!}
            alt="Fine-tuned model attention"
          />
        }
        className="rounded-lg overflow-hidden"
        position={50}
      />

      <p className="text-xs text-gray-500 text-center">
        Drag slider to compare frozen vs fine-tuned attention
      </p>
    </div>
  );
}
