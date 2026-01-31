/**
 * Frozen vs Fine-tuned comparison using react-compare-slider.
 */

import { ReactCompareSlider, ReactCompareSliderImage } from 'react-compare-slider';
import { attentionAPI } from '../../api/client';

interface FrozenVsFinetunedProps {
  imageId: string;
  model: string;
  layer: number;
}

export function FrozenVsFinetuned({ imageId, model, layer }: FrozenVsFinetunedProps) {
  const frozenUrl = attentionAPI.getOverlayUrl(imageId, model, layer, false);
  const finetunedUrl = attentionAPI.getOverlayUrl(imageId, `${model}_finetuned`, layer, false);

  // Note: Fine-tuned models not yet available (Phase 5)
  const finetunedAvailable = false;

  if (!finetunedAvailable) {
    return (
      <div className="space-y-4">
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

        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
          <h4 className="font-medium text-yellow-800">Fine-tuned Model Not Yet Available</h4>
          <p className="text-sm text-yellow-700 mt-1">
            Fine-tuned models will be available after Phase 5 training is complete.
            The slider comparison will work once both frozen and fine-tuned versions exist.
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
            src={frozenUrl}
            alt="Frozen model attention"
          />
        }
        itemTwo={
          <ReactCompareSliderImage
            src={finetunedUrl}
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
