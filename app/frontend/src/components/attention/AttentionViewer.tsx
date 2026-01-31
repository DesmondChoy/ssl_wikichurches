/**
 * Main attention heatmap viewer with overlay controls.
 */

import { useState } from 'react';
import { attentionAPI, imagesAPI } from '../../api/client';

interface AttentionViewerProps {
  imageId: string;
  model: string;
  layer: number;
  showBboxes: boolean;
  className?: string;
}

export function AttentionViewer({
  imageId,
  model,
  layer,
  showBboxes,
  className = '',
}: AttentionViewerProps) {
  const [showOverlay, setShowOverlay] = useState(true);
  const [imageError, setImageError] = useState(false);

  const overlayUrl = attentionAPI.getOverlayUrl(imageId, model, layer, showBboxes);
  const originalUrl = imagesAPI.getImageUrl(imageId, 224);

  if (imageError) {
    return (
      <div className={`flex items-center justify-center bg-gray-100 rounded-lg ${className}`}>
        <div className="text-center text-gray-500">
          <p className="text-sm">Heatmap not available</p>
          <p className="text-xs mt-1">Run pre-computation first</p>
        </div>
      </div>
    );
  }

  return (
    <div className={`relative group ${className}`}>
      {/* Original image (visible when overlay is off) */}
      <img
        src={showOverlay ? overlayUrl : originalUrl}
        alt={`Attention for ${imageId}`}
        className="w-full h-auto rounded-lg"
        onError={() => setImageError(true)}
      />

      {/* Toggle overlay button */}
      <button
        onClick={() => setShowOverlay(!showOverlay)}
        className="absolute top-2 right-2 px-2 py-1 text-xs bg-black/50 text-white rounded opacity-0 group-hover:opacity-100 transition-opacity"
      >
        {showOverlay ? 'Hide Overlay' : 'Show Overlay'}
      </button>

      {/* Info badge */}
      <div className="absolute bottom-2 left-2 px-2 py-1 text-xs bg-black/50 text-white rounded">
        {model} / Layer {layer}
      </div>
    </div>
  );
}
