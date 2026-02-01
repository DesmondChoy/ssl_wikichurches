/**
 * Main attention heatmap viewer with overlay controls and bbox similarity.
 */

import { useState, useEffect, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { attentionAPI, imagesAPI } from '../../api/client';
import { InteractiveBboxOverlay } from './InteractiveBboxOverlay';
import { renderHeatmap, renderHeatmapLegend, computeSimilarityStats } from '../../utils/renderHeatmap';
import { useHeatmapOpacity } from '../../store/viewStore';
import type { BoundingBox } from '../../types';

interface AttentionViewerProps {
  imageId: string;
  model: string;
  layer: number;
  showBboxes: boolean;
  bboxes?: BoundingBox[];
  selectedBboxIndex: number | null;
  onBboxSelect: (index: number | null) => void;
  className?: string;
}

export function AttentionViewer({
  imageId,
  model,
  layer,
  showBboxes,
  bboxes = [],
  selectedBboxIndex,
  onBboxSelect,
  className = '',
}: AttentionViewerProps) {
  const [showOverlay, setShowOverlay] = useState(true);
  const [imageError, setImageError] = useState(false);

  // Get heatmap settings from store
  const heatmapOpacity = useHeatmapOpacity();

  const overlayUrl = attentionAPI.getOverlayUrl(imageId, model, layer, showBboxes);
  const originalUrl = imagesAPI.getImageUrl(imageId, 224);

  // Get selected bbox
  const selectedBbox = selectedBboxIndex !== null ? bboxes[selectedBboxIndex] : null;

  // Fetch similarity when a bbox is selected
  const { data: similarityData, isLoading: similarityLoading } = useQuery({
    queryKey: ['similarity', imageId, model, layer, selectedBbox],
    queryFn: () => {
      if (!selectedBbox) return null;
      return attentionAPI.getSimilarity(
        imageId,
        {
          left: selectedBbox.left,
          top: selectedBbox.top,
          width: selectedBbox.width,
          height: selectedBbox.height,
          label: selectedBbox.label_name || undefined,
        },
        model,
        layer
      );
    },
    enabled: !!selectedBbox,
    staleTime: 60000, // Cache for 1 minute
  });

  // Render heatmap when similarity data is available
  const heatmapUrl = useMemo(() => {
    if (!similarityData) return null;
    try {
      return renderHeatmap({
        similarity: similarityData.similarity,
        patchGrid: similarityData.patch_grid as [number, number],
        opacity: heatmapOpacity,
      });
    } catch {
      return null;
    }
  }, [similarityData, heatmapOpacity]);

  // Compute stats for display
  const stats = useMemo(() => {
    if (!similarityData) return null;
    return computeSimilarityStats(similarityData.similarity);
  }, [similarityData]);

  // Generate legend URL (static, computed once)
  const legendUrl = useMemo(() => renderHeatmapLegend(120, 12), []);

  // Reset selection when image changes
  useEffect(() => {
    onBboxSelect(null);
  }, [imageId, onBboxSelect]);

  const handleBboxClick = (_bbox: BoundingBox, index: number) => {
    // Toggle selection if clicking the same bbox
    if (selectedBboxIndex === index) {
      onBboxSelect(null);
    } else {
      onBboxSelect(index);
    }
  };

  const handleImageClick = () => {
    // Deselect when clicking outside bboxes
    if (selectedBboxIndex !== null) {
      onBboxSelect(null);
    }
  };

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

  // Determine which image to show
  const showSimilarityHeatmap = selectedBbox && heatmapUrl && !similarityLoading;
  const baseImageUrl = showOverlay ? overlayUrl : originalUrl;

  return (
    <div className={`relative group ${className}`}>
      {/* Base image (attention overlay or original) */}
      <img
        src={baseImageUrl}
        alt={`Attention for ${imageId}`}
        className="w-full h-auto rounded-lg"
        onError={() => setImageError(true)}
        onClick={handleImageClick}
      />

      {/* Similarity heatmap overlay */}
      {showSimilarityHeatmap && (
        <img
          src={heatmapUrl}
          alt="Similarity heatmap"
          className="absolute inset-0 w-full h-full rounded-lg pointer-events-none"
          style={{ mixBlendMode: 'normal' }}
        />
      )}

      {/* Loading spinner for similarity computation */}
      {similarityLoading && selectedBbox && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/20 rounded-lg">
          <div className="animate-spin rounded-full h-8 w-8 border-2 border-white border-t-transparent" />
        </div>
      )}

      {/* Interactive bbox overlay */}
      {showBboxes && bboxes.length > 0 && (
        <InteractiveBboxOverlay
          bboxes={bboxes}
          selectedIndex={selectedBboxIndex}
          onBboxClick={handleBboxClick}
        />
      )}

      {/* Toggle overlay button */}
      <button
        onClick={() => setShowOverlay(!showOverlay)}
        className="absolute top-2 right-2 px-2 py-1 text-xs bg-black/50 text-white rounded opacity-0 group-hover:opacity-100 transition-opacity"
      >
        {showOverlay ? 'Hide Attention' : 'Show Attention'}
      </button>

      {/* Clear selection button (when bbox is selected) */}
      {selectedBbox && (
        <button
          onClick={() => onBboxSelect(null)}
          className="absolute top-2 right-24 px-2 py-1 text-xs bg-green-600/80 text-white rounded opacity-0 group-hover:opacity-100 transition-opacity"
        >
          Clear Selection
        </button>
      )}

      {/* Info badge */}
      <div className="absolute bottom-2 left-2 px-2 py-1 text-xs bg-black/50 text-white rounded">
        {model} / Layer {layer}
        {selectedBbox && (
          <span className="ml-2 text-green-300">
            {selectedBbox.label_name || `Feature ${selectedBbox.label}`}
          </span>
        )}
      </div>

      {/* Similarity stats badge */}
      {stats && selectedBbox && (
        <div className="absolute bottom-2 right-2 px-2 py-1 text-xs bg-black/50 text-white rounded">
          Sim: {stats.min.toFixed(2)} - {stats.max.toFixed(2)}
        </div>
      )}

      {/* Colormap legend */}
      {stats && selectedBbox && (
        <div className="absolute bottom-8 right-2 flex items-center gap-1 px-2 py-1 bg-black/50 rounded text-xs text-white">
          <span>0</span>
          <img src={legendUrl} alt="Similarity scale" className="h-3 rounded-sm" />
          <span>1</span>
        </div>
      )}
    </div>
  );
}
