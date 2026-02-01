/**
 * Global view settings store using Zustand.
 */

import { create } from 'zustand';
import type { ViewSettings, HeatmapShape } from '../types';

interface ViewStore extends ViewSettings {
  // Selection state for bbox similarity
  selectedBboxIndex: number | null;

  // Actions
  setModel: (model: string) => void;
  setLayer: (layer: number) => void;
  setPercentile: (percentile: number) => void;
  setShowBboxes: (show: boolean) => void;
  setHeatmapOpacity: (opacity: number) => void;
  setHeatmapShape: (shape: HeatmapShape) => void;
  setSelectedBboxIndex: (index: number | null) => void;
  reset: () => void;
}

const DEFAULT_SETTINGS: ViewSettings = {
  model: 'dinov2',
  layer: 11,
  percentile: 90,
  showBboxes: false,
  heatmapOpacity: 0.5,
  heatmapShape: 'squares',
};

export const useViewStore = create<ViewStore>((set) => ({
  ...DEFAULT_SETTINGS,
  selectedBboxIndex: null,

  setModel: (model) => set({ model, selectedBboxIndex: null }), // Reset selection on model change
  setLayer: (layer) => set({ layer }), // Keep selection on layer change to compare
  setPercentile: (percentile) => set({ percentile }),
  setShowBboxes: (showBboxes) => set({ showBboxes }),
  setHeatmapOpacity: (heatmapOpacity) => set({ heatmapOpacity }),
  setHeatmapShape: (heatmapShape) => set({ heatmapShape }),
  setSelectedBboxIndex: (selectedBboxIndex) => set({ selectedBboxIndex }),
  reset: () => set({ ...DEFAULT_SETTINGS, selectedBboxIndex: null }),
}));

// Selector hooks for specific values
export const useModel = () => useViewStore((state) => state.model);
export const useLayer = () => useViewStore((state) => state.layer);
export const usePercentile = () => useViewStore((state) => state.percentile);
export const useShowBboxes = () => useViewStore((state) => state.showBboxes);
export const useHeatmapOpacity = () => useViewStore((state) => state.heatmapOpacity);
export const useHeatmapShape = () => useViewStore((state) => state.heatmapShape);
export const useSelectedBboxIndex = () => useViewStore((state) => state.selectedBboxIndex);
