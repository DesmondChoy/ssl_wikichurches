/**
 * Global view settings store using Zustand.
 */

import { create } from 'zustand';
import type { ViewSettings } from '../types';

interface ViewStore extends ViewSettings {
  // Actions
  setModel: (model: string) => void;
  setLayer: (layer: number) => void;
  setPercentile: (percentile: number) => void;
  setShowBboxes: (show: boolean) => void;
  reset: () => void;
}

const DEFAULT_SETTINGS: ViewSettings = {
  model: 'dinov2',
  layer: 11,
  percentile: 90,
  showBboxes: false,
};

export const useViewStore = create<ViewStore>((set) => ({
  ...DEFAULT_SETTINGS,

  setModel: (model) => set({ model }),
  setLayer: (layer) => set({ layer }),
  setPercentile: (percentile) => set({ percentile }),
  setShowBboxes: (showBboxes) => set({ showBboxes }),
  reset: () => set(DEFAULT_SETTINGS),
}));

// Selector hooks for specific values
export const useModel = () => useViewStore((state) => state.model);
export const useLayer = () => useViewStore((state) => state.layer);
export const usePercentile = () => useViewStore((state) => state.percentile);
export const useShowBboxes = () => useViewStore((state) => state.showBboxes);
