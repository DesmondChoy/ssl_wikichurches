/**
 * Global view settings store using Zustand.
 */

import { create } from 'zustand';
import type { ViewSettings } from '../types';

interface ViewStore extends ViewSettings {
  // Selection state for bbox similarity
  selectedBboxIndex: number | null;

  // Available methods per model (populated from API)
  availableMethods: Record<string, string[]>;
  defaultMethods: Record<string, string>;

  // Actions
  setModel: (model: string) => void;
  setLayer: (layer: number) => void;
  setMethod: (method: string) => void;
  setPercentile: (percentile: number) => void;
  setShowBboxes: (show: boolean) => void;
  setHeatmapOpacity: (opacity: number) => void;
  setSelectedBboxIndex: (index: number | null) => void;
  setMethodsConfig: (methods: Record<string, string[]>, defaults: Record<string, string>) => void;
  reset: () => void;
}

const DEFAULT_SETTINGS: ViewSettings = {
  model: 'dinov2',
  layer: 11,
  method: 'cls',
  percentile: 90,
  showBboxes: false,
  heatmapOpacity: 0.5,
};

export const useViewStore = create<ViewStore>((set, get) => ({
  ...DEFAULT_SETTINGS,
  selectedBboxIndex: null,
  availableMethods: {},
  defaultMethods: {},

  setModel: (model) => {
    const { defaultMethods } = get();
    // Reset method to default for new model
    const newMethod = defaultMethods[model] || 'cls';
    set({ model, method: newMethod, selectedBboxIndex: null });
  },
  setLayer: (layer) => set({ layer }), // Keep selection on layer change to compare
  setMethod: (method) => set({ method }),
  setPercentile: (percentile) => set({ percentile }),
  setShowBboxes: (showBboxes) => set({ showBboxes }),
  setHeatmapOpacity: (heatmapOpacity) => set({ heatmapOpacity }),
  setSelectedBboxIndex: (selectedBboxIndex) => set({ selectedBboxIndex }),
  setMethodsConfig: (availableMethods, defaultMethods) => {
    const { model, method } = get();
    // Update method if current one isn't available for current model
    const available = availableMethods[model] || [];
    const newMethod = available.includes(method) ? method : (defaultMethods[model] || 'cls');
    set({ availableMethods, defaultMethods, method: newMethod });
  },
  reset: () => set({ ...DEFAULT_SETTINGS, selectedBboxIndex: null }),
}));

// Selector hooks for specific values
export const useModel = () => useViewStore((state) => state.model);
export const useLayer = () => useViewStore((state) => state.layer);
export const useMethod = () => useViewStore((state) => state.method);
export const usePercentile = () => useViewStore((state) => state.percentile);
export const useShowBboxes = () => useViewStore((state) => state.showBboxes);
export const useHeatmapOpacity = () => useViewStore((state) => state.heatmapOpacity);
export const useSelectedBboxIndex = () => useViewStore((state) => state.selectedBboxIndex);
export const useAvailableMethods = () => useViewStore((state) => state.availableMethods);
