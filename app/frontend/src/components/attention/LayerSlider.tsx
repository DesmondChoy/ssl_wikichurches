/**
 * Layer progression slider with animation support.
 */

import { useState, useEffect, useRef } from 'react';
import { Slider } from '../ui/Slider';

interface LayerSliderProps {
  currentLayer: number;
  maxLayers?: number;
  onChange: (layer: number) => void;
  autoPlay?: boolean;
  playSpeed?: number; // ms between frames
}

export function LayerSlider({
  currentLayer,
  maxLayers = 12,
  onChange,
  autoPlay = false,
  playSpeed = 500,
}: LayerSliderProps) {
  const [isPlaying, setIsPlaying] = useState(autoPlay);
  const intervalRef = useRef<number | null>(null);

  useEffect(() => {
    if (isPlaying) {
      intervalRef.current = window.setInterval(() => {
        onChange((currentLayer + 1) % maxLayers);
      }, playSpeed);
    } else {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [isPlaying, currentLayer, maxLayers, playSpeed, onChange]);

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center gap-4">
        <button
          onClick={() => setIsPlaying(!isPlaying)}
          className="px-3 py-1 text-sm font-medium bg-primary-600 text-white rounded hover:bg-primary-700 transition-colors"
        >
          {isPlaying ? 'Pause' : 'Play'}
        </button>

        <button
          onClick={() => onChange(0)}
          className="px-2 py-1 text-sm text-gray-600 hover:text-gray-900"
          disabled={isPlaying}
        >
          |&lt;
        </button>

        <button
          onClick={() => onChange(Math.max(0, currentLayer - 1))}
          className="px-2 py-1 text-sm text-gray-600 hover:text-gray-900"
          disabled={isPlaying}
        >
          &lt;
        </button>

        <button
          onClick={() => onChange(Math.min(maxLayers - 1, currentLayer + 1))}
          className="px-2 py-1 text-sm text-gray-600 hover:text-gray-900"
          disabled={isPlaying}
        >
          &gt;
        </button>

        <button
          onClick={() => onChange(maxLayers - 1)}
          className="px-2 py-1 text-sm text-gray-600 hover:text-gray-900"
          disabled={isPlaying}
        >
          &gt;|
        </button>
      </div>

      <Slider
        value={currentLayer}
        onChange={onChange}
        min={0}
        max={maxLayers - 1}
        label={`Layer ${currentLayer}`}
        showValue={false}
      />

      <div className="flex justify-between text-xs text-gray-500">
        <span>Early (L0)</span>
        <span>Late (L{maxLayers - 1})</span>
      </div>
    </div>
  );
}
