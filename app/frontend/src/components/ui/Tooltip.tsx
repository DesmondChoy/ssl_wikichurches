/**
 * Tooltip component for displaying help information on hover.
 */

import { useState } from 'react';

interface TooltipProps {
  content: string;
  /** Horizontal alignment of the popup relative to the "?" button. */
  align?: 'center' | 'left' | 'right';
}

const popupAlign = {
  center: 'left-1/2 -translate-x-1/2',
  left: 'left-0',
  right: 'right-0',
} as const;

const arrowAlign = {
  center: 'left-1/2 -translate-x-1/2',
  left: 'left-2',
  right: 'right-2',
} as const;

export function Tooltip({ content, align = 'center' }: TooltipProps) {
  const [isVisible, setIsVisible] = useState(false);

  return (
    <span className="relative inline-flex items-center">
      <button
        type="button"
        className="ml-1 inline-flex items-center justify-center w-4 h-4 text-xs text-gray-500 bg-gray-200 rounded-full hover:bg-gray-300 focus:outline-none focus:ring-2 focus:ring-offset-1 focus:ring-primary-500"
        onMouseEnter={() => setIsVisible(true)}
        onMouseLeave={() => setIsVisible(false)}
        onFocus={() => setIsVisible(true)}
        onBlur={() => setIsVisible(false)}
        aria-label="Help"
      >
        ?
      </button>
      {isVisible && (
        <div className={`absolute z-50 w-64 p-2 text-xs text-white bg-gray-900 rounded-lg bottom-full ${popupAlign[align]} mb-2 shadow-lg`}>
          {content}
          <div className={`absolute top-full ${arrowAlign[align]} border-4 border-transparent border-t-gray-900`} />
        </div>
      )}
    </span>
  );
}
