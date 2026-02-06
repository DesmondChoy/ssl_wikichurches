/**
 * Tooltip component for displaying help information on hover.
 *
 * Uses a React portal so the popup escapes ancestor overflow:hidden
 * containers (e.g. Card).
 */

import { useRef, useState } from 'react';
import { createPortal } from 'react-dom';

interface TooltipProps {
  content: string;
  /** Horizontal alignment of the popup relative to the "?" button. */
  align?: 'center' | 'left' | 'right';
}

export function Tooltip({ content, align = 'center' }: TooltipProps) {
  const [isVisible, setIsVisible] = useState(false);
  const buttonRef = useRef<HTMLButtonElement>(null);

  const popup = () => {
    if (!isVisible || !buttonRef.current) return null;

    const rect = buttonRef.current.getBoundingClientRect();

    // Horizontal position based on align prop
    let left: number;
    if (align === 'left') {
      left = rect.left;
    } else if (align === 'right') {
      left = rect.right - 256; // w-64 = 256px
    } else {
      left = rect.left + rect.width / 2 - 128; // center
    }

    // Arrow offset (distance from the aligned edge)
    let arrowLeft: number;
    if (align === 'left') {
      arrowLeft = rect.width / 2;
    } else if (align === 'right') {
      arrowLeft = 256 - rect.width / 2;
    } else {
      arrowLeft = 128; // center
    }

    return createPortal(
      <div
        className="fixed z-50 w-64 p-2 text-xs text-white bg-gray-900 rounded-lg shadow-lg pointer-events-none"
        style={{
          left,
          top: rect.top - 8, // 8px gap (mb-2)
          transform: 'translateY(-100%)',
        }}
      >
        {content}
        <div
          className="absolute border-4 border-transparent border-t-gray-900"
          style={{ top: '100%', left: arrowLeft, transform: 'translateX(-50%)' }}
        />
      </div>,
      document.body,
    );
  };

  return (
    <span className="relative inline-flex items-center">
      <button
        ref={buttonRef}
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
      {popup()}
    </span>
  );
}
