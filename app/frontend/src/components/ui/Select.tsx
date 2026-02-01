/**
 * Select dropdown component.
 */

import { Tooltip } from './Tooltip';

interface SelectProps {
  value: string | number;
  onChange: (value: string) => void;
  options: Array<{ value: string | number; label: string }>;
  label?: string;
  tooltip?: string;
  className?: string;
}

export function Select({ value, onChange, options, label, tooltip, className = '' }: SelectProps) {
  return (
    <div className={`flex flex-col gap-1 ${className}`}>
      {label && (
        <div className="flex items-center">
          <label className="text-sm font-medium text-gray-700">{label}</label>
          {tooltip && <Tooltip content={tooltip} />}
        </div>
      )}
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="px-3 py-2 bg-white border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
      >
        {options.map((opt) => (
          <option key={opt.value} value={opt.value}>
            {opt.label}
          </option>
        ))}
      </select>
    </div>
  );
}
