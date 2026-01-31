/**
 * IoU metrics display component.
 */

import type { IoUResult } from '../../types';

interface IoUDisplayProps {
  metrics: IoUResult | null | undefined;
  isLoading?: boolean;
  compact?: boolean;
}

export function IoUDisplay({ metrics, isLoading, compact = false }: IoUDisplayProps) {
  if (isLoading) {
    return (
      <div className="animate-pulse space-y-2">
        <div className="h-4 bg-gray-200 rounded w-20" />
        <div className="h-4 bg-gray-200 rounded w-24" />
      </div>
    );
  }

  if (!metrics) {
    return (
      <div className="text-sm text-gray-500">
        Metrics not available
      </div>
    );
  }

  if (compact) {
    return (
      <div className="flex gap-4 text-sm">
        <div>
          <span className="text-gray-500">IoU:</span>{' '}
          <span className="font-semibold">{metrics.iou.toFixed(3)}</span>
        </div>
        <div>
          <span className="text-gray-500">Coverage:</span>{' '}
          <span className="font-semibold">{(metrics.coverage * 100).toFixed(1)}%</span>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 gap-4">
        <MetricCard
          label="IoU Score"
          value={metrics.iou.toFixed(3)}
          description="Intersection over Union"
          color={getIoUColor(metrics.iou)}
        />
        <MetricCard
          label="Coverage"
          value={`${(metrics.coverage * 100).toFixed(1)}%`}
          description="Attention in annotated regions"
          color={getIoUColor(metrics.coverage)}
        />
      </div>

      <div className="grid grid-cols-2 gap-4 text-sm">
        <div>
          <span className="text-gray-500">Attention Area:</span>{' '}
          <span>{(metrics.attention_area * 100).toFixed(1)}%</span>
        </div>
        <div>
          <span className="text-gray-500">Annotation Area:</span>{' '}
          <span>{(metrics.annotation_area * 100).toFixed(1)}%</span>
        </div>
      </div>
    </div>
  );
}

interface MetricCardProps {
  label: string;
  value: string;
  description?: string;
  color: string;
}

function MetricCard({ label, value, description, color }: MetricCardProps) {
  return (
    <div className={`p-3 rounded-lg ${color}`}>
      <div className="text-xs text-gray-600">{label}</div>
      <div className="text-2xl font-bold">{value}</div>
      {description && (
        <div className="text-xs text-gray-500 mt-1">{description}</div>
      )}
    </div>
  );
}

function getIoUColor(value: number): string {
  if (value >= 0.6) return 'bg-green-100';
  if (value >= 0.4) return 'bg-yellow-100';
  if (value >= 0.2) return 'bg-orange-100';
  return 'bg-red-100';
}
