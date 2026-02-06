/**
 * IoU metrics display component.
 */

import type { IoUResult } from '../../types';
import { Tooltip } from '../ui/Tooltip';
import type { ComponentProps } from 'react';
import { GLOSSARY } from '../../constants/glossary';

interface IoUDisplayProps {
  metrics: IoUResult | null | undefined;
  isLoading?: boolean;
  compact?: boolean;
  bboxLabel?: string | null;
}

export function IoUDisplay({ metrics, isLoading, compact = false, bboxLabel }: IoUDisplayProps) {
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
      {bboxLabel && (
        <div className="text-xs text-green-600 bg-green-50 px-2 py-1 rounded">
          Showing metrics for: <span className="font-semibold">{bboxLabel}</span>
        </div>
      )}
      <div className="grid grid-cols-2 gap-4">
        {(() => {
          const maxIoU = computeMaxIoU(metrics.attention_area, metrics.annotation_area);
          const ratio = computeIoURatio(metrics.iou, maxIoU);
          const colors = getRelativeIoUColor(ratio);
          return (
            <div className={`p-3 rounded-lg ${colors.bg}`}>
              <div className="text-xs text-gray-600">
                IoU Score
                <Tooltip content={GLOSSARY['IoU Score']} align="left" />
              </div>
              <div className="mt-1">
                <span className="text-2xl font-bold">{metrics.iou.toFixed(3)}</span>
                {maxIoU > 0 && (
                  <span className="text-sm text-gray-400 ml-1">/ {maxIoU.toFixed(3)} max</span>
                )}
              </div>
              {maxIoU > 0 && (
                <>
                  <div className="mt-2 h-2 bg-gray-200 rounded-full">
                    <div
                      className={`h-2 rounded-full transition-all duration-300 ${colors.bar}`}
                      style={{ width: `${ratio * 100}%` }}
                    />
                  </div>
                  <div className={`text-xs mt-1 ${colors.text}`}>
                    {(ratio * 100).toFixed(1)}% of theoretical max
                  </div>
                </>
              )}
            </div>
          );
        })()}
        <MetricCard
          label="Coverage"
          value={`${(metrics.coverage * 100).toFixed(1)}%`}
          description="Attention in annotated regions"
          color={getIoUColor(metrics.coverage)}
          tooltip={GLOSSARY['Coverage']}
          tooltipAlign="right"
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
  tooltip?: string;
  tooltipAlign?: ComponentProps<typeof Tooltip>['align'];
}

function MetricCard({ label, value, description, color, tooltip, tooltipAlign }: MetricCardProps) {
  return (
    <div className={`p-3 rounded-lg ${color}`}>
      <div className="text-xs text-gray-600">
        {label}
        {tooltip && <Tooltip content={tooltip} align={tooltipAlign} />}
      </div>
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

function computeMaxIoU(attentionArea: number, annotationArea: number): number {
  if (attentionArea === 0 || annotationArea === 0) return 0;
  return Math.min(attentionArea, annotationArea) / Math.max(attentionArea, annotationArea);
}

function computeIoURatio(iou: number, maxIoU: number): number {
  if (maxIoU === 0) return 0;
  return Math.min(iou / maxIoU, 1); // clamp for float imprecision
}

function getRelativeIoUColor(ratio: number): { bg: string; bar: string; text: string } {
  if (ratio >= 0.75) return { bg: 'bg-green-50',  bar: 'bg-green-500',  text: 'text-green-700' };
  if (ratio >= 0.50) return { bg: 'bg-yellow-50', bar: 'bg-yellow-500', text: 'text-yellow-700' };
  if (ratio >= 0.25) return { bg: 'bg-orange-50', bar: 'bg-orange-500', text: 'text-orange-700' };
  return                    { bg: 'bg-red-50',    bar: 'bg-red-500',    text: 'text-red-700' };
}
