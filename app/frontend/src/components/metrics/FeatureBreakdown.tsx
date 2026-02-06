/**
 * Feature breakdown component showing IoU metrics for architectural feature types.
 */

import { useState, useMemo } from 'react';
import { useFeatureBreakdown } from '../../hooks/useMetrics';
import { Card, CardHeader, CardContent } from '../ui/Card';
import type { FeatureIoUEntry } from '../../types';

interface FeatureBreakdownProps {
  model: string;
  layer: number;
  percentile: number;
  method?: string;
}

type SortField = 'mean_iou' | 'bbox_count' | 'feature_name' | 'feature_label';

const ITEMS_PER_PAGE = 20;

/**
 * Returns a color class based on IoU value.
 * Green >= 0.6, Yellow >= 0.4, Orange >= 0.2, Red < 0.2
 */
function getIoUColorClass(iou: number): string {
  if (iou >= 0.6) return 'text-green-600 bg-green-50';
  if (iou >= 0.4) return 'text-yellow-600 bg-yellow-50';
  if (iou >= 0.2) return 'text-orange-600 bg-orange-50';
  return 'text-red-600 bg-red-50';
}

export function FeatureBreakdown({ model, layer, percentile, method }: FeatureBreakdownProps) {
  const [sortBy, setSortBy] = useState<SortField>('mean_iou');
  const [searchQuery, setSearchQuery] = useState('');
  const [showCount, setShowCount] = useState(ITEMS_PER_PAGE);

  const { data, isLoading, error } = useFeatureBreakdown(model, layer, percentile, sortBy, 0, method);

  // Filter features by search query
  const filteredFeatures = useMemo(() => {
    if (!data?.features) return [];
    if (!searchQuery.trim()) return data.features;

    const query = searchQuery.toLowerCase();
    return data.features.filter(f =>
      f.feature_name.toLowerCase().includes(query)
    );
  }, [data?.features, searchQuery]);

  // Pagination
  const visibleFeatures = filteredFeatures.slice(0, showCount);
  const hasMore = showCount < filteredFeatures.length;

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <h3 className="font-semibold">Feature Type Breakdown</h3>
        </CardHeader>
        <CardContent>
          <div className="animate-pulse space-y-2">
            {[1, 2, 3, 4, 5].map((i) => (
              <div key={i} className="h-10 bg-gray-200 rounded" />
            ))}
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardHeader>
          <h3 className="font-semibold">Feature Type Breakdown</h3>
        </CardHeader>
        <CardContent>
          <div className="text-red-500 text-sm">Failed to load feature breakdown</div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex justify-between items-center">
          <h3 className="font-semibold">Feature Type Breakdown</h3>
          <span className="text-xs text-gray-500 capitalize">{model} • Layer {layer}</span>
        </div>
      </CardHeader>
      <CardContent className="p-0">
        {/* Controls */}
        <div className="px-4 py-3 border-b border-gray-100 space-y-3">
          {/* Search */}
          <input
            type="text"
            placeholder="Search features..."
            value={searchQuery}
            onChange={(e) => {
              setSearchQuery(e.target.value);
              setShowCount(ITEMS_PER_PAGE); // Reset pagination on search
            }}
            className="w-full px-3 py-2 text-sm border border-gray-200 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
          />

          {/* Sort controls */}
          <div className="flex gap-2 text-xs">
            <span className="text-gray-500 py-1">Sort by:</span>
            {(['mean_iou', 'bbox_count', 'feature_name'] as SortField[]).map((field) => (
              <button
                key={field}
                onClick={() => {
                  setSortBy(field);
                  setShowCount(ITEMS_PER_PAGE);
                }}
                className={`px-2 py-1 rounded ${
                  sortBy === field
                    ? 'bg-primary-100 text-primary-700 font-medium'
                    : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                }`}
              >
                {field === 'mean_iou' ? 'IoU' : field === 'bbox_count' ? 'Count' : 'Name'}
              </button>
            ))}
          </div>
        </div>

        {/* Results summary */}
        <div className="px-4 py-2 bg-gray-50 text-xs text-gray-500 border-b border-gray-100">
          Showing {visibleFeatures.length} of {filteredFeatures.length} feature types
          {searchQuery && ` matching "${searchQuery}"`}
        </div>

        {/* Feature list */}
        <div className="max-h-[400px] overflow-y-auto">
          {visibleFeatures.length === 0 ? (
            <div className="px-4 py-8 text-center text-gray-500 text-sm">
              {searchQuery ? 'No features match your search' : 'No feature data available'}
            </div>
          ) : (
            <table className="w-full text-sm">
              <thead className="sticky top-0 bg-white border-b border-gray-200">
                <tr className="text-left text-gray-500 text-xs">
                  <th className="px-4 py-2 font-medium">Feature</th>
                  <th className="px-4 py-2 font-medium text-right">IoU</th>
                  <th className="px-4 py-2 font-medium text-right">Count</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {visibleFeatures.map((feature: FeatureIoUEntry) => (
                  <tr key={feature.feature_label} className="hover:bg-gray-50">
                    <td className="px-4 py-2">
                      <div className="font-medium text-gray-900">{feature.feature_name}</div>
                      <div className="text-xs text-gray-400">ID: {feature.feature_label}</div>
                    </td>
                    <td className="px-4 py-2 text-right">
                      <span className={`inline-block px-2 py-0.5 rounded text-xs font-medium ${getIoUColorClass(feature.mean_iou)}`}>
                        {feature.mean_iou.toFixed(3)}
                      </span>
                      <div className="text-xs text-gray-400">±{feature.std_iou.toFixed(3)}</div>
                    </td>
                    <td className="px-4 py-2 text-right text-gray-600">
                      {feature.bbox_count}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>

        {/* Show more button */}
        {hasMore && (
          <div className="px-4 py-3 border-t border-gray-100 text-center">
            <button
              onClick={() => setShowCount(prev => prev + ITEMS_PER_PAGE)}
              className="text-sm text-primary-600 hover:text-primary-700 font-medium"
            >
              Show more ({filteredFeatures.length - showCount} remaining)
            </button>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
