/**
 * Model leaderboard component showing rankings.
 */

import { Card, CardHeader, CardContent } from '../ui/Card';
import type { DashboardMetric, LeaderboardEntry } from '../../types';
import { DASHBOARD_METRIC_METADATA } from '../../constants/metricMetadata';
import { getAttentionMethodLabel } from '../../constants/attentionMethods';

interface ModelLeaderboardProps {
  leaderboard?: LeaderboardEntry[];
  percentile: number;
  metric: DashboardMetric;
  isLoading?: boolean;
  hasError?: boolean;
  emptyMessage?: string;
  onModelSelect?: (entry: LeaderboardEntry) => void;
}

export function ModelLeaderboard({
  leaderboard,
  percentile,
  metric,
  isLoading = false,
  hasError = false,
  emptyMessage = 'No compatible models available for this method.',
  onModelSelect,
}: ModelLeaderboardProps) {
  const metricMetadata = DASHBOARD_METRIC_METADATA[metric];
  const metricLabel = metricMetadata.shortLabel;
  const metricHint = metricMetadata.hint(percentile);

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <h3 className="font-semibold">Model Leaderboard</h3>
        </CardHeader>
        <CardContent>
          <div className="animate-pulse space-y-3">
            {[1, 2, 3].map((i) => (
              <div key={i} className="h-12 bg-gray-200 rounded" />
            ))}
          </div>
        </CardContent>
      </Card>
    );
  }

  if (hasError) {
    return (
      <Card>
        <CardHeader>
          <h3 className="font-semibold">Model Leaderboard</h3>
        </CardHeader>
        <CardContent>
          <div className="text-red-500 text-sm">Failed to load leaderboard</div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex justify-between items-center">
          <h3 className="font-semibold">Model Leaderboard</h3>
          <span className="text-xs text-gray-500">{metricHint}</span>
        </div>
      </CardHeader>
      <CardContent className="p-0">
        {!leaderboard?.length ? (
          <div className="px-4 py-6 text-sm text-gray-500">
            {emptyMessage}
          </div>
        ) : (
          <div className="divide-y">
            {leaderboard.map((entry) => (
              <div
                key={entry.model}
                data-testid={`leaderboard-row-${entry.model}`}
                className={`flex items-center gap-3 px-4 py-3 ${
                  onModelSelect ? 'hover:bg-gray-50 cursor-pointer' : ''
                }`}
                onClick={() => onModelSelect?.(entry)}
              >
                {/* Rank badge */}
                <div
                  className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold ${
                    entry.rank === 1
                      ? 'bg-yellow-100 text-yellow-700'
                      : entry.rank === 2
                      ? 'bg-gray-100 text-gray-700'
                      : entry.rank === 3
                      ? 'bg-orange-100 text-orange-700'
                      : 'bg-gray-50 text-gray-500'
                  }`}
                >
                  #{entry.rank}
                </div>

                {/* Model info */}
                <div className="flex-1">
                  <div className="font-medium capitalize">{entry.model}</div>
                  <div className="text-xs text-gray-500" data-testid={`leaderboard-row-meta-${entry.model}`}>
                    Best: {entry.best_layer} • {getAttentionMethodLabel(entry.method_used)}
                  </div>
                </div>

                {/* Metric score */}
                <div className="text-right">
                  <div className="text-lg font-bold text-primary-600">
                    {entry.score.toFixed(3)}
                  </div>
                  <div className="text-xs text-gray-500">{metricLabel}</div>
                </div>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
