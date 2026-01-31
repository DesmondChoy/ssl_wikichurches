/**
 * Model leaderboard component showing rankings.
 */

import { useLeaderboard } from '../../hooks/useMetrics';
import { Card, CardHeader, CardContent } from '../ui/Card';

interface ModelLeaderboardProps {
  percentile: number;
  onModelSelect?: (model: string) => void;
}

export function ModelLeaderboard({ percentile, onModelSelect }: ModelLeaderboardProps) {
  const { data: leaderboard, isLoading, error } = useLeaderboard(percentile);

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

  if (error) {
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
          <span className="text-xs text-gray-500">Top {100 - percentile}% threshold</span>
        </div>
      </CardHeader>
      <CardContent className="p-0">
        <div className="divide-y">
          {leaderboard?.map((entry) => (
            <div
              key={entry.model}
              className={`flex items-center gap-3 px-4 py-3 ${
                onModelSelect ? 'hover:bg-gray-50 cursor-pointer' : ''
              }`}
              onClick={() => onModelSelect?.(entry.model)}
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
                <div className="text-xs text-gray-500">
                  Best: {entry.best_layer}
                </div>
              </div>

              {/* IoU score */}
              <div className="text-right">
                <div className="text-lg font-bold text-primary-600">
                  {entry.best_iou.toFixed(3)}
                </div>
                <div className="text-xs text-gray-500">IoU</div>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
