import { Card, CardContent, CardHeader } from '../ui/Card';
import type { ImageAnnotation } from '../../types';

interface AnnotationsCardProps {
  annotation: ImageAnnotation;
  showBboxes: boolean;
  selectedBboxIndex: number | null;
  onBboxSelect: (index: number | null) => void;
}

export function AnnotationsCard({
  annotation,
  showBboxes,
  selectedBboxIndex,
  onBboxSelect,
}: AnnotationsCardProps) {
  return (
    <div data-testid="annotations-card">
      <Card>
        <CardHeader>
          <h3 className="font-semibold">Annotations</h3>
        </CardHeader>
        <CardContent className="space-y-3">
          <div>
            <span className="text-sm text-gray-500">Styles:</span>
            <div className="mt-1 flex flex-wrap gap-1">
              {annotation.style_names.map((style) => (
                <span
                  key={style}
                  className="rounded bg-primary-100 px-2 py-0.5 text-sm text-primary-700"
                >
                  {style}
                </span>
              ))}
            </div>
          </div>

          <div>
            <span className="text-sm text-gray-500">Bounding Boxes:</span>
            <span className="ml-2 font-medium">{annotation.num_bboxes}</span>
          </div>

          {showBboxes && (
            <div className="rounded bg-green-50 px-2 py-1 text-xs text-green-700">
              Click a bounding box to see feature similarity heatmap
            </div>
          )}

          <div className="max-h-48 space-y-1 overflow-y-auto text-xs text-gray-500">
            {annotation.bboxes.map((bbox, index) => {
              const isSelected = selectedBboxIndex === index;
              return (
                <button
                  key={`${bbox.label}-${index}`}
                  type="button"
                  data-testid={`bbox-list-item-${index}`}
                  className={`flex w-full items-center justify-between rounded px-2 py-1 text-left transition-colors ${
                    showBboxes ? 'hover:bg-gray-100' : 'cursor-default'
                  } ${isSelected ? 'bg-green-100 text-green-700' : ''}`}
                  onClick={() => {
                    if (!showBboxes) return;
                    onBboxSelect(isSelected ? null : index);
                  }}
                >
                  <span>{bbox.label_name || `Label ${bbox.label}`}</span>
                  <span className="text-gray-400">
                    {(bbox.width * 100).toFixed(0)}% x {(bbox.height * 100).toFixed(0)}%
                  </span>
                </button>
              );
            })}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
