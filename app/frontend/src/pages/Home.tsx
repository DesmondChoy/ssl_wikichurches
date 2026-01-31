/**
 * Home page with image grid browser.
 */

import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { imagesAPI } from '../api/client';
import { Card } from '../components/ui/Card';
import { Select } from '../components/ui/Select';
import type { ImageListItem } from '../types';

export function HomePage() {
  const navigate = useNavigate();
  const [styleFilter, setStyleFilter] = useState<string>('');

  // Fetch styles
  const { data: styles } = useQuery({
    queryKey: ['styles'],
    queryFn: () => imagesAPI.getStyles(),
  });

  // Fetch images
  const { data: images, isLoading, error } = useQuery({
    queryKey: ['images', styleFilter],
    queryFn: () => imagesAPI.list({ style: styleFilter || undefined }),
  });

  const styleOptions = [
    { value: '', label: 'All Styles' },
    ...(styles?.map((s) => ({ value: s, label: s })) || []),
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">WikiChurches Attention Analysis</h1>
          <p className="text-gray-600 mt-1">
            {images?.length || 0} annotated images with {images?.reduce((sum, img) => sum + img.num_bboxes, 0) || 0} bounding boxes
          </p>
        </div>

        <Select
          value={styleFilter}
          onChange={setStyleFilter}
          options={styleOptions}
          className="w-48"
        />
      </div>

      {/* Loading state */}
      {isLoading && (
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
          {Array.from({ length: 10 }).map((_, i) => (
            <div key={i} className="animate-pulse">
              <div className="aspect-square bg-gray-200 rounded-lg" />
              <div className="h-4 bg-gray-200 rounded mt-2 w-3/4" />
            </div>
          ))}
        </div>
      )}

      {/* Error state */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-700">
          Failed to load images. Make sure the backend is running.
        </div>
      )}

      {/* Image grid */}
      {images && (
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
          {images.map((image) => (
            <ImageCard
              key={image.image_id}
              image={image}
              onClick={() => navigate(`/image/${encodeURIComponent(image.image_id)}`)}
            />
          ))}
        </div>
      )}
    </div>
  );
}

interface ImageCardProps {
  image: ImageListItem;
  onClick: () => void;
}

function ImageCard({ image, onClick }: ImageCardProps) {
  const [imgError, setImgError] = useState(false);

  return (
    <Card hoverable onClick={onClick}>
      <div className="aspect-square relative bg-gray-100">
        {!imgError ? (
          <img
            src={imagesAPI.getThumbnailUrl(image.image_id)}
            alt={image.image_id}
            className="w-full h-full object-cover"
            onError={() => setImgError(true)}
            loading="lazy"
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center text-gray-400">
            No image
          </div>
        )}

        {/* Bbox count badge */}
        <div className="absolute top-2 right-2 px-2 py-0.5 bg-black/50 text-white text-xs rounded">
          {image.num_bboxes} boxes
        </div>
      </div>

      <div className="p-2">
        <div className="text-xs text-gray-500 truncate" title={image.image_id}>
          {image.image_id}
        </div>
        <div className="flex gap-1 mt-1 flex-wrap">
          {image.style_names.map((style) => (
            <span
              key={style}
              className="px-1.5 py-0.5 bg-primary-100 text-primary-700 text-xs rounded"
            >
              {style}
            </span>
          ))}
        </div>
      </div>
    </Card>
  );
}
