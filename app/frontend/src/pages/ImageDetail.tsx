/**
 * Image detail page with attention viewer and metrics.
 */

import { useCallback, useEffect, useState } from 'react';
import { Link, useParams, useSearchParams } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { imagesAPI } from '../api/client';
import { useViewStore } from '../store/viewStore';
import { useModels } from '../hooks/useAttention';
import { AttentionViewer } from '../components/attention/AttentionViewer';
import { ControlPanel } from '../components/attention/ControlPanel';
import { LayerSlider } from '../components/attention/LayerSlider';
import { ImageDetailMetricsPanel } from '../components/metrics/ImageDetailMetricsPanel';
import { Q3StudyScopeCallout } from '../components/metrics/Q3ScopeFraming';
import { Card, CardContent } from '../components/ui/Card';
import { ErrorBoundary } from '../components/ui/ErrorBoundary';
import { PageTabs } from '../components/ui/PageTabs';
import { AnnotationsCard } from '../components/image-detail/AnnotationsCard';
import { ImageDetailModeSwitch } from '../components/image-detail/ImageDetailModeSwitch';
import { parseImageDetailMode } from '../constants/imageDetailModes';
import { parsePageTab } from '../constants/pageTabs';
import { Q3_DEFAULTS, getQ3ModelScopeStatus } from '../constants/q3Scope';
import type { ImageDetailMode, PageTab } from '../types';

export function ImageDetailPage() {
  const { imageId } = useParams<{ imageId: string }>();
  const [searchParams, setSearchParams] = useSearchParams();
  const decodedId = imageId ? decodeURIComponent(imageId) : '';
  const [isPlaying, setIsPlaying] = useState(false);

  const {
    model,
    layer,
    method,
    head,
    imageDetailMode,
    percentile,
    showBboxes,
    selectedBboxIndex,
    setLayer,
    setHead,
    setImageDetailMode,
    setModelWithPreferredMethod,
    setPercentile,
    setSelectedBboxIndex,
  } = useViewStore();
  const requestedMode = parseImageDetailMode(searchParams.get('mode'));
  const currentTab = parsePageTab(searchParams.get('tab'));
  const currentMode = requestedMode;
  const isQ3Tab = currentTab === 'q3';
  const activeViewerMode: ImageDetailMode = isQ3Tab ? currentMode : 'head_attention';

  useEffect(() => {
    if (imageDetailMode !== requestedMode) {
      setImageDetailMode(requestedMode);
    }
  }, [imageDetailMode, requestedMode, setImageDetailMode]);

  // Get models config for per-model layer counts
  const { data: modelsData } = useModels();
  const maxLayers = modelsData?.num_layers_per_model?.[model] ?? modelsData?.num_layers ?? 12;

  const persistSearchParam = useCallback((key: string, value: string) => {
    const nextParams = new URLSearchParams(searchParams);
    nextParams.set(key, value);
    setSearchParams(nextParams);
  }, [searchParams, setSearchParams]);
  const persistModeToUrl = useCallback((nextMode: ImageDetailMode) => {
    persistSearchParam('mode', nextMode);
  }, [persistSearchParam]);
  const persistTabToUrl = useCallback((nextTab: PageTab) => {
    persistSearchParam('tab', nextTab);
  }, [persistSearchParam]);

  const handleBboxSelect = useCallback((index: number | null) => {
    setSelectedBboxIndex(index);
  }, [setSelectedBboxIndex]);
  const handleModeChange = useCallback((nextMode: ImageDetailMode) => {
    setImageDetailMode(nextMode);
    persistModeToUrl(nextMode);
  }, [persistModeToUrl, setImageDetailMode]);
  const handleTabChange = useCallback((nextTab: PageTab) => {
    persistTabToUrl(nextTab);
  }, [persistTabToUrl]);
  const handleApplyQ3Defaults = useCallback(() => {
    setIsPlaying(false);
    setImageDetailMode('head_attention');
    setModelWithPreferredMethod(Q3_DEFAULTS.model, Q3_DEFAULTS.method);
    setHead(null);
    setLayer(Q3_DEFAULTS.layer);
    setPercentile(Q3_DEFAULTS.percentile);
    setSelectedBboxIndex(null);
    persistModeToUrl('head_attention');
  }, [
    persistModeToUrl,
    setHead,
    setImageDetailMode,
    setLayer,
    setModelWithPreferredMethod,
    setPercentile,
    setSelectedBboxIndex,
  ]);

  // Fetch image details
  const { data: imageDetail, isLoading: detailLoading, error } = useQuery({
    queryKey: ['imageDetail', decodedId],
    queryFn: () => imagesAPI.getDetail(decodedId),
    enabled: !!decodedId,
  });
  const canQueryProgression = !!decodedId && !!imageDetail;

  if (!decodedId) {
    return <div>Invalid image ID</div>;
  }

  if (error) {
    return (
      <div className="space-y-4">
        <Link to="/" className="text-primary-600 hover:underline">
          &larr; Back to gallery
        </Link>
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-700">
          Failed to load image: {decodedId}
        </div>
      </div>
    );
  }

  if (!detailLoading && !imageDetail) {
    return (
      <div className="space-y-4">
        <Link to="/" className="text-primary-600 hover:underline">
          &larr; Back to gallery
        </Link>
        <div className="rounded-lg border border-amber-200 bg-amber-50 p-4 text-amber-900">
          Image not found: {decodedId}
        </div>
      </div>
    );
  }

  if (detailLoading) {
    return (
      <div className="animate-pulse space-y-4">
        <div className="h-8 bg-gray-200 rounded w-1/4" />
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-12 xl:grid-cols-[22rem_minmax(0,1fr)_minmax(0,1fr)] xl:gap-8">
          <div className="space-y-4 lg:col-span-3 xl:col-span-1">
            <div className="h-80 rounded-lg bg-gray-200" />
            <div className="h-64 rounded-lg bg-gray-200" />
          </div>
          <div className="min-w-0 space-y-4 lg:col-span-5 xl:col-span-1">
            <div className="aspect-square rounded-lg bg-gray-200" />
            <div className="h-24 rounded-lg bg-gray-200" />
          </div>
          <div className="min-w-0 lg:col-span-4 xl:col-span-1">
            <div className="h-[32rem] rounded-lg bg-gray-200" />
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Breadcrumb */}
      <div className="flex items-center gap-2 text-sm">
        <Link to="/" className="text-primary-600 hover:underline">
          Gallery
        </Link>
        <span className="text-gray-400">/</span>
        <span className="text-gray-600">{decodedId}</span>
      </div>

      <PageTabs
        label="Image Detail sections"
        activeTab={currentTab}
        onChange={handleTabChange}
        tabs={[
          {
            value: 'main',
            label: 'Image Detail',
            id: 'image-detail-page-tab-main',
            dataTestId: 'image-detail-page-tab-main',
          },
          {
            value: 'q3',
            label: 'Q3',
            id: 'image-detail-page-tab-q3',
            dataTestId: 'image-detail-page-tab-q3',
          },
        ]}
      />

      {/* Main content */}
      <div
        className={`grid grid-cols-1 gap-6 ${
          isQ3Tab
            ? 'lg:grid-cols-[22rem_minmax(0,1fr)] xl:grid-cols-[22rem_minmax(0,1fr)] xl:gap-8'
            : 'lg:grid-cols-12 xl:grid-cols-[22rem_minmax(0,1fr)_minmax(0,1fr)] xl:gap-8'
        }`}
      >
        <div
          className={`space-y-4 ${isQ3Tab ? '' : 'lg:col-span-3 xl:col-span-1'}`}
          data-testid="image-detail-left-column"
        >
          {!isQ3Tab && (
            <div data-testid="view-settings-panel">
              <ControlPanel mode={activeViewerMode} />
            </div>
          )}
          {isQ3Tab && (
            <div data-testid="q3-controls-panel">
              <ControlPanel
                mode={activeViewerMode}
                title="Q3 Controls"
                showLayerControl={false}
                showPercentileControl={false}
                showBoundingBoxesToggle
                showOverlayAppearanceControls={false}
              />
            </div>
          )}
          {isQ3Tab && (
            <Q3StudyScopeCallout
              context="imageDetail"
              dataTestId="image-detail-q3-scope-card"
              currentModelLabel={model}
              currentModelStatus={getQ3ModelScopeStatus(model)}
              action={{
                label: 'Use Q3 defaults',
                onClick: handleApplyQ3Defaults,
                dataTestId: 'image-detail-use-q3-defaults',
              }}
            />
          )}
          {imageDetail && (
            <AnnotationsCard
              annotation={imageDetail.annotation}
              mode={activeViewerMode}
              showBboxes={showBboxes}
              selectedBboxIndex={selectedBboxIndex}
              onBboxSelect={handleBboxSelect}
            />
          )}
        </div>

        {/* Center: Attention viewer */}
        <div
          className={`min-w-0 space-y-4 ${isQ3Tab ? '' : 'lg:col-span-5 xl:col-span-1'}`}
          data-testid="image-detail-center-column"
        >
          {isQ3Tab && (
            <ImageDetailModeSwitch
              mode={currentMode}
              onChange={handleModeChange}
            />
          )}

          <ErrorBoundary resetKeys={[model, layer, method, head, activeViewerMode, percentile]}>
            <AttentionViewer
              imageId={decodedId}
              model={model}
              layer={layer}
              method={method}
              head={head}
              mode={activeViewerMode}
              percentile={percentile}
              showBboxes={showBboxes}
              bboxes={imageDetail?.annotation.bboxes}
              selectedBboxIndex={selectedBboxIndex}
              onBboxSelect={handleBboxSelect}
              className="aspect-square"
            />
          </ErrorBoundary>

          {/* Layer slider */}
          <Card>
            <CardContent>
              <LayerSlider
                currentLayer={layer}
                maxLayers={maxLayers}
                onChange={setLayer}
                isPlaying={isPlaying}
                onPlayingChange={setIsPlaying}
                playSpeed={400}
              />
            </CardContent>
          </Card>
        </div>

        {!isQ3Tab && (
          <div className="min-w-0 lg:col-span-4 xl:col-span-1" data-testid="image-detail-right-column">
            <ErrorBoundary resetKeys={[model, layer, percentile, method, activeViewerMode, selectedBboxIndex, isPlaying]}>
              <ImageDetailMetricsPanel
                imageId={decodedId}
                model={model}
                percentile={percentile}
                method={method}
                mode={activeViewerMode}
                selectedBboxIndex={selectedBboxIndex}
                currentLayer={layer}
                isPlaying={isPlaying}
                enabled={canQueryProgression}
              />
            </ErrorBoundary>
          </div>
        )}
      </div>
    </div>
  );
}
