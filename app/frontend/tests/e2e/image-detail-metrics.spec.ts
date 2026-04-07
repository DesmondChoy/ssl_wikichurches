import type { Page } from '@playwright/test';
import { expect, test } from '@playwright/test';

const IMAGE_ID = 'Q2034923_wd0.jpg';

function getSelectByLabel(page: Page, label: string) {
  return page
    .locator('div.flex.flex-col.gap-1')
    .filter({ has: page.locator('label', { hasText: label }) })
    .locator('select');
}

async function openImageDetail(page: Page) {
  await page.goto(`/image/${encodeURIComponent(IMAGE_ID)}`);
  await expect(page.getByTestId('metrics-panel')).toBeVisible({ timeout: 20000 });
}

async function stubImageDetailModeApis(page: Page) {
  await page.route('**/api/attention/models', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        models: ['dinov2'],
        num_layers: 12,
        num_layers_per_model: { dinov2: 12 },
        methods: { dinov2: ['cls', 'rollout'] },
        num_heads_per_model: { dinov2: 12 },
        per_head_methods: ['cls'],
        per_head_available_models: ['dinov2'],
        default_methods: { dinov2: 'cls' },
      }),
    });
  });

  await page.route(`**/api/images/${IMAGE_ID}`, async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        image_id: IMAGE_ID,
        image_url: `/api/images/${IMAGE_ID}/file`,
        thumbnail_url: `/api/images/${IMAGE_ID}/thumbnail`,
        available_models: ['dinov2'],
        annotation: {
          image_id: IMAGE_ID,
          styles: ['gothic'],
          style_names: ['Gothic'],
          num_bboxes: 1,
          bboxes: [
            {
              left: 0.1,
              top: 0.1,
              width: 0.35,
              height: 0.4,
              label: 1,
              label_name: 'Spire',
            },
          ],
        },
      }),
    });
  });

  await page.route(`**/api/attention/${IMAGE_ID}/raw?**`, async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        attention: Array.from({ length: 16 }, (_, idx) => idx / 15),
        shape: [4, 4],
        min_value: 0,
        max_value: 1,
      }),
    });
  });

  await page.route(`**/api/attention/${IMAGE_ID}/similarity?**`, async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        similarity: Array.from({ length: 16 }, (_, idx) => 0.15 + idx * 0.05),
        patch_grid: [4, 4],
        min_similarity: 0.15,
        max_similarity: 0.9,
        bbox_patch_indices: [0, 1],
      }),
    });
  });

  await page.route(`**/api/metrics/${IMAGE_ID}/progression?**`, async (route) => {
    const url = new URL(route.request().url());
    const bboxIndexParam = url.searchParams.get('bbox_index');
    const bboxIndex = bboxIndexParam === null ? null : Number(bboxIndexParam);
    const mode = bboxIndex === null ? 'union' : 'bbox';
    const bboxLabel = bboxIndex === null ? null : 'Spire';

    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        image_id: IMAGE_ID,
        model: 'dinov2',
        method: url.searchParams.get('method') ?? 'cls',
        percentile: Number(url.searchParams.get('percentile') ?? '90'),
        selection: {
          mode,
          bbox_index: bboxIndex,
          bbox_label: bboxLabel,
        },
        metrics: [
          {
            key: 'iou',
            label: 'IoU Score',
            direction: 'higher',
            default_enabled: true,
            percentile_dependent: true,
          },
          {
            key: 'coverage',
            label: 'Coverage',
            direction: 'higher',
            default_enabled: true,
            percentile_dependent: false,
          },
          {
            key: 'mse',
            label: 'MSE',
            direction: 'lower',
            default_enabled: true,
            percentile_dependent: false,
          },
          {
            key: 'emd',
            label: 'EMD',
            direction: 'lower',
            default_enabled: true,
            percentile_dependent: false,
          },
        ],
        layers: Array.from({ length: 12 }, (_, layer) => ({
          layer,
          layer_key: `layer${layer}`,
          values: {
            iou: 0.2 + layer * 0.01,
            coverage: 0.35 + layer * 0.005,
            mse: 0.08 - layer * 0.002,
            emd: 0.14 - layer * 0.003,
          },
        })),
      }),
    });
  });
}

test.describe('Image detail metrics chart', () => {
  test.beforeEach(async ({ page }) => {
    await stubImageDetailModeApis(page);
  });

  test('uses the new desktop layout and removes the page-local compare CTA', async ({ page }) => {
    await openImageDetail(page);

    const leftColumn = page.getByTestId('image-detail-left-column');
    const centerColumn = page.getByTestId('image-detail-center-column');
    const rightColumn = page.getByTestId('image-detail-right-column');
    const viewSettings = page.getByTestId('view-settings-panel');
    const annotations = page.getByTestId('annotations-card');

    await expect(leftColumn).toBeVisible();
    await expect(rightColumn).toBeVisible();
    await expect(rightColumn.getByTestId('metrics-panel')).toBeVisible();

    const viewBox = await viewSettings.boundingBox();
    const annotationsBox = await annotations.boundingBox();

    expect(viewBox).not.toBeNull();
    expect(annotationsBox).not.toBeNull();
    expect(viewBox!.y).toBeLessThan(annotationsBox!.y);

    const centerBox = await centerColumn.boundingBox();
    const rightBox = await rightColumn.boundingBox();

    expect(centerBox).not.toBeNull();
    expect(rightBox).not.toBeNull();
    expect(Math.abs(centerBox!.width - rightBox!.width)).toBeLessThanOrEqual(16);

    await expect(rightColumn.getByRole('link', { name: 'Compare Models' })).toHaveCount(0);
  });

  test('supports metric toggle labels with explicit directionality and bbox switching', async ({ page }) => {
    await openImageDetail(page);

    const toggleGroup = page.getByTestId('metric-toggle-group');
    const coverageToggle = toggleGroup.getByRole('checkbox', { name: 'Coverage (Higher better)' });
    const iouToggle = page.getByTestId('metric-toggle-iou');
    const coverageTogglePill = page.getByTestId('metric-toggle-coverage');
    const mseToggle = page.getByTestId('metric-toggle-mse');
    const emdToggle = page.getByTestId('metric-toggle-emd');

    await expect(toggleGroup.getByText('IoU Score (Higher better)')).toBeVisible();
    await expect(toggleGroup.getByText('MSE (Lower better)')).toBeVisible();
    await expect(toggleGroup.getByText('EMD (Lower better)')).toBeVisible();
    await expect(coverageToggle).toBeChecked();
    await expect(iouToggle).toHaveAttribute('data-selected', 'true');
    await expect(iouToggle).toHaveClass(/from-blue-50/);
    await expect(coverageTogglePill).toHaveClass(/from-lime-50/);
    await expect(mseToggle).toHaveClass(/from-rose-50/);
    await expect(emdToggle).toHaveClass(/from-slate-50/);

    await iouToggle.hover();
    await expect(page.getByText(/Overlap between thresholded attention and the annotation\./)).toBeVisible();
    await expect(page.getByText(/Use it to judge how tightly the highlighted region lines up with the labeled feature\./)).toBeVisible();

    await coverageTogglePill.hover();
    await expect(page.getByText(/Fraction of attention mass inside the annotation\./)).toBeVisible();
    await expect(page.getByText(/spending its attention on the feature rather than the background\./)).toBeVisible();

    await mseToggle.hover();
    await expect(page.getByText(/Mean squared error against the Gaussian soft-union target\./)).toBeVisible();
    await expect(page.getByText(/Use it to judge whether the overall attention shape matches the annotated feature/)).toBeVisible();

    await emdToggle.hover();
    await expect(page.getByText(/Earth Mover's Distance \(Wasserstein-1\) on a shared 8x8 support/)).toBeVisible();
    await expect(page.getByText(/how far the attention mass would need to move spatially/)).toBeVisible();

    await coverageToggle.uncheck();
    await expect(coverageToggle).not.toBeChecked();
    await expect(coverageTogglePill).toHaveAttribute('data-selected', 'false');

    const firstBbox = page.getByTestId('bbox-list-item-0');
    await firstBbox.click();
    await expect(page.getByText('Showing bbox metrics')).toBeVisible();

    await firstBbox.click();
    await expect(page.getByText('Showing union metrics')).toBeVisible();
  });

  test('hides the per-head selector when cache-backed support is not advertised', async ({ page }) => {
    await page.route('**/api/attention/models', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          models: ['dinov2', 'dinov3', 'mae', 'clip', 'siglip', 'siglip2', 'resnet50'],
          num_layers: 12,
          num_layers_per_model: {
            dinov2: 12,
            dinov3: 12,
            mae: 12,
            clip: 12,
            siglip: 12,
            siglip2: 12,
            resnet50: 4,
          },
          methods: {
            dinov2: ['cls', 'rollout'],
            dinov3: ['cls', 'rollout'],
            mae: ['cls', 'rollout'],
            clip: ['cls', 'rollout'],
            siglip: ['mean'],
            siglip2: ['mean'],
            resnet50: ['gradcam'],
          },
          num_heads_per_model: {
            dinov2: 12,
            dinov3: 12,
            mae: 12,
            clip: 12,
            siglip: 12,
            siglip2: 12,
            resnet50: 0,
          },
          per_head_methods: ['cls', 'mean'],
          per_head_available_models: [],
          default_methods: {
            dinov2: 'cls',
            dinov3: 'cls',
            mae: 'cls',
            clip: 'cls',
            siglip: 'mean',
            siglip2: 'mean',
            resnet50: 'gradcam',
          },
        }),
      });
    });

    await openImageDetail(page);
    await expect(getSelectByLabel(page, 'Attention Head')).toHaveCount(0);
  });

  test('shows the per-head selector only when cache-backed support is advertised and resets back to fused mode', async ({ page }) => {
    await page.route('**/api/attention/models', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          models: ['dinov2', 'dinov3', 'mae', 'clip', 'siglip', 'siglip2', 'resnet50'],
          num_layers: 12,
          num_layers_per_model: {
            dinov2: 12,
            dinov3: 12,
            mae: 12,
            clip: 12,
            siglip: 12,
            siglip2: 12,
            resnet50: 4,
          },
          methods: {
            dinov2: ['cls', 'rollout'],
            dinov3: ['cls', 'rollout'],
            mae: ['cls', 'rollout'],
            clip: ['cls', 'rollout'],
            siglip: ['mean'],
            siglip2: ['mean'],
            resnet50: ['gradcam'],
          },
          num_heads_per_model: {
            dinov2: 12,
            dinov3: 12,
            mae: 12,
            clip: 12,
            siglip: 12,
            siglip2: 12,
            resnet50: 0,
          },
          per_head_methods: ['cls', 'mean'],
          per_head_available_models: ['dinov2'],
          default_methods: {
            dinov2: 'cls',
            dinov3: 'cls',
            mae: 'cls',
            clip: 'cls',
            siglip: 'mean',
            siglip2: 'mean',
            resnet50: 'gradcam',
          },
        }),
      });
    });

    await page.route(`**/api/attention/${IMAGE_ID}/raw?**`, async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          attention: Array.from({ length: 16 * 16 }, (_, idx) => (idx % 16) / 15),
          shape: [16, 16],
          min_value: 0,
          max_value: 1,
        }),
      });
    });

    await openImageDetail(page);

    const modelSelect = getSelectByLabel(page, 'Model');
    const methodSelect = getSelectByLabel(page, 'Attention Method');
    const headSelect = () => getSelectByLabel(page, 'Attention Head');
    const viewerHeadBadge = page.getByTestId('image-detail-center-column').getByText('Head 3');

    await expect(headSelect()).toBeVisible();
    await expect(headSelect()).toHaveValue('-1');

    await headSelect().selectOption('3');

    await expect(viewerHeadBadge).toBeVisible();
    await expect(headSelect()).toHaveValue('3');

    await methodSelect.selectOption('rollout');
    await expect(headSelect()).toHaveCount(0);
    await expect(viewerHeadBadge).toHaveCount(0);

    await methodSelect.selectOption('cls');
    await expect(headSelect()).toBeVisible();
    await expect(headSelect()).toHaveValue('-1');
    await expect(viewerHeadBadge).toHaveCount(0);

    await modelSelect.selectOption('resnet50');
    await expect(headSelect()).toHaveCount(0);

    await modelSelect.selectOption('dinov2');
    await expect(headSelect()).toBeVisible();
    await expect(headSelect()).toHaveValue('-1');
  });

  test('keeps Q3 framing behind the Q3 tab and resets the inspector to Q3 defaults there', async ({ page }) => {
    await page.route('**/api/attention/models', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          models: ['dinov2', 'dinov3', 'mae', 'clip', 'siglip', 'siglip2', 'resnet50'],
          num_layers: 12,
          num_layers_per_model: {
            dinov2: 12,
            dinov3: 12,
            mae: 12,
            clip: 12,
            siglip: 12,
            siglip2: 12,
            resnet50: 4,
          },
          methods: {
            dinov2: ['cls', 'rollout'],
            dinov3: ['cls', 'rollout'],
            mae: ['cls', 'rollout'],
            clip: ['cls', 'rollout'],
            siglip: ['mean'],
            siglip2: ['mean'],
            resnet50: ['gradcam'],
          },
          num_heads_per_model: {
            dinov2: 12,
            dinov3: 12,
            mae: 12,
            clip: 12,
            siglip: 12,
            siglip2: 12,
            resnet50: 0,
          },
          per_head_methods: ['cls', 'mean'],
          per_head_available_models: ['dinov2'],
          default_methods: {
            dinov2: 'cls',
            dinov3: 'cls',
            mae: 'cls',
            clip: 'cls',
            siglip: 'mean',
            siglip2: 'mean',
            resnet50: 'gradcam',
          },
        }),
      });
    });

    await page.route(`**/api/attention/${IMAGE_ID}/raw?**`, async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          attention: Array.from({ length: 16 * 16 }, (_, idx) => (idx % 16) / 15),
          shape: [16, 16],
          min_value: 0,
          max_value: 1,
        }),
      });
    });

    await openImageDetail(page);

    const mainTab = page.getByRole('tab', { name: 'Image Detail' });
    const q3Tab = page.getByRole('tab', { name: 'Q3' });
    const modelSelect = getSelectByLabel(page, 'Model');
    const methodSelect = getSelectByLabel(page, 'Attention Method');
    const headSelect = getSelectByLabel(page, 'Attention Head');
    const q3ScopeCard = page.getByTestId('image-detail-q3-scope-card');
    const currentModelStatus = page.getByTestId('image-detail-q3-current-model-status');
    const centerColumn = page.getByTestId('image-detail-center-column');

    await expect(mainTab).toHaveAttribute('aria-selected', 'true');
    await expect(q3ScopeCard).toHaveCount(0);
    await expect(page.getByTestId('image-detail-use-q3-defaults')).toHaveCount(0);
    await expect(page.getByTestId('image-detail-mode-switch')).toHaveCount(0);
    await expect(page.getByTestId('view-settings-panel')).toBeVisible();
    await expect(page.getByTestId('metrics-panel')).toBeVisible();
    await expect(modelSelect.locator('option')).toHaveText([
      'Dinov2',
      'Dinov3',
      'Mae',
      'Clip',
      'Siglip',
      'Siglip2',
      'Resnet50',
    ]);

    const mainCenterBox = await centerColumn.boundingBox();
    expect(mainCenterBox).not.toBeNull();

    await q3Tab.click();

    await expect(q3Tab).toHaveAttribute('aria-selected', 'true');
    await expect(page.getByTestId('view-settings-panel')).toHaveCount(0);
    await expect(page.getByTestId('q3-controls-panel')).toBeVisible();
    await expect(page.getByTestId('metrics-panel')).toHaveCount(0);
    await expect(page.getByTestId('image-detail-q3-spacer')).toBeVisible();
    await expect(page.getByTestId('image-detail-mode-switch')).toBeVisible();
    await expect(page.getByTestId('q3-controls-panel').getByText('Show Bounding Boxes')).toBeVisible();
    await expect(q3ScopeCard).toContainText('Primary Q3 study scope');
    await expect(q3ScopeCard).toContainText(
      'Primary claim centers dinov2, dinov3, mae, and clip'
    );
    await expect(q3ScopeCard).toContainText(
      'Use Dashboard Q3 to compare frozen, LoRA, and full'
    );
    await expect(currentModelStatus).toHaveText('Primary study');
    await expect(page.getByTestId('annotations-card')).toBeVisible();
    await expect(headSelect).toBeVisible();

    const q3CenterBox = await centerColumn.boundingBox();
    expect(q3CenterBox).not.toBeNull();
    expect(Math.abs(q3CenterBox!.width - mainCenterBox!.width)).toBeLessThanOrEqual(16);

    await modelSelect.selectOption('siglip2');
    await expect(currentModelStatus).toHaveText('Outside primary scope');

    await page.getByTestId('image-detail-use-q3-defaults').click();

    await expect(page.getByTestId('image-detail-q3-current-model-status')).toHaveText('Primary study');

    await mainTab.click();

    await expect(mainTab).toHaveAttribute('aria-selected', 'true');
    await expect(q3ScopeCard).toHaveCount(0);
    await expect(page.getByTestId('image-detail-mode-switch')).toHaveCount(0);
    await expect(page.getByTestId('view-settings-panel')).toBeVisible();
    await expect(page.getByTestId('q3-controls-panel')).toHaveCount(0);
    await expect(page.getByTestId('metrics-panel')).toBeVisible();
    await expect(page.getByTestId('active-layer-indicator')).toContainText('Focused: Layer 11');
    await expect(modelSelect).toHaveValue('dinov2');
    await expect(methodSelect).toHaveValue('cls');
    await expect(getSelectByLabel(page, 'Attention Head')).toBeVisible();
    await expect(getSelectByLabel(page, 'Attention Head')).toHaveValue('-1');
    await expect(modelSelect.locator('option')).toHaveText([
      'Dinov2',
      'Dinov3',
      'Mae',
      'Clip',
      'Siglip',
      'Siglip2',
      'Resnet50',
    ]);
    await expect(page.getByTestId('active-layer-indicator')).toContainText('Focused: Layer 11');
  });

  test('keeps the chart synced with layer controls and playback reveal state', async ({ page }) => {
    await openImageDetail(page);

    const activeLayerIndicator = page.getByTestId('active-layer-indicator');
    const revealStatus = page.getByTestId('chart-reveal-status');
    const chart = page.getByTestId('layer-metrics-chart');

    await expect(activeLayerIndicator).toContainText('Focused: Layer 0');
    await expect(revealStatus).toContainText('Showing full layer history');
    await expect(page.getByTestId('chart-x-axis-caption')).toHaveText('Layers');
    await expect(chart.getByText('L0')).toHaveCount(0);
    await expect(chart.getByText('L1')).toHaveCount(0);

    await page.getByTestId('layer-next').click();
    await expect(activeLayerIndicator).toContainText('Focused: Layer 1');

    await page.getByTestId('layer-play-toggle').click();
    await expect(revealStatus).toContainText('Revealing layers 0-');

    await expect.poll(async () => {
      const text = await activeLayerIndicator.textContent();
      const match = text?.match(/Layer (\d+)/);
      return match ? Number(match[1]) : -1;
    }).toBeGreaterThanOrEqual(2);

    await page.getByTestId('layer-play-toggle').click();
    await expect(activeLayerIndicator).toContainText('Focused: Layer');
    await expect(chart).toBeVisible();
  });

  test('switches between head attention and feature similarity without mixing overlays or copy', async ({ page }) => {
    await openImageDetail(page);

    await page.getByRole('tab', { name: 'Q3' }).click();

    const headModeButton = page.getByTestId('image-detail-mode-head_attention');
    const featureModeButton = page.getByTestId('image-detail-mode-feature_similarity');
    const headSelect = () => getSelectByLabel(page, 'Attention Head');

    await expect(headModeButton).toHaveAttribute('aria-pressed', 'true');
    await expect(page.getByTestId('viewer-info-badge')).toContainText('Head Attention');
    await expect(page.getByTestId('annotations-helper-copy')).toContainText('context while you inspect attention');
    await expect(page.getByTestId('metrics-panel')).toHaveCount(0);
    await expect(headSelect()).toBeVisible();
    await expect(headSelect()).toHaveValue('-1');
    await expect(page.getByTestId('attention-overlay-image')).toBeVisible();

    await headSelect().selectOption('3');
    await expect(page.getByTestId('viewer-info-badge')).toContainText('Head 3');

    await featureModeButton.click();

    await expect(featureModeButton).toHaveAttribute('aria-pressed', 'true');
    await expect(headSelect()).toHaveCount(0);
    await expect(page.getByTestId('attention-overlay-image')).toHaveCount(0);
    await expect(page.getByTestId('annotations-helper-copy')).toContainText('feature-similarity query');
    await expect(page.getByTestId('metrics-panel')).toHaveCount(0);
    await expect(page.getByTestId('similarity-selection-hint')).toBeVisible();
    await expect(page.getByTestId('viewer-info-badge')).toContainText('Feature Similarity');

    await page.getByTestId('bbox-list-item-0').click();

    await expect(page.getByTestId('similarity-overlay-image')).toBeVisible();
    await expect(page.getByTestId('similarity-stats')).toBeVisible();
    await expect(page.getByTestId('similarity-legend')).toBeVisible();
    await expect(page.getByTestId('annotations-helper-copy')).toContainText('Spire is driving the feature-similarity overlay');
    await expect(page.getByTestId('metrics-panel')).toHaveCount(0);

    await page.getByRole('tab', { name: 'Image Detail' }).click();

    await expect(page.getByTestId('image-detail-mode-switch')).toHaveCount(0);
    await expect(page.getByTestId('attention-overlay-image')).toBeVisible();
    await expect(page.getByTestId('similarity-stats')).toHaveCount(0);
    await expect(page.getByTestId('similarity-legend')).toHaveCount(0);
    await expect(page.getByTestId('annotations-helper-copy')).toContainText('context while you inspect attention');

    await page.getByRole('tab', { name: 'Q3' }).click();

    await expect(featureModeButton).toHaveAttribute('aria-pressed', 'true');
    await expect(page.getByTestId('similarity-overlay-image')).toBeVisible();
    await expect(page.getByTestId('similarity-stats')).toBeVisible();
    await expect(page.getByTestId('similarity-legend')).toBeVisible();

    await headModeButton.click();

    await expect(headModeButton).toHaveAttribute('aria-pressed', 'true');
    await expect(headSelect()).toBeVisible();
    await expect(headSelect()).toHaveValue('3');
    await expect(page.getByTestId('attention-overlay-image')).toBeVisible();
    await expect(page.getByTestId('similarity-stats')).toHaveCount(0);
    await expect(page.getByTestId('similarity-legend')).toHaveCount(0);
    await expect(page.getByTestId('viewer-info-badge')).toContainText('Head Attention');
  });

  test('restores tab and mode query params together and falls back to the default tab for invalid values', async ({ page }) => {
    await page.goto(`/image/${encodeURIComponent(IMAGE_ID)}?tab=q3&mode=feature_similarity`);
    await expect(page.getByTestId('image-detail-q3-scope-card')).toBeVisible({ timeout: 20000 });

    await expect(page.getByRole('tab', { name: 'Q3' })).toHaveAttribute('aria-selected', 'true');
    await expect(page.getByTestId('image-detail-q3-scope-card')).toBeVisible();
    await expect(page.getByTestId('view-settings-panel')).toHaveCount(0);
    await expect(page.getByTestId('q3-controls-panel')).toBeVisible();
    await expect(page.getByTestId('metrics-panel')).toHaveCount(0);
    await expect(page.getByTestId('image-detail-mode-feature_similarity')).toHaveAttribute('aria-pressed', 'true');
    await expect(getSelectByLabel(page, 'Attention Head')).toHaveCount(0);
    await expect(page.getByTestId('viewer-info-badge')).toContainText('Feature Similarity');

    await page.reload();

    await expect(page.getByRole('tab', { name: 'Q3' })).toHaveAttribute('aria-selected', 'true');
    await expect(page.getByTestId('image-detail-mode-feature_similarity')).toHaveAttribute('aria-pressed', 'true');
    await expect(page).toHaveURL(/tab=q3/);
    await expect(page).toHaveURL(/mode=feature_similarity/);
    await expect(getSelectByLabel(page, 'Attention Head')).toHaveCount(0);

    await page.goto(`/image/${encodeURIComponent(IMAGE_ID)}?tab=not_a_real_tab&mode=not_a_real_mode`);
    await expect(page.getByTestId('view-settings-panel')).toBeVisible({ timeout: 20000 });

    await expect(page.getByRole('tab', { name: 'Image Detail' })).toHaveAttribute('aria-selected', 'true');
    await expect(page.getByTestId('image-detail-q3-scope-card')).toHaveCount(0);
    await expect(page.getByTestId('image-detail-mode-switch')).toHaveCount(0);
    await expect(page.getByTestId('viewer-info-badge')).toContainText('Head Attention');
    await expect(getSelectByLabel(page, 'Attention Head')).toBeVisible();
  });
});
