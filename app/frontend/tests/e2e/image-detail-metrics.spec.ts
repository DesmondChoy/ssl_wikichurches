import { expect, test } from '@playwright/test';

const IMAGE_ID = 'Q2034923_wd0.jpg';

test.describe('Image detail metrics chart', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto(`/image/${encodeURIComponent(IMAGE_ID)}`);
    await expect(page.getByTestId('metrics-panel')).toBeVisible({ timeout: 20000 });
  });

  test('uses the new desktop layout and removes the page-local compare CTA', async ({ page }) => {
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
    await expect(coverageTogglePill).toHaveClass(/from-teal-50/);
    await expect(mseToggle).toHaveClass(/from-rose-50/);
    await expect(emdToggle).toHaveClass(/from-orange-50/);

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

  test('keeps the chart synced with layer controls and playback reveal state', async ({ page }) => {
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
});
