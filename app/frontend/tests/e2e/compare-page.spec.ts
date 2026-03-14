import { expect, test } from '@playwright/test';

const IMAGE_ID = 'Q2034923_wd0.jpg';

function getModelResult(
  payload: {
    results: Array<{ model: string; iou: number; mse: number }>;
  },
  model: string,
) {
  const result = payload.results.find((entry) => entry.model === model);
  expect(result).toBeTruthy();
  return result!;
}

test.describe('Compare page', () => {
  test('preserves a shared attention method when both selected models support it', async ({ page }) => {
    await page.goto(`/image/${encodeURIComponent(IMAGE_ID)}`);
    await page.getByRole('combobox').nth(1).selectOption('rollout');

    await page.getByRole('link', { name: 'Compare' }).click();

    const comparisonResponse = page.waitForResponse(
      (response) =>
        response.url().includes('/api/compare/models')
        && response.url().includes('method=rollout')
        && response.status() === 200
    );
    await page.getByRole('combobox').first().selectOption(IMAGE_ID);
    await comparisonResponse;

    await expect(page.getByTestId('comparison-method-context')).toContainText(
      'Comparing with shared method: rollout'
    );
    await expect(
      page.locator('.grid.grid-cols-2').getByText(/^Method:\s*rollout$/)
    ).toHaveCount(2);
  });

  test('clamps the shared layer when switching to lower-depth model pairs', async ({ page }) => {
    await page.goto(`/image/${encodeURIComponent(IMAGE_ID)}`);
    await page.getByTestId('layer-last').click();
    await expect(page.getByTestId('active-layer-indicator')).toContainText('Focused: Layer 11');

    await page.getByRole('link', { name: 'Compare' }).click();
    await page.getByRole('combobox').first().selectOption(IMAGE_ID);

    await expect(page.getByText('Model Comparison')).toBeVisible();
    await page.getByRole('combobox').nth(2).selectOption('siglip');
    await page.getByRole('combobox').nth(3).selectOption('resnet50');

    await expect(page.getByText('Failed to load comparison')).toHaveCount(0);
    await expect(page.getByRole('combobox').nth(2)).toHaveValue('siglip');
    await expect(page.getByRole('combobox').nth(3)).toHaveValue('resnet50');
    await expect(page.getByTestId('comparison-method-context')).toContainText(
      "Using each model's default attention method because cls is not shared by both selected models."
    );
    await expect(
      page.locator('.grid.grid-cols-2').getByText(/^Method:\s*mean$/)
    ).toHaveCount(1);
    await expect(
      page.locator('.grid.grid-cols-2').getByText(/^Method:\s*gradcam$/)
    ).toHaveCount(1);
    await expect(page.getByText(/KL:/)).toHaveCount(2);
    await expect(page.getByText(/EMD:/)).toHaveCount(2);
  });

  test('switches compare metrics to bbox scope when a bounding box is selected', async ({ page }) => {
    const initialCompareResponse = page.waitForResponse(
      (response) =>
        response.url().includes('/api/compare/models')
        && !response.url().includes('bbox_index=')
        && response.status() === 200
    );

    await page.goto(`/compare?image=${encodeURIComponent(IMAGE_ID)}&type=models`);
    const initialPayload = await (await initialCompareResponse).json();

    const leftModel = await page.getByRole('combobox').nth(2).inputValue();
    const initialLeft = getModelResult(initialPayload, leftModel);

    const bboxCompareResponse = page.waitForResponse(
      (response) =>
        response.url().includes('/api/compare/models')
        && response.url().includes('bbox_index=0')
        && response.status() === 200
    );

    await page.getByTestId('bbox-hitbox-0').first().click({ force: true });
    const bboxPayload = await (await bboxCompareResponse).json();
    const bboxLeft = getModelResult(bboxPayload, leftModel);

    expect(bboxPayload.selection.mode).toBe('bbox');
    expect(bboxPayload.selection.bbox_index).toBe(0);
    expect(bboxLeft.iou).not.toBe(initialLeft.iou);
    expect(bboxLeft.mse).not.toBe(initialLeft.mse);

    const leftMetrics = page.getByTestId('comparison-metrics-left');
    await expect(leftMetrics).toContainText('Feature-level metrics');
    await expect(leftMetrics).toContainText(bboxPayload.selection.bbox_label);
    await expect(leftMetrics).toContainText(`IoU: ${bboxLeft.iou.toFixed(3)}`);
    await expect(leftMetrics).toContainText(`MSE: ${bboxLeft.mse.toFixed(4)}`);
  });

  test('shows per-model unavailable messaging instead of stale whole-image metrics', async ({ page }) => {
    await page.goto(`/compare?image=${encodeURIComponent(IMAGE_ID)}&type=models`);
    await expect(page.getByTestId('comparison-metrics-left')).toContainText('Whole-image metrics');

    const leftModel = await page.getByRole('combobox').nth(2).inputValue();
    const rightModel = await page.getByRole('combobox').nth(3).inputValue();
    const rightMetrics = page.getByTestId('comparison-metrics-right');
    const previousRightText = (await rightMetrics.textContent()) ?? '';

    await page.route('**/api/compare/models?**', async (route) => {
      const url = new URL(route.request().url());
      if (url.searchParams.get('bbox_index') !== '0') {
        await route.continue();
        return;
      }

      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          image_id: IMAGE_ID,
          models: [leftModel, rightModel],
          layer: 'layer0',
          percentile: 90,
          selection: {
            mode: 'bbox',
            bbox_index: 0,
            bbox_label: 'Window',
          },
          results: [
            {
              image_id: IMAGE_ID,
              model: leftModel,
              layer: 'layer0',
              percentile: 90,
              iou: 0.222,
              coverage: 0.444,
              mse: 0.0123,
              kl: 0.0456,
              emd: 0.0678,
              attention_area: 0.12,
              annotation_area: 0.09,
              method: 'cls',
            },
          ],
          heatmap_urls: {},
          unavailable_models: {
            [rightModel]: `Feature-level metrics unavailable because cached attention is missing for ${rightModel}/layer0/cls/${IMAGE_ID}.`,
          },
        }),
      });
    });

    await page.getByTestId('bbox-hitbox-0').first().click({ force: true });

    await expect(page.getByTestId('comparison-metrics-right-unavailable')).toContainText(
      'Feature-level metrics unavailable because cached attention is missing'
    );
    await expect(rightMetrics).toContainText('Feature-level metrics');
    await expect(rightMetrics).not.toContainText('Whole-image metrics');
    await expect(rightMetrics).not.toContainText(previousRightText.trim());
  });
});
