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

async function stubVariantCompareApis(page: import('@playwright/test').Page) {
  await page.route('**/api/images?**', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify([
        {
          image_id: IMAGE_ID,
          thumbnail_url: `/api/images/${IMAGE_ID}/thumbnail`,
          styles: ['gothic'],
          style_names: ['Gothic'],
          num_bboxes: 1,
        },
      ]),
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
          bboxes: [],
        },
      }),
    });
  });

  await page.route('**/api/attention/models', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        models: ['dinov2'],
        num_layers: 12,
        num_layers_per_model: { dinov2: 12 },
        methods: { dinov2: ['cls'] },
        default_methods: { dinov2: 'cls' },
      }),
    });
  });

  await page.route('**/api/compare/variants?**', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        image_id: IMAGE_ID,
        model: 'dinov2',
        layer: 'layer0',
        method: 'cls',
        show_bboxes: true,
        left: {
          model_key: 'dinov2',
          strategy: null,
          label: 'Frozen (Pretrained)',
          available: true,
          url: '/api/attention/Q2034923_wd0.jpg/overlay?model=dinov2&layer=0&method=cls',
        },
        right: {
          model_key: 'dinov2_finetuned_lora',
          strategy: 'lora',
          label: 'LoRA',
          available: true,
          url: '/api/attention/Q2034923_wd0.jpg/overlay?model=dinov2_finetuned_lora&layer=0&method=cls',
        },
        note: 'ok',
      }),
    });
  });

  await page.route('**/api/metrics/q2_summary?**', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        metric: 'mse',
        label: 'MSE',
        direction: 'lower',
        percentile_dependent: false,
        selected_percentile: null,
        analyzed_layer: 11,
        timestamp: null,
        rows: [],
        strategy_comparisons: [],
      }),
    });
  });
}

async function stubVariantCompareLandingApis(page: import('@playwright/test').Page) {
  await page.route('**/api/images?**', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify([]),
    });
  });

  await page.route('**/api/attention/models', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        models: ['dinov2', 'siglip2', 'clip'],
        num_layers: 12,
        num_layers_per_model: { dinov2: 12, siglip2: 12, clip: 12 },
        methods: { dinov2: ['cls'], siglip2: ['mean'], clip: ['cls'] },
        default_methods: { dinov2: 'cls', siglip2: 'mean', clip: 'cls' },
      }),
    });
  });
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

  test('normalizes legacy fine-tuning URLs into variant compare and disables percentile for threshold-free metrics', async ({ page }) => {
    await stubVariantCompareApis(page);

    const compareResponse = page.waitForResponse(
      (response) =>
        response.url().includes('/api/compare/variants')
        && response.url().includes('left_variant=frozen')
        && response.url().includes('right_variant=lora')
        && response.status() === 200
    );

    await page.goto(`/compare?image=${encodeURIComponent(IMAGE_ID)}&type=frozen&model=dinov2&strategy=lora&metric=mse&percentile=90`);
    await compareResponse;

    await expect(page).toHaveURL(/type=variants/);
    await expect(page).toHaveURL(/left_variant=frozen/);
    await expect(page).toHaveURL(/right_variant=lora/);
    await expect(page.getByText(/threshold-free, so percentile stays visible/)).toBeVisible();
    await expect(page.getByRole('combobox').nth(3)).toBeDisabled();
  });

  test('keeps the Q2-selected model available before an image is chosen', async ({ page }) => {
    await stubVariantCompareLandingApis(page);

    await page.goto('/compare?type=variants&model=siglip2&metric=emd&left_variant=frozen&right_variant=full');

    const modelSelect = page.getByRole('combobox').nth(2);
    await expect(modelSelect).toHaveValue('siglip2');
    await expect(modelSelect.locator('option')).toContainText(['dinov2', 'siglip2', 'clip']);
    await expect(page.getByText('Select an image above to start comparing')).toBeVisible();
  });
});
