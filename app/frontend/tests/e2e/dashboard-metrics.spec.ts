import { expect, test } from '@playwright/test';

test.describe('Dashboard metrics', () => {
  test('supports selecting EMD and shows the threshold-free guidance', async ({ page }) => {
    await page.goto('/dashboard');

    await expect(page.getByRole('heading', { name: 'Dashboard' })).toBeVisible();
    await page.getByRole('combobox').first().selectOption('emd');

    await expect(page.getByText('EMD (lower better)')).toBeVisible();
    await expect(
      page.getByText(/Earth Mover's Distance \/ Wasserstein-1 on a shared 8x8 support/)
    ).toBeVisible();
  });

  test('keeps the dashboard summary aligned to the selected shared method', async ({ page }) => {
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

    await page.route('**/api/compare/all_models_summary?**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          percentile: 90,
          metric: 'iou',
          method: 'cls',
          excluded_models: ['siglip', 'siglip2', 'resnet50'],
          leaderboard: [
            { rank: 1, model: 'dinov2', metric: 'iou', score: 0.58, best_layer: 'layer11' },
            { rank: 2, model: 'dinov3', metric: 'iou', score: 0.54, best_layer: 'layer10' },
            { rank: 3, model: 'mae', metric: 'iou', score: 0.5, best_layer: 'layer9' },
            { rank: 4, model: 'clip', metric: 'iou', score: 0.46, best_layer: 'layer8' },
          ],
          models: {
            dinov2: {
              rank: 1,
              best_layer: 'layer11',
              best_score: 0.58,
              layer_progression: { layer0: 0.22, layer1: 0.31, layer11: 0.58 },
            },
            dinov3: {
              rank: 2,
              best_layer: 'layer10',
              best_score: 0.54,
              layer_progression: { layer0: 0.2, layer1: 0.29, layer10: 0.54 },
            },
            mae: {
              rank: 3,
              best_layer: 'layer9',
              best_score: 0.5,
              layer_progression: { layer0: 0.18, layer1: 0.27, layer9: 0.5 },
            },
            clip: {
              rank: 4,
              best_layer: 'layer8',
              best_score: 0.46,
              layer_progression: { layer0: 0.16, layer1: 0.24, layer8: 0.46 },
            },
          },
        }),
      });
    });

    await page.route('**/api/metrics/model/*/style_breakdown?**', async (route) => {
      const url = new URL(route.request().url());
      const match = url.pathname.match(/\/api\/metrics\/model\/([^/]+)\/style_breakdown/);
      const model = match?.[1] ?? 'dinov2';
      const scores = model === 'clip'
        ? { Gothic: 0.43, Romanesque: 0.35 }
        : { Gothic: 0.55, Romanesque: 0.48 };
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          model,
          layer: 'layer0',
          percentile: 90,
          method: 'cls',
          styles: scores,
          style_counts: { Gothic: 20, Romanesque: 15 },
        }),
      });
    });

    await page.route('**/api/metrics/model/*/feature_breakdown?**', async (route) => {
      const url = new URL(route.request().url());
      const match = url.pathname.match(/\/api\/metrics\/model\/([^/]+)\/feature_breakdown/);
      const model = match?.[1] ?? 'dinov2';
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          model,
          layer: 'layer0',
          percentile: 90,
          method: 'cls',
          total_feature_types: 1,
          features: [
            {
              feature_label: 1,
              feature_name: 'Window',
              mean_iou: model === 'clip' ? 0.32 : 0.41,
              std_iou: 0.05,
              bbox_count: 12,
            },
          ],
        }),
      });
    });

    const summaryResponse = page.waitForResponse(
      (response) =>
        response.url().includes('/api/compare/all_models_summary')
        && response.url().includes('method=cls')
        && response.status() === 200
    );

    await page.goto('/dashboard');
    await summaryResponse;
    await expect(page.getByRole('heading', { name: 'Dashboard' })).toBeVisible();

    await expect(page.getByTestId('dashboard-method-context')).toContainText(
      'Summary panels are using CLS Attention.'
    );
    await expect(page.getByTestId('dashboard-method-context')).toContainText(
      'Excluded models: siglip, siglip2, resnet50.'
    );

    await expect(page.getByTestId('leaderboard-row-dinov2')).toBeVisible();
    await expect(page.getByTestId('leaderboard-row-dinov3')).toBeVisible();
    await expect(page.getByTestId('leaderboard-row-mae')).toBeVisible();
    await expect(page.getByTestId('leaderboard-row-clip')).toBeVisible();
    await expect(page.getByTestId('leaderboard-row-siglip')).toHaveCount(0);
    await expect(page.getByTestId('leaderboard-row-siglip2')).toHaveCount(0);
    await expect(page.getByTestId('leaderboard-row-resnet50')).toHaveCount(0);

    const clipStyleResponse = page.waitForResponse(
      (response) =>
        response.url().includes('/api/metrics/model/clip/style_breakdown')
        && response.url().includes('method=cls')
        && response.status() === 200
    );

    await page.getByTestId('leaderboard-row-clip').click();
    await clipStyleResponse;

    await expect(page.getByTestId('dashboard-method-context')).toContainText(
      'Summary panels are using CLS Attention.'
    );
  });
});
