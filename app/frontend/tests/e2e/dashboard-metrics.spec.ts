import { expect, test, type Page } from '@playwright/test';

async function stubDashboardApis(page: Page) {
  await page.route('**/api/attention/models', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        models: ['dinov2', 'clip', 'siglip2', 'resnet50'],
        num_layers: 12,
        num_layers_per_model: {
          dinov2: 12,
          clip: 12,
          siglip2: 12,
          resnet50: 4,
        },
        methods: {
          dinov2: ['cls', 'rollout'],
          clip: ['cls', 'rollout'],
          siglip2: ['mean'],
          resnet50: ['gradcam'],
        },
        default_methods: {
          dinov2: 'cls',
          clip: 'cls',
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
        ranking_mode: 'default_method',
        method: null,
        excluded_models: [],
        leaderboard: [
          { rank: 1, model: 'dinov2', metric: 'iou', score: 0.58, best_layer: 'layer11', method_used: 'cls' },
        ],
        models: {
          dinov2: {
            rank: 1,
            best_layer: 'layer11',
            best_score: 0.58,
            method_used: 'cls',
            layer_progression: { layer0: 0.22, layer11: 0.58 },
          },
        },
      }),
    });
  });

  await page.route('**/api/metrics/model/*/style_breakdown?**', async (route) => {
    const url = new URL(route.request().url());
    const match = url.pathname.match(/\/api\/metrics\/model\/([^/]+)\/style_breakdown/);
    const model = match?.[1] ?? 'dinov2';
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        model,
        layer: 'layer0',
        percentile: 90,
        method: 'cls',
        styles: { Gothic: 0.55, Romanesque: 0.48 },
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
            mean_iou: 0.41,
            std_iou: 0.05,
            bbox_count: 12,
          },
        ],
      }),
    });
  });
}

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

  test('switches dashboard ranking modes and uses the selected row method', async ({ page }) => {
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
      const url = new URL(route.request().url());
      const rankingMode = url.searchParams.get('ranking_mode');
      const isBestAvailable = rankingMode === 'best_available';
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          percentile: 90,
          metric: 'iou',
          ranking_mode: isBestAvailable ? 'best_available' : 'default_method',
          method: null,
          excluded_models: [],
          leaderboard: isBestAvailable
            ? [
                { rank: 1, model: 'dinov2', metric: 'iou', score: 0.64, best_layer: 'layer10', method_used: 'rollout' },
                { rank: 2, model: 'clip', metric: 'iou', score: 0.57, best_layer: 'layer7', method_used: 'rollout' },
                { rank: 3, model: 'dinov3', metric: 'iou', score: 0.54, best_layer: 'layer10', method_used: 'cls' },
                { rank: 4, model: 'mae', metric: 'iou', score: 0.5, best_layer: 'layer9', method_used: 'cls' },
              ]
            : [
                { rank: 1, model: 'dinov2', metric: 'iou', score: 0.58, best_layer: 'layer11', method_used: 'cls' },
                { rank: 2, model: 'dinov3', metric: 'iou', score: 0.54, best_layer: 'layer10', method_used: 'cls' },
                { rank: 3, model: 'mae', metric: 'iou', score: 0.5, best_layer: 'layer9', method_used: 'cls' },
                { rank: 4, model: 'clip', metric: 'iou', score: 0.46, best_layer: 'layer8', method_used: 'cls' },
              ],
          models: {
            dinov2: {
              rank: 1,
              best_layer: isBestAvailable ? 'layer10' : 'layer11',
              best_score: isBestAvailable ? 0.64 : 0.58,
              method_used: isBestAvailable ? 'rollout' : 'cls',
              layer_progression: isBestAvailable
                ? { layer0: 0.28, layer1: 0.39, layer10: 0.64 }
                : { layer0: 0.22, layer1: 0.31, layer11: 0.58 },
            },
            dinov3: {
              rank: isBestAvailable ? 3 : 2,
              best_layer: 'layer10',
              best_score: 0.54,
              method_used: 'cls',
              layer_progression: { layer0: 0.2, layer1: 0.29, layer10: 0.54 },
            },
            mae: {
              rank: 4,
              best_layer: 'layer9',
              best_score: 0.5,
              method_used: 'cls',
              layer_progression: { layer0: 0.18, layer1: 0.27, layer9: 0.5 },
            },
            clip: {
              rank: isBestAvailable ? 2 : 4,
              best_layer: isBestAvailable ? 'layer7' : 'layer8',
              best_score: isBestAvailable ? 0.57 : 0.46,
              method_used: isBestAvailable ? 'rollout' : 'cls',
              layer_progression: isBestAvailable
                ? { layer0: 0.21, layer1: 0.33, layer7: 0.57 }
                : { layer0: 0.16, layer1: 0.24, layer8: 0.46 },
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
        && response.url().includes('ranking_mode=default_method')
        && response.status() === 200
    );

    await page.goto('/dashboard');
    await summaryResponse;
    await expect(page.getByRole('heading', { name: 'Dashboard' })).toBeVisible();

    await expect(page.getByTestId('dashboard-method-context')).toContainText(
      'Leaderboard and layer progression rank each model by its default attention method.'
    );
    await expect(page.getByTestId('leaderboard-row-meta-clip')).toContainText('CLS Attention');

    await expect(page.getByTestId('leaderboard-row-dinov2')).toBeVisible();
    await expect(page.getByTestId('leaderboard-row-dinov3')).toBeVisible();
    await expect(page.getByTestId('leaderboard-row-mae')).toBeVisible();
    await expect(page.getByTestId('leaderboard-row-clip')).toBeVisible();

    const bestAvailableResponse = page.waitForResponse(
      (response) =>
        response.url().includes('/api/compare/all_models_summary')
        && response.url().includes('ranking_mode=best_available')
        && response.status() === 200
    );

    await page.getByTestId('dashboard-ranking-mode-best_available').click();
    await bestAvailableResponse;

    await expect(page.getByTestId('dashboard-method-context')).toContainText(
      "Leaderboard and layer progression rank each model by its strongest available attention method."
    );
    await expect(page.getByTestId('leaderboard-row-meta-dinov2')).toContainText('Rollout Attention');
    await expect(page.getByTestId('leaderboard-row-meta-clip')).toContainText('Rollout Attention');

    const clipStyleResponse = page.waitForResponse(
      (response) =>
        response.url().includes('/api/metrics/model/clip/style_breakdown')
        && response.url().includes('method=rollout')
        && response.status() === 200
    );

    await page.getByTestId('leaderboard-row-clip').click();
    await clipStyleResponse;

    await expect(page.getByTestId('dashboard-method-context')).toContainText(
      "Leaderboard and layer progression rank each model by its strongest available attention method."
    );
  });

  test('opens the Q2 analysis page from the dashboard link', async ({ page }) => {
    await stubDashboardApis(page);
    await page.route('**/api/metrics/q2_summary?**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          metric: 'iou',
          label: 'IoU',
          direction: 'higher',
          percentile_dependent: true,
          selected_percentile: 90,
          analyzed_layer: 11,
          timestamp: null,
          rows: [],
          strategy_comparisons: [],
        }),
      });
    });

    await page.goto('/dashboard');
    await expect(page.getByRole('link', { name: 'Q2 Analysis' })).toBeVisible();

    await page.getByRole('link', { name: 'Q2 Analysis' }).click();

    await expect(page).toHaveURL(/\/q2\?/);
    await expect(page.getByRole('heading', { name: 'Q2 Strategy-Aware Attention Shift' })).toBeVisible();
    await expect(page.getByText('No Q2 rows available for current filters.')).toBeVisible();
  });
});
