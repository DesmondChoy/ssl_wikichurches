import { expect, test } from '@playwright/test';

const IMAGE_ID = 'Q2034923_wd0.jpg';

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
});
