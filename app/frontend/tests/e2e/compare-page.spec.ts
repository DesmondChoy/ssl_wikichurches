import { expect, test } from '@playwright/test';

const IMAGE_ID = 'Q2034923_wd0.jpg';

test.describe('Compare page', () => {
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
    await expect(page.getByText(/KL:/)).toHaveCount(2);
  });
});
