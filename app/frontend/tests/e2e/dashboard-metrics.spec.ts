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
});
