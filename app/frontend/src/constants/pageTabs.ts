import type { PageTab } from '../types';

export const DEFAULT_PAGE_TAB: PageTab = 'main';

export function parsePageTab(value: string | null | undefined): PageTab {
  return value === 'q3' ? 'q3' : DEFAULT_PAGE_TAB;
}
