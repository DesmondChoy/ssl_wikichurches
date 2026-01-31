---
name: playwright-testing
description: Systematic Playwright MCP testing workflow for SSL Attention Visualization app with detailed checklists
triggers:
  - test the app
  - playwright testing
  - visual testing
  - run playwright
  - test frontend
---

# Playwright MCP Testing Guide

This skill provides a systematic checklist for visually testing and inspecting the SSL Attention Visualization app using Playwright MCP.

---

## Workflow Instructions

**IMPORTANT**: Follow these instructions exactly:

1. **Use the Checklist System**: Work through the structured checklists systematically. Do NOT skip items or test ad-hoc.

2. **Track Progress**: Use your task/todo management tools to track which checklist items have been completed. Mark items as you complete them.

3. **Sequential Testing**: Work through the phases in order (Phase 1 → Phase 2 → Phase 3 → etc.). Within each phase, complete all checklist items before moving to the next phase.

4. **Document Everything**: For each checklist item:
   - Take a `browser_snapshot` or `browser_take_screenshot` as evidence
   - Note whether the item passed or failed
   - If failed, stop and follow the Bug Handling Workflow below

5. **Report Format**: When reporting results, use this format:
   ```
   ✅ [Item description] - PASSED
   ❌ [Item description] - FAILED: [brief description of issue]
   ```

---

## Prerequisites

1. Start the development servers:
   ```bash
   ./dev.sh
   ```

2. Ensure Playwright MCP is configured in Claude Code

3. **App URLs:**
   - Frontend: http://localhost:5173
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

---

## Bug Handling Workflow

**CRITICAL**: When any bug or visual error is detected during testing, follow this workflow exactly. Do NOT continue testing until the bug is fixed and verified.

1. **STOP testing immediately** - Do not continue to the next checklist item
2. **Document the bug** - Note the page, steps to reproduce, and expected vs actual behavior
3. **Fix the bug** - Make the necessary code changes
4. **Restart the servers**:
   ```bash
   # Kill existing servers
   pkill -f "uvicorn"; pkill -f "vite"

   # Restart
   ./dev.sh
   ```
5. **Verify the fix with Playwright MCP**:
   - Navigate back to the same page/state where the bug occurred
   - Confirm the bug is resolved
   - Take a screenshot or snapshot as evidence
6. **Resume testing** from where you left off

This ensures bugs are caught and fixed immediately rather than accumulating a backlog of issues.

---

## Getting Started with Playwright MCP

### Navigation Commands
```
# Navigate to the app
mcp__playwright__browser_navigate → url: "http://localhost:5173"

# Take a snapshot (preferred over screenshot for accessibility)
mcp__playwright__browser_snapshot

# Take a screenshot for visual inspection
mcp__playwright__browser_take_screenshot

# Click an element (use ref from snapshot)
mcp__playwright__browser_click → element: "description", ref: "e123"

# Run custom JavaScript for complex interactions
mcp__playwright__browser_run_code → code: "async (page) => { ... }"
```

### Useful Patterns
```javascript
// Check for element visibility
await page.locator('text=Some Text').isVisible();

// Wait for element
await page.waitForSelector('selector');

// Get element count
await page.locator('.grid-item').count();

// Check network requests
await page.waitForResponse(response => response.url().includes('/api/'));
```

---

## Testing Workflow

### Phase 1: Navigation & Layout

#### Initial Load
- [ ] Page loads at http://localhost:5173 without console errors
- [ ] "SSL Attention" heading is visible in navigation
- [ ] Navigation bar renders with all links: Gallery, Compare, Dashboard
- [ ] Footer is visible with correct text

#### Navigation Links
- [ ] "Gallery" nav link is clickable and highlights when active
- [ ] "Compare" nav link navigates to /compare
- [ ] "Dashboard" nav link navigates to /dashboard
- [ ] "SSL Attention" logo/title returns to home

---

### Phase 2: Gallery Page (Home)

#### Layout
- [ ] Gallery page loads at "/"
- [ ] Page title or heading indicates gallery content
- [ ] Image grid/list container is visible

#### Image Display
- [ ] Images load and display correctly
- [ ] Image thumbnails are appropriately sized
- [ ] Loading states show while images fetch
- [ ] No broken image placeholders

#### Interaction
- [ ] Hovering over images shows visual feedback (if applicable)
- [ ] Clicking an image navigates to /image/:imageId detail page
- [ ] Pagination or infinite scroll works (if applicable)

#### API Integration
- [ ] Network requests to backend succeed (check for /api/ calls)
- [ ] Error states display gracefully if API fails

---

### Phase 3: Image Detail Page

#### Navigation
- [ ] Can reach page by clicking image in gallery
- [ ] URL shows correct pattern: /image/:imageId
- [ ] Back navigation returns to gallery

#### Content Display
- [ ] Full-size image displays correctly
- [ ] Image metadata/details are visible
- [ ] Attention visualization overlay displays (if applicable)

#### Attention Visualization
- [ ] Attention heatmap/overlay renders correctly
- [ ] Legend or color scale is visible (if applicable)
- [ ] Controls for toggling visualization work
- [ ] Different attention heads/layers can be selected (if applicable)

#### Annotations
- [ ] Expert annotations display correctly
- [ ] Annotation regions are highlighted
- [ ] Annotation labels/descriptions are readable

---

### Phase 4: Compare Page

#### Layout
- [ ] Compare page loads at "/compare"
- [ ] Page layout supports side-by-side comparison
- [ ] Selection controls are visible

#### Image Selection
- [ ] Can select first image for comparison
- [ ] Can select second image for comparison
- [ ] Selected images display clearly

#### Comparison Features
- [ ] Both images render side-by-side
- [ ] Attention patterns can be compared visually
- [ ] Sync/lock view controls work (if applicable)
- [ ] Difference visualization renders (if applicable)

---

### Phase 5: Dashboard Page

#### Layout
- [ ] Dashboard page loads at "/dashboard"
- [ ] Page has clear structure/sections
- [ ] Loading states show while data fetches

#### Statistics/Metrics
- [ ] Key metrics display correctly
- [ ] Numbers/values are formatted properly
- [ ] Charts/visualizations render (if applicable)

#### Data Visualization
- [ ] Graphs/charts are readable
- [ ] Axis labels and legends are present
- [ ] Interactive tooltips work (if applicable)
- [ ] Data appears consistent with expectations

---

### Phase 6: Responsive Design

#### Mobile Viewport (375px width)
- [ ] Navigation collapses to mobile menu (if applicable)
- [ ] Content remains readable and not cut off
- [ ] Images scale appropriately
- [ ] Touch targets are adequately sized

#### Tablet Viewport (768px width)
- [ ] Layout adapts appropriately
- [ ] Grid adjusts column count
- [ ] No horizontal scrolling required

#### Desktop Viewport (1280px width)
- [ ] Full layout displays correctly
- [ ] Maximum content width is respected
- [ ] Adequate whitespace and spacing

---

### Phase 7: Error Handling

#### Network Errors
- [ ] Disconnecting backend shows appropriate error state
- [ ] Error messages are user-friendly
- [ ] Retry mechanisms work (if applicable)

#### Invalid Routes
- [ ] Navigating to /invalid-route shows 404 or redirects
- [ ] Invalid image IDs handled gracefully

#### Edge Cases
- [ ] Empty states display when no data
- [ ] Very long text doesn't break layout
- [ ] Special characters render correctly

---

### Phase 8: Performance

- [ ] Initial page load feels responsive (< 3 seconds)
- [ ] Navigation between pages is instant
- [ ] Images lazy-load appropriately
- [ ] No visible layout shifts during load
- [ ] No memory leaks during extended use

---

## Quick Smoke Test Checklist

For rapid testing, verify these critical paths:

1. [ ] App loads at localhost:5173
2. [ ] Gallery page shows images
3. [ ] Can click image to view detail page
4. [ ] Image detail shows attention visualization
5. [ ] Compare page allows side-by-side view
6. [ ] Dashboard shows statistics/metrics
7. [ ] Navigation between all pages works
8. [ ] No console errors throughout

---

## Reporting Issues

When documenting bugs, include:
1. **Page/Route** where issue occurred
2. **Steps to reproduce**
3. **Expected behavior**
4. **Actual behavior**
5. **Screenshot or snapshot** (use `mcp__playwright__browser_take_screenshot`)
6. **Console messages** (use `mcp__playwright__browser_console_messages`)

---

## Notes

- Playwright's mouse simulation may not perfectly replicate human interaction
- Use `browser_run_code` for complex JavaScript interactions
- Always take a `browser_snapshot` before clicking to get accurate element refs
- The `browser_snapshot` tool is preferred over screenshots for accessibility testing

---

## Checklist Summary

When instructed to perform Playwright testing, follow this workflow:

```
1. Start dev servers (./dev.sh)
2. Navigate to http://localhost:5173
3. Work through phases sequentially:

   PHASE 1: Navigation & Layout
   └── Initial Load (4 items)
   └── Navigation Links (4 items)

   PHASE 2: Gallery Page
   └── Layout (3 items)
   └── Image Display (4 items)
   └── Interaction (4 items)
   └── API Integration (2 items)

   PHASE 3: Image Detail Page
   └── Navigation (3 items)
   └── Content Display (3 items)
   └── Attention Visualization (4 items)
   └── Annotations (3 items)

   PHASE 4: Compare Page
   └── Layout (3 items)
   └── Image Selection (3 items)
   └── Comparison Features (4 items)

   PHASE 5: Dashboard Page
   └── Layout (3 items)
   └── Statistics/Metrics (3 items)
   └── Data Visualization (4 items)

   PHASE 6: Responsive Design
   └── Mobile (4 items)
   └── Tablet (3 items)
   └── Desktop (3 items)

   PHASE 7: Error Handling
   └── Network Errors (3 items)
   └── Invalid Routes (2 items)
   └── Edge Cases (3 items)

   PHASE 8: Performance (5 items)

4. For each item: test → document result (✅/❌) → if failed, STOP and fix
5. Provide final summary report with all results
```

**Total checklist items**: ~70 items
