# Plan: Per-Head Attention Visualization

## Overview

Add UI to visualize attention from individual transformer heads (0-11) in addition to the current "fused" (averaged) view. This reveals head specialization patterns documented in DINO literature.

## Scope

- **Supported:** CLS and Mean attention methods on ViT models (DINOv2, DINOv3, MAE, CLIP, SigLIP)
- **Not supported:** Rollout (complex to decompose), Grad-CAM (ResNet-50 is a CNN)
- **All ViTs have 12 attention heads**

## Implementation Phases

### Phase 1: Backend Core

**1.1 Add backend config** (`app/backend/config.py`)
```python
MODEL_NUM_HEADS = {"dinov2": 12, "dinov3": 12, "mae": 12, "clip": 12, "siglip": 12, "resnet50": 0}
PER_HEAD_METHODS = {"cls", "mean"}
```

**1.2 Add per-head mean attention function** (`src/ssl_attention/attention/cls_attention.py`)
- Add `get_per_head_mean_attention()` similar to existing `get_per_head_attention()`

**1.3 Update cache generation** (`app/precompute/generate_attention_cache.py`)
- Add `--per-head` flag
- Store variants: `cls_head0`, `cls_head1`, ... `cls_head11` (same for mean)
- Storage: ~40MB additional (acceptable)

**1.4 Extend attention service** (`app/backend/services/attention_service.py`)
- Add method to load per-head attention from cache

**1.5 Update API endpoint** (`app/backend/routers/attention.py`)
- Add `head: int | None` query parameter to `GET /{image_id}/raw`
- Validate: head only valid for CLS/Mean methods, ViT models only
- Update `/models` endpoint to return `num_heads_per_model` and `per_head_methods`

### Phase 2: Frontend Core

**2.1 Update types** (`app/frontend/src/types/index.ts`)
- Add `num_heads_per_model` and `per_head_methods` to ModelsResponse

**2.2 Update viewStore** (`app/frontend/src/store/viewStore.ts`)
- Add `head: number | null` state (null = fused)
- Add `numHeadsPerModel` and `perHeadMethods` from API
- Add `setHead()` action
- Reset head to null when method changes to non-supporting method

**2.3 Update ControlPanel** (`app/frontend/src/components/attention/ControlPanel.tsx`)
- Add "Attention Head" dropdown
- Options: "All (Fused)", "Head 0", "Head 1", ... "Head 11"
- Hide dropdown for Rollout method and ResNet-50

**2.4 Update API client** (`app/frontend/src/api/client.ts`)
- Add `head` parameter to `getRawAttention()`

**2.5 Update AttentionViewer** (`app/frontend/src/components/attention/AttentionViewer.tsx`)
- Add `head` to query key and API call

**2.6 Add glossary entry** (`app/frontend/src/constants/glossary.ts`)
- Explain attention head specialization

### Phase 3: Cache Generation & Testing

**3.1 Run cache generation**
```bash
python -m app.precompute.generate_attention_cache --models all --per-head
```

**3.2 Test backend**
- Test head parameter validation
- Test per-head data retrieval

**3.3 Test frontend**
- Test head selector visibility logic
- Test head reset on method change

### Phase 4: Documentation

**4.1 Create feature docs** (`docs/per_head_attention.md`)
- Document feature, supported models/methods, API

## Critical Files to Modify

| File | Changes |
|------|---------|
| `app/backend/config.py` | Add MODEL_NUM_HEADS, PER_HEAD_METHODS |
| `src/ssl_attention/attention/cls_attention.py` | Add get_per_head_mean_attention() |
| `app/precompute/generate_attention_cache.py` | Add --per-head flag, store per-head variants |
| `app/backend/services/attention_service.py` | Add per-head loading method |
| `app/backend/routers/attention.py` | Add head parameter, update /models response |
| `app/frontend/src/types/index.ts` | Extend ModelsResponse type |
| `app/frontend/src/store/viewStore.ts` | Add head state and actions |
| `app/frontend/src/components/attention/ControlPanel.tsx` | Add head selector dropdown |
| `app/frontend/src/api/client.ts` | Add head to getRawAttention() |
| `app/frontend/src/components/attention/AttentionViewer.tsx` | Pass head to API |

## Verification

1. **Backend:** `pytest tests/` - all tests pass
2. **Cache:** Run precompute with `--per-head` flag
3. **Frontend:** `cd app/frontend && npm run build` - no errors
4. **E2E:** Start app with `./dev.sh`, select different heads, verify heatmaps change
5. **Edge cases:**
   - Verify head selector hidden for Rollout
   - Verify head selector hidden for ResNet-50
   - Verify head resets to "All" when switching to Rollout