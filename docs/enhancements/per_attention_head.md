# Per-Head Attention Visualization

> **Enhancement proposed:** January 2026
> **Status:** Planned
> **Purpose:** Reveal attention head specialization patterns documented in DINO literature

> **Related documents:**
> - [Project Proposal — Q3: Head Specialization](../core/project_proposal.md#research-questions-and-approaches)
> - [Attention Methods — Head Fusion Strategies](../research/attention_methods.md#head-fusion-strategies)
> - [Implementation Plan](../core/implementation_plan.md)

## Executive Summary

This enhancement adds UI controls to visualize attention from individual transformer heads (0-11) instead of only the current "fused" (averaged) view. Research shows that different attention heads in Vision Transformers specialize for different visual patterns—some focus on texture, others on object boundaries, and others on semantic regions.

**Key insight:** By examining per-head attention, users can understand *which* heads align with expert-annotated architectural features, not just whether the model as a whole attends to them.

---

## 1. Motivation

### Why Per-Head Visualization Matters

Vision Transformers contain multiple attention heads that learn to focus on different aspects of the input:

| Head Type | Typical Behavior | Relevance to Architecture |
|-----------|------------------|---------------------------|
| **Low-level heads** | Edges, textures, local patterns | Stone textures, brick patterns |
| **Mid-level heads** | Object parts, boundaries | Window frames, arch outlines |
| **High-level heads** | Semantic regions, whole objects | Entire towers, facades |

### Evidence from DINO Literature

The original DINO paper (Caron et al., 2021) demonstrated that self-supervised ViTs develop heads with distinct specializations:

- Some heads produce attention maps that closely resemble semantic segmentation
- Different heads activate for foreground vs. background
- Head specialization emerges without any supervised signal

**Implication for this project:** If certain heads consistently align better with expert bounding boxes, that reveals which learned representations capture architectural semantics.

---

## 2. Scope

### Supported Configurations

| Model | Attention Method | Per-Head Support | Reason |
|-------|------------------|------------------|--------|
| DINOv2 | CLS | ✅ Yes | Direct head access |
| DINOv2 | Mean | ✅ Yes | Direct head access |
| DINOv2 | Rollout | ❌ No | Complex multi-layer aggregation |
| DINOv3 | CLS | ✅ Yes | Direct head access |
| DINOv3 | Mean | ✅ Yes | Direct head access |
| DINOv3 | Rollout | ❌ No | Complex multi-layer aggregation |
| MAE | CLS | ✅ Yes | Direct head access |
| MAE | Mean | ✅ Yes | Direct head access |
| MAE | Rollout | ❌ No | Complex multi-layer aggregation |
| CLIP | CLS | ✅ Yes | Direct head access |
| CLIP | Mean | ✅ Yes | Direct head access |
| CLIP | Rollout | ❌ No | Complex multi-layer aggregation |
| SigLIP | Mean | ✅ Yes | Direct head access (no CLS token) |
| ResNet-50 | Grad-CAM | ❌ No | CNN architecture, no attention heads |

### Technical Constraints

- **All ViT-Base models have 12 attention heads** (index 0-11)
- **Rollout cannot be decomposed** — it accumulates attention across all layers and heads, making per-head visualization mathematically undefined
- **Grad-CAM is gradient-based** — ResNet-50 uses convolutional layers, not attention mechanisms

---

## 3. Implementation Phases

### Phase 1: Backend Core

#### 1.1 Add Backend Configuration

**File:** `app/backend/config.py`

```python
# Number of attention heads per model (0 = not applicable)
MODEL_NUM_HEADS = {
    "dinov2": 12,
    "dinov3": 12,
    "mae": 12,
    "clip": 12,
    "siglip": 12,
    "resnet50": 0,  # CNN, no attention heads
}

# Methods that support per-head visualization
PER_HEAD_METHODS = {"cls", "mean"}
```

#### 1.2 Add Per-Head Mean Attention Function

**File:** `src/ssl_attention/attention/cls_attention.py`

```python
def get_per_head_mean_attention(
    attention_weights: torch.Tensor,
    head_index: int,
    num_registers: int = 0,
) -> torch.Tensor:
    """
    Extract mean attention for a specific head.

    Args:
        attention_weights: Shape [batch, heads, seq_len, seq_len]
        head_index: Which head to extract (0-11 for ViT-Base)
        num_registers: Number of register tokens to skip

    Returns:
        Attention map for the specified head [batch, num_patches]
    """
    # Similar implementation to get_per_head_attention()
    # but using mean pooling instead of CLS token
    ...
```

**Why this is needed:** The existing `get_per_head_attention()` extracts CLS-to-patch attention for a single head. We need an equivalent for Mean attention (averaging all patch-to-patch attention for a single head).

#### 1.3 Update Cache Generation

**File:** `app/precompute/generate_attention_cache.py`

Add `--per-head` flag to generate per-head variants:

```bash
python -m app.precompute.generate_attention_cache --models all --per-head
```

**Storage format:**
```
cache/attention/
├── dinov2/
│   ├── image_001_cls.npy          # Fused (existing)
│   ├── image_001_cls_head0.npy    # Per-head (new)
│   ├── image_001_cls_head1.npy
│   ├── ...
│   ├── image_001_cls_head11.npy
│   ├── image_001_mean.npy         # Fused (existing)
│   ├── image_001_mean_head0.npy   # Per-head (new)
│   └── ...
```

**Storage estimate:** ~40MB additional (12 heads × 2 methods × 139 images × ~12KB each) — acceptable overhead.

#### 1.4 Extend Attention Service

**File:** `app/backend/services/attention_service.py`

```python
def get_attention(
    self,
    image_id: str,
    model: str,
    method: str,
    head: int | None = None,  # NEW: None = fused, 0-11 = specific head
) -> np.ndarray:
    """Load attention map, optionally for a specific head."""
    if head is not None:
        # Validate head is supported for this model/method
        if method not in PER_HEAD_METHODS:
            raise ValueError(f"Per-head not supported for {method}")
        if MODEL_NUM_HEADS.get(model, 0) == 0:
            raise ValueError(f"Per-head not supported for {model}")
        cache_key = f"{image_id}_{method}_head{head}"
    else:
        cache_key = f"{image_id}_{method}"

    return self._load_from_cache(model, cache_key)
```

#### 1.5 Update API Endpoint

**File:** `app/backend/routers/attention.py`

```python
@router.get("/{image_id}/raw")
async def get_raw_attention(
    image_id: str,
    model: str,
    method: str,
    head: int | None = Query(None, ge=0, le=11),  # NEW parameter
) -> AttentionResponse:
    """
    Get raw attention map for an image.

    Args:
        image_id: Image identifier
        model: Model name (dinov2, clip, etc.)
        method: Attention method (cls, mean, rollout, gradcam)
        head: Optional attention head index (0-11).
              Only valid for CLS/Mean methods on ViT models.
    """
    # Validation
    if head is not None:
        if method not in PER_HEAD_METHODS:
            raise HTTPException(400, f"head parameter not supported for {method}")
        if MODEL_NUM_HEADS.get(model, 0) == 0:
            raise HTTPException(400, f"head parameter not supported for {model}")

    attention = attention_service.get_attention(image_id, model, method, head)
    return AttentionResponse(attention=attention.tolist())
```

**Update `/models` endpoint response:**

```python
@router.get("/models")
async def get_models() -> ModelsResponse:
    return ModelsResponse(
        models=SUPPORTED_MODELS,
        methods=ATTENTION_METHODS,
        num_heads_per_model=MODEL_NUM_HEADS,      # NEW
        per_head_methods=list(PER_HEAD_METHODS),  # NEW
    )
```

---

### Phase 2: Frontend Core

#### 2.1 Update Types

**File:** `app/frontend/src/types/index.ts`

```typescript
interface ModelsResponse {
  models: string[];
  methods: string[];
  num_heads_per_model: Record<string, number>;  // NEW
  per_head_methods: string[];                    // NEW
}
```

#### 2.2 Update View Store

**File:** `app/frontend/src/store/viewStore.ts`

```typescript
interface ViewState {
  // Existing state...
  model: string;
  method: string;

  // NEW state
  head: number | null;  // null = fused, 0-11 = specific head
  numHeadsPerModel: Record<string, number>;
  perHeadMethods: string[];
}

const useViewStore = create<ViewState>((set, get) => ({
  // ... existing state
  head: null,
  numHeadsPerModel: {},
  perHeadMethods: [],

  // NEW action
  setHead: (head: number | null) => set({ head }),

  // MODIFIED: Reset head when method changes to unsupported
  setMethod: (method: string) => {
    const { perHeadMethods, head } = get();
    const newHead = perHeadMethods.includes(method) ? head : null;
    set({ method, head: newHead });
  },

  // MODIFIED: Reset head when model changes to unsupported
  setModel: (model: string) => {
    const { numHeadsPerModel, head } = get();
    const supportsHead = (numHeadsPerModel[model] ?? 0) > 0;
    const newHead = supportsHead ? head : null;
    set({ model, head: newHead });
  },
}));
```

#### 2.3 Update Control Panel

**File:** `app/frontend/src/components/attention/ControlPanel.tsx`

```tsx
function HeadSelector() {
  const { model, method, head, setHead, numHeadsPerModel, perHeadMethods } = useViewStore();

  // Determine if per-head is available
  const numHeads = numHeadsPerModel[model] ?? 0;
  const methodSupportsHead = perHeadMethods.includes(method);
  const showHeadSelector = numHeads > 0 && methodSupportsHead;

  if (!showHeadSelector) return null;

  const options = [
    { value: null, label: "All (Fused)" },
    ...Array.from({ length: numHeads }, (_, i) => ({
      value: i,
      label: `Head ${i}`,
    })),
  ];

  return (
    <Select
      label="Attention Head"
      value={head}
      onChange={setHead}
      options={options}
    />
  );
}
```

**UI behavior:**
- Dropdown shows "All (Fused)", "Head 0", "Head 1", ... "Head 11"
- Hidden when Rollout method is selected
- Hidden when ResNet-50 is selected
- Resets to "All (Fused)" when switching to unsupported configuration

#### 2.4 Update API Client

**File:** `app/frontend/src/api/client.ts`

```typescript
export async function getRawAttention(
  imageId: string,
  model: string,
  method: string,
  head: number | null = null,  // NEW parameter
): Promise<number[][]> {
  const params = new URLSearchParams({ model, method });
  if (head !== null) {
    params.append("head", head.toString());
  }

  const response = await fetch(`/api/attention/${imageId}/raw?${params}`);
  return response.json();
}
```

#### 2.5 Update Attention Viewer

**File:** `app/frontend/src/components/attention/AttentionViewer.tsx`

```tsx
function AttentionViewer({ imageId }: Props) {
  const { model, method, head } = useViewStore();

  const { data: attention } = useQuery({
    queryKey: ["attention", imageId, model, method, head],  // Include head in cache key
    queryFn: () => getRawAttention(imageId, model, method, head),
  });

  // ... rest of component
}
```

#### 2.6 Add Glossary Entry

**File:** `app/frontend/src/constants/glossary.ts`

```typescript
export const GLOSSARY = {
  // ... existing entries

  attentionHead: {
    term: "Attention Head",
    definition: `Vision Transformers use multiple attention heads (12 in ViT-Base) that
    learn to focus on different visual patterns. Some heads specialize in edges, others
    in textures, and others in semantic regions. Viewing individual heads reveals which
    learned representations align with specific architectural features.`,
    relatedTerms: ["CLS Attention", "Multi-Head Attention"],
  },
};
```

---

### Phase 3: Cache Generation & Testing

#### 3.1 Generate Per-Head Cache

```bash
# Generate per-head attention maps for all supported models
python -m app.precompute.generate_attention_cache \
  --models dinov2 dinov3 mae clip siglip \
  --methods cls mean \
  --per-head

# Verify storage
du -sh cache/attention/*/
```

**Expected output:**
```
120M    cache/attention/dinov2/
120M    cache/attention/dinov3/
120M    cache/attention/mae/
120M    cache/attention/clip/
60M     cache/attention/siglip/   # Mean only
```

#### 3.2 Backend Testing

```python
# tests/test_attention_api.py

def test_per_head_parameter_validation():
    """Test that head parameter is validated correctly."""
    # Valid: CLS method with head
    response = client.get("/api/attention/img001/raw?model=dinov2&method=cls&head=5")
    assert response.status_code == 200

    # Invalid: Rollout method with head
    response = client.get("/api/attention/img001/raw?model=dinov2&method=rollout&head=5")
    assert response.status_code == 400

    # Invalid: ResNet with head
    response = client.get("/api/attention/img001/raw?model=resnet50&method=gradcam&head=5")
    assert response.status_code == 400

    # Invalid: Head out of range
    response = client.get("/api/attention/img001/raw?model=dinov2&method=cls&head=15")
    assert response.status_code == 422  # Validation error

def test_per_head_data_differs():
    """Test that different heads return different attention maps."""
    head0 = client.get("/api/attention/img001/raw?model=dinov2&method=cls&head=0").json()
    head5 = client.get("/api/attention/img001/raw?model=dinov2&method=cls&head=5").json()

    # Attention maps should differ between heads
    assert head0 != head5
```

#### 3.3 Frontend Testing

```typescript
// tests/HeadSelector.test.tsx

describe("HeadSelector", () => {
  it("shows selector for CLS method on DINOv2", () => {
    render(<ControlPanel />, { model: "dinov2", method: "cls" });
    expect(screen.getByLabelText("Attention Head")).toBeVisible();
  });

  it("hides selector for Rollout method", () => {
    render(<ControlPanel />, { model: "dinov2", method: "rollout" });
    expect(screen.queryByLabelText("Attention Head")).not.toBeInTheDocument();
  });

  it("hides selector for ResNet-50", () => {
    render(<ControlPanel />, { model: "resnet50", method: "gradcam" });
    expect(screen.queryByLabelText("Attention Head")).not.toBeInTheDocument();
  });

  it("resets head when switching to Rollout", () => {
    const store = useViewStore.getState();
    store.setHead(5);
    store.setMethod("rollout");
    expect(store.head).toBe(null);
  });
});
```

---

### Phase 4: Documentation

#### 4.1 Create User Documentation

**File:** `docs/features/per_head_attention.md`

Document:
- What attention heads are and why they matter
- How to use the head selector in the UI
- Interpretation guide for different head patterns
- Known limitations (Rollout, Grad-CAM)

---

## 4. Critical Files Summary

| File | Changes |
|------|---------|
| `app/backend/config.py` | Add `MODEL_NUM_HEADS`, `PER_HEAD_METHODS` constants |
| `src/ssl_attention/attention/cls_attention.py` | Add `get_per_head_mean_attention()` function |
| `app/precompute/generate_attention_cache.py` | Add `--per-head` flag, store per-head variants |
| `app/backend/services/attention_service.py` | Add per-head loading logic |
| `app/backend/routers/attention.py` | Add `head` query parameter, update `/models` response |
| `app/frontend/src/types/index.ts` | Extend `ModelsResponse` type |
| `app/frontend/src/store/viewStore.ts` | Add `head` state, reset logic |
| `app/frontend/src/components/attention/ControlPanel.tsx` | Add head selector dropdown |
| `app/frontend/src/api/client.ts` | Add `head` parameter to `getRawAttention()` |
| `app/frontend/src/components/attention/AttentionViewer.tsx` | Include `head` in query key |
| `app/frontend/src/constants/glossary.ts` | Add attention head definition |

---

## 5. Verification Checklist

```
┌─────────────────────────────────────────────────────────────┐
│  VERIFICATION CHECKLIST                                     │
├─────────────────────────────────────────────────────────────┤
│  □ Backend tests pass: pytest tests/                        │
│  □ Cache generated: --per-head flag works                   │
│  □ Frontend builds: npm run build (no errors)               │
│  □ E2E test: ./dev.sh, select heads, heatmaps change        │
├─────────────────────────────────────────────────────────────┤
│  EDGE CASES                                                 │
│  □ Head selector hidden for Rollout method                  │
│  □ Head selector hidden for ResNet-50                       │
│  □ Head resets to "All (Fused)" when switching to Rollout   │
│  □ Head resets when switching to ResNet-50                  │
│  □ Head persists when switching between CLS and Mean        │
│  □ Invalid head values (>11) rejected by API                │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. Future Extensions

### Per-Head IoU Analysis

Once per-head visualization is implemented, extend the metrics pipeline:

```python
def compute_per_head_iou(image_id: str, model: str) -> dict[int, float]:
    """Compute IoU for each attention head separately."""
    results = {}
    for head in range(12):
        attention = get_attention(image_id, model, "cls", head=head)
        iou = compute_iou(attention, get_bboxes(image_id))
        results[head] = iou
    return results
```

**Research questions this enables:**
- Which heads align best with expert annotations?
- Do some heads specialize in specific architectural feature types?
- Does head specialization differ between DINO, CLIP, and MAE?

### Head Specialization Clustering

Analyze which heads share similar attention patterns:

```python
def cluster_heads_by_pattern(model: str) -> list[set[int]]:
    """Group heads with similar attention patterns across dataset."""
    # Compute pairwise similarity between heads
    # Cluster into groups (e.g., "edge heads", "semantic heads")
    ...
```

---

## 7. References

### DINO Head Specialization
- Caron, M., et al. (2021). [Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294) — Original DINO paper showing head specialization
- Darcet, T., et al. (2023). [Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588) — Analysis of attention artifacts and head behavior

### Multi-Head Attention Interpretation
- Voita, E., et al. (2019). [Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting](https://arxiv.org/abs/1905.09418) — NLP analysis applicable to ViT
- Clark, K., et al. (2019). [What Does BERT Look At?](https://arxiv.org/abs/1906.04341) — Attention head analysis methodology

### Attention Visualization
- Abnar, S., & Zuidema, W. (2020). [Quantifying Attention Flow in Transformers](https://arxiv.org/abs/2005.00928) — Attention rollout method
- Chefer, H., et al. (2021). [Transformer Interpretability Beyond Attention Visualization](https://arxiv.org/abs/2012.09838) — Advanced attention analysis
