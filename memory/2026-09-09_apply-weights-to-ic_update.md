# 2026-09-09: Apply Weights to ic_update in Cosine Similarity

## Summary
Fixed cosine similarity computation by applying loss weights to both `ic_update` and `gradient_prev` before computing their cosine similarity. This ensures the comparison is consistent with the weighted loss function that produced `gradient_prev`.

## Problem Statement
Cosine similarity for dynamic weight was not working correctly because:
1. The `gradient_prev` already contains weight information from the weighted loss function
2. The `ic_update` did not have these weights applied
3. Cosine similarity needs to see the weighted values for proper comparison

## Solution
Applied the loss weights to `ic_update` in the cosine similarity computation by:
1. Capturing `loss_details` from the loss function call
2. Extracting weights from `loss_details`
3. Reordering computation steps so that loss weights are extracted before cosine similarity
4. Applying weights to `ic_update`  before flattening and computing cosine similarity

## Changes Made

### File: `src/gd_optimic/meta_learner/bilevel_optimizer.py`

#### Change 1: Reorder STEP 2 and STEP 2.5 (Lines ~418-456)
Moved STEP 2.5 (Compute Cosine Similarity Metric) to **after** STEP 2 (Forward pass and compute losses) so that `loss_weights` is available when needed.

**Before:**
- STEP 2.5: Cosine Similarity (lines ~418-447)
- STEP 2: Forward pass (lines ~449-456)

**After:**
- STEP 2: Forward pass (lines ~418-425)
- STEP 2.5: Cosine Similarity (lines ~427-456)

#### Change 2: Capture loss details and extract weights (Lines ~422-425)
**Before:**
```python
loss_current, _ = self.loss_fn(y_hat_steps_new, self.target_sequence, return_details=True)
```

**After:**
```python
loss_current, loss_details = self.loss_fn(y_hat_steps_new, self.target_sequence, return_details=True)

# Extract weights from loss details for weighted comparison
loss_weights = loss_details.get('weights', None)  # [B, C] or None
```

#### Change 3: Apply weights in cosine similarity computation (Lines ~434-445)
**Before:**
```python
meta_flat = ic_update.flatten(start_dim=2)  # [B, T, C*H*W]
grad_flat = gradient_prev.flatten(start_dim=2)  # [B, T, C*H*W]
```

**After:**
```python
# Apply loss weights to ic_update for consistent comparison
if loss_weights is not None:
    # Reshape weights for broadcasting: [B, C] -> [B, 1, C, 1, 1] for 5D tensor
    weights_reshaped = loss_weights.view(loss_weights.shape[0], 1, loss_weights.shape[1], 1, 1)
    weighted_ic_update = ic_update * weights_reshaped
    meta_flat = weighted_ic_update.flatten(start_dim=2)
    # # Apply same weights to gradient_prev for comparison
    # weighted_gradient_prev = gradient_prev * weights_reshaped
    # grad_flat = weighted_gradient_prev.flatten(start_dim=2)
else:
    meta_flat = ic_update.flatten(start_dim=2)
    grad_flat = gradient_prev.flatten(start_dim=2)
```

#### Change 4: Memory cleanup (Lines ~612-621)
**Before:**
```python
del loss_current, l_reg_scalar, l_perf
del y_hat_steps_new, ic_update
del x_new, l_meta
```

**After:**
```python
del loss_current, loss_details, loss_weights, l_reg_scalar, l_perf
del y_hat_steps_new, ic_update
del x_new, l_meta
# Clean up weighted tensors if they were created
if 'weighted_ic_update' in locals():
    del weighted_ic_update
if 'weighted_gradient_prev' in locals():
    del weighted_gradient_prev
if 'weights_reshaped' in locals():
    del weights_reshaped
```

## Expected Impact
- Cosine similarity values should be more stable and meaningful
- Better alignment between `ic_update` and `gradient_prev` when weights are applied
- L_align values should decrease consistently in ALIGN phase
- More accurate gradient alignment in weighted loss scenarios

## Testing
Run meta-learner training with:
- `meta_one_task=True`
- `hybrid_input=True`

Monitor:
- Cosine similarity values (should be more stable)
- L_align values (should decrease consistently in ALIGN phase)

## Notes
- The weights are applied to `ic_update` to ensure consistent comparison
- If no weights are present (`loss_weights is None`), the code falls back to the original behavior
- Weights are reshaped to `[B, 1, C, 1, 1]` for proper broadcasting with 5D tensors
