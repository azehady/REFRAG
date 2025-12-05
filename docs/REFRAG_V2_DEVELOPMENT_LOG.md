# REFRAG v2 Development Log

This document summarizes the development process of REFRAG v2, including all issues encountered, their root causes, and the fixes applied.

## Overview

REFRAG v2 implements a "Compress → Sense → Expand" framework for optimized retrieval-augmented generation:

1. **Compress**: Encode retrieved passages into dense chunk embeddings using RoBERTa-Large
2. **Sense**: Project encoder embeddings to decoder space via a learned projection layer
3. **Expand**: Selectively expand compressed tokens back to full text using a learned policy

The architecture consists of:
- **ChunkEncoder**: RoBERTa-Large for encoding text chunks
- **ProjectionLayer**: 2-layer MLP (encoder_dim → 4*decoder_dim → decoder_dim)
- **ExpansionPolicy**: MLP classifier for determining which chunks to expand
- **Decoder**: Llama-3.2-3B for generation

Training follows a 3-stage curriculum:
1. **Reconstruction**: Train encoder + projector to reconstruct text (decoder frozen)
2. **CPT (Continual Pre-Training)**: Fine-tune full model on next-token prediction
3. **Policy**: Train expansion policy with REINFORCE

---

## Issues and Fixes

### Issue 1: CPT Loss = 0.0000 (Not Training)

**Symptoms:**
- CPT training showed `Loss: 0.0000` immediately after step 10
- Training completed in seconds instead of minutes
- No actual learning was happening

**Root Cause:**
The `train_cpt` function was checking if individual batch losses were NaN/Inf and skipping them. However, ALL batches were being skipped due to NaN issues, resulting in zero accumulated loss.

```python
# Original code (problematic)
if torch.isnan(loss) or torch.isinf(loss):
    continue  # Silently skipped ALL batches
```

**Investigation:**
Added debug logging to trace the issue:
```python
logger.debug(f"CPT loss: {loss.item()}, context_len={context_len}, output_len={len(output_ids)-1}")
if torch.isnan(loss) or torch.isinf(loss):
    logger.debug("CPT skipped: NaN/Inf in projected embeddings")
```

**Initial Misdiagnosis:**
Initially thought the issue was that training texts were too short, causing empty context windows. This was partially true but not the root cause.

---

### Issue 2: NaN Loss After First Optimizer Step

**Symptoms:**
- First batch processed correctly with valid loss (~3.8)
- After first `optimizer.step()`, all subsequent losses became NaN
- Policy network weights became NaN

**Root Cause:**
**fp16 precision overflow** when passing custom `inputs_embeds` to the LLaMA decoder.

The issue manifests as follows:
1. Projector outputs fp32 embeddings (reasonable magnitude)
2. Decoder operates in fp16 mode (`torch.float16`)
3. During forward pass, embeddings are automatically converted to fp16
4. After backward pass and optimizer step, gradients overflow in fp16 precision
5. This causes NaN weights and subsequent NaN losses

**Attempted Fixes (Failed):**
1. ✗ Reduced learning rates (5e-5 → 1e-5, 5e-7)
2. ✗ Added gradient clipping (max_norm=1.0)
3. ✗ Removed normalization from projector output
4. ✗ Froze encoder/projector during CPT
5. ✗ Added dtype conversion with clamping to fp16 range

**Fix (Partial - dtype conversion):**
```python
# In compute_cpt_loss
decoder_dtype = next(self.decoder.parameters()).dtype
if decoder_dtype == torch.float16:
    context_embs = context_embs.clamp(-65000, 65000)
context_embs = context_embs.to(dtype=decoder_dtype)
```

This helped but didn't fully solve the problem.

**Fix (Final - Disable fp16):**
```python
# In REFRAGConfig class (src/refrag_v2.py:172)
@dataclass
class REFRAGConfig:
    # ...
    # Hardware
    fp16: bool = False  # Disabled fp16 to fix NaN issues during CPT training
```

**Location:** `src/refrag_v2.py:172`

---

### Issue 3: Index File Not Found

**Symptoms:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'runs/index/faiss_passages.npy'
```

**Root Cause:**
The `load_index` function was looking for `faiss_passages.npy` but the actual file was saved as `texts.npy` during index building.

**Fix:**
```python
# In load_index method
# Try both possible filenames for backward compatibility
passages_path = index_path.replace('.index', '_passages.npy')
if not os.path.exists(passages_path):
    passages_path = os.path.join(os.path.dirname(index_path), 'texts.npy')
self.passages = np.load(passages_path, allow_pickle=True).tolist()
```

**Location:** `src/refrag_v2.py:378-382`

---

### Issue 4: MPS Backend Memory Alignment Error

**Symptoms:**
When running debug scripts on Apple Silicon (MPS):
```
RuntimeError: destOffset % 4 == 0 INTERNAL ASSERT FAILED
```

**Root Cause:**
Known PyTorch bug with MPS backend and certain tensor operations during model loading.

**Workaround:**
Force CPU execution in debug scripts:
```python
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
torch.set_default_device('cpu')
```

This wasn't needed after finding the fp16 root cause.

---

### Issue 5: Text Too Short for Context Window

**Symptoms:**
- Empty context chunks when text is shorter than expected
- Training crashes or produces zero loss

**Root Cause:**
Training data texts shorter than the context window size (128 tokens) resulted in empty context, causing invalid loss calculations.

**Fix:**
Added minimum length validation in `compute_cpt_loss`:
```python
if len(ids) < context_size + output_size:
    return None  # Skip short texts gracefully
```

---

## Current Status

After applying all fixes:

**Reconstruction Training:** ✅ Working
- Loss decreases from ~6.3 to ~5.3 over 300 steps
- Curriculum learning progresses through 3 stages (1→2→4 chunks)

**CPT Training:** ✅ Working
- Loss decreases from ~3.8 to ~0.5 over 300 steps
- Individual batch losses decrease from ~3.8 to ~0.5
- Proper gradient flow through decoder

**Remaining Work:**
- Policy training not yet tested with fixed code
- Evaluation comparison pending

---

## Training Results (After Fixes)

### Reconstruction Training (100 steps per stage, 3 stages)
```
Stage 1 (max_chunks=1): Loss 6.29 → 6.34
Stage 2 (max_chunks=2): Loss 5.98 → 5.77
Stage 3 (max_chunks=4): Loss 5.49 → 5.30
```

### CPT Training (100 steps per stage, 3 stages)
```
Stage 1 (expand_frac=0.05): Loss 0.1052 → 0.0166 (batch loss: 3.8 → 0.5)
Stage 2 (expand_frac=0.10): Loss 0.0165 → 0.0147
Stage 3 (expand_frac=0.15): Loss 0.0152 → 0.0141
```

---

## Key Learnings

1. **fp16 training with custom embeddings is tricky**: When using `inputs_embeds` instead of `input_ids`, the automatic mixed precision can cause gradient overflow.

2. **Silent failure modes are dangerous**: The original code silently skipped NaN batches without logging, making the 0.0 loss hard to debug.

3. **Debug logging is essential**: Adding verbose debug logging revealed the NaN pattern immediately.

4. **File naming consistency matters**: The index build/load mismatch caused unnecessary errors.

5. **Curriculum learning helps**: The staged approach (1→2→4 chunks for reconstruction, 0.05→0.10→0.15 expansion for CPT) provides stable training.

---

## Files Modified

1. **`src/refrag_v2.py`**:
   - Line 172: Changed `fp16: bool = True` to `fp16: bool = False`
   - Lines 763-769: Added dtype conversion in `compute_reconstruction_loss`
   - Lines 870-875: Added dtype conversion in `compute_cpt_loss`
   - Lines 378-382: Fixed index file loading path

2. **`scripts/refrag_v2_quickstart.sh`**: Created convenience script for running full pipeline

3. **`scripts/debug_nan.py`**: Created diagnostic script for NaN debugging

---

## Improvement Plans

1. **Enable fp16 with proper mixed precision**: Use `torch.cuda.amp.autocast()` properly or use `bfloat16` which has better dynamic range

2. **Add gradient checkpointing**: Reduce memory usage for larger context windows

3. **Implement proper evaluation metrics**: Beyond accuracy, measure compression ratio, latency, and generation quality

4. **Add early stopping**: Monitor validation loss and stop training when converged

5. **Hyperparameter tuning**: Grid search for optimal learning rates, batch sizes, and curriculum stages

6. **Support for longer contexts**: Currently limited by chunk size and max_chunks parameters
