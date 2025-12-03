# RAG vs REFRAG Comparison Report

## Executive Summary

| Metric | Standard RAG | REFRAG |
|--------|--------------|--------|
| **Accuracy** | **100%** (15/15) | 40% (6/15) |
| **Avg Time/Query** | 8.27s | 7.54s |
| **Throughput** | 20.3 tok/s | **70.7 tok/s** |

## Key Findings

### 1. Accuracy
- **Standard RAG achieves 100% accuracy** on factual retrieval questions
- **REFRAG only achieves 40% accuracy** despite retrieving correct passages
- REFRAG often generates gibberish (e.g., "def main():", "definir", "definitely...")

### 2. Speed & Throughput
- REFRAG is **faster at generation** (70.7 vs 20.3 tok/s) due to context compression
- However, the speed advantage is meaningless if answers are wrong

### 3. Root Cause Analysis

REFRAG's poor performance stems from:

1. **Untrained Compression Layer**: The projector network that compresses encoder embeddings into decoder space needs extensive training to preserve semantic information.

2. **No Prompt Engineering**: Standard RAG uses a structured prompt:
   ```
   Use the following passages to answer the question...
   Passages: [context]
   Question: [query]
   Answer:
   ```
   REFRAG directly concatenates embeddings without guiding instructions.

3. **Training Data Volume**: 100 steps of training is insufficient for:
   - CPT reconstruction: Loss dropped from 9.76 to 5.99 (still high)
   - Policy training: Rewards are unstable/negative

4. **Information Loss**: Compressing a full passage (~200 tokens) into a single vector loses critical details.

## Detailed Results

### Sample Questions

| Question | Expected | RAG | REFRAG |
|----------|----------|-----|--------|
| What is the primary research institution in City_848? | University_072 | ✓ University_072 | ✗ "def main():" |
| In what year did Event_2803 occur? | 1919 | ✓ 1919 | ✗ "definitely in 1920" |
| In which field does Person_454 work? | Field_52 | ✓ Field_52 | ✗ "defining Contribution..." |
| Which country contains City_888? | Country_040 | ✓ Country_040 | ✓ Contains "Country_040" |

### Observations

1. **RAG answers are clean and accurate**: Always starts with the correct answer followed by explanation
2. **REFRAG answers are chaotic**: Often starts with code fragments, Spanish words, or incomplete sentences
3. **REFRAG occasionally works**: When it does find the answer, it's often embedded in garbled text

## Recommendations

### To Improve REFRAG:

1. **More Training**:
   - Increase CPT steps to 1000-5000
   - Train until reconstruction loss < 2.0
   - Use larger batch sizes if memory allows

2. **Add Instruction Tuning**:
   - Modify `build_decoder_inputs()` to prepend instruction tokens
   - Fine-tune on instruction-following data

3. **Improve Data Quality**:
   - Ensure training questions exactly match corpus entities
   - Add diverse question types beyond factual lookup

4. **Architecture Changes**:
   - Consider multi-vector representations instead of single-vector compression
   - Add attention over compressed chunks before generation

### When to Use Each System:

| Use Case | Recommendation |
|----------|----------------|
| Production QA system | **Standard RAG** |
| Research on compression | REFRAG (with more training) |
| Latency-critical applications | REFRAG (if accuracy improves) |
| Maximum accuracy required | **Standard RAG** |

## Test Configuration

```
Embedding Model: BAAI/bge-small-en-v1.5
Decoder Model: meta-llama/Llama-3.2-3B
Top-K Retrieval: 4
REFRAG Chunk Size (k): 32
REFRAG Expansion Fraction (p): 0.25
Test Samples: 15
```

## Files Created

| File | Description |
|------|-------------|
| `rag.py` | Standard RAG with Qdrant |
| `evaluate_comparison.py` | Comparison evaluation script |
| `rag_quickstart.sh` | Quick start script for RAG |
| `requirements.txt` | Dependencies for both systems |
| `data/rag_train_aligned.jsonl` | Aligned training data (5250 samples) |
| `data/rag_eval_test.jsonl` | Test set (1050 samples) |

## How to Reproduce

```bash
# Standard RAG
./rag_quickstart.sh

# REFRAG (with current training)
uv run python refrag.py generate \
  --index_dir runs/index \
  --load_dir runs/policy_aligned \
  --question "Which river flows through City_20?" \
  --topk 4 --k 32 --p 0.25

# Run comparison
uv run python evaluate_comparison.py \
  --test_json data/rag_eval_test.jsonl \
  --max_samples 20 \
  --output runs/comparison_results.json
```

## Conclusion

**Standard RAG is the clear winner** for production use cases. REFRAG shows promise for faster inference throughput but requires significantly more training and architectural improvements to be viable. The compression approach fundamentally trades accuracy for speed, and the current implementation hasn't found the right balance.
