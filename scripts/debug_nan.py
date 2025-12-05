#!/usr/bin/env python3
"""Debug NaN issue in CPT training."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Force CPU to avoid MPS issues
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
import json

# Force CPU
torch.set_default_device('cpu')

from src.refrag_v2 import REFRAGConfig, REFRAGModel

def main():
    print("Loading model from reconstruction checkpoint (CPU mode)...")

    config = REFRAGConfig(
        encoder_name='roberta-large',
        decoder_name='meta-llama/Llama-3.2-3B',
        chunk_size_k=16,
    )

    # Create model on CPU
    model = REFRAGModel(config, device='cpu')

    # Load checkpoint
    load_dir = 'runs/refrag_v2_recon'
    model.load_checkpoint(load_dir)

    # Check projector weights for NaN
    print("\n=== Checking projector weights ===")
    for name, param in model.projector.named_parameters():
        has_nan = torch.isnan(param).any().item()
        has_inf = torch.isinf(param).any().item()
        print(f"  {name}: shape={param.shape}, nan={has_nan}, inf={has_inf}, "
              f"min={param.min().item():.6f}, max={param.max().item():.6f}")

    # Load a sample text
    print("\n=== Loading test data ===")
    with open('data/cpt_train.jsonl', 'r') as f:
        sample = json.loads(f.readline())

    if 'text' not in sample:
        text = model.decoder_tokenizer.decode(sample['tokens'][:512])
    else:
        text = sample['text']
    print(f"Sample text: {text[:200]}...")

    # Test forward pass
    print("\n=== Testing forward pass ===")
    model.eval()

    k = config.chunk_size_k

    # Tokenize
    input_ids = model._tokenize_text(text, 256)
    ids = input_ids.squeeze(0)

    # Split
    context_ids = ids[:128]
    output_ids = ids[128:256]

    print(f"Context tokens: {len(context_ids)}, Output tokens: {len(output_ids)}")

    # Chunk the context
    context_chunks = model._chunk_tokens(context_ids, k)
    chunk_texts = model._chunks_to_text(context_chunks)
    print(f"Number of chunks: {len(context_chunks)}")

    # Encode chunks
    print("\n=== Encoding chunks ===")
    chunk_embeddings = model.encoder(chunk_texts, device=model.device)
    print(f"Chunk embeddings: shape={chunk_embeddings.shape}")
    print(f"  nan={torch.isnan(chunk_embeddings).any().item()}")
    print(f"  inf={torch.isinf(chunk_embeddings).any().item()}")
    print(f"  min={chunk_embeddings.min().item():.6f}, max={chunk_embeddings.max().item():.6f}")
    print(f"  mean={chunk_embeddings.mean().item():.6f}, std={chunk_embeddings.std().item():.6f}")

    # Project
    print("\n=== Projecting ===")
    projected = model.projector(chunk_embeddings)
    print(f"Projected: shape={projected.shape}")
    print(f"  nan={torch.isnan(projected).any().item()}")
    print(f"  inf={torch.isinf(projected).any().item()}")
    print(f"  min={projected.min().item():.6f}, max={projected.max().item():.6f}")
    print(f"  mean={projected.mean().item():.6f}, std={projected.std().item():.6f}")

    # Get output embeddings
    print("\n=== Output embeddings ===")
    output_embs = model._get_decoder_embeddings(output_ids[:-1].unsqueeze(0).to(model.device))
    print(f"Output embeddings: shape={output_embs.shape}")
    print(f"  min={output_embs.min().item():.6f}, max={output_embs.max().item():.6f}")
    print(f"  mean={output_embs.mean().item():.6f}, std={output_embs.std().item():.6f}")

    # Concatenate
    print("\n=== Full input ===")
    context_embs = projected.unsqueeze(0)
    full_input = torch.cat([context_embs, output_embs], dim=1)
    print(f"Full input: shape={full_input.shape}")
    print(f"  nan={torch.isnan(full_input).any().item()}")
    print(f"  inf={torch.isinf(full_input).any().item()}")

    # Labels
    context_len = context_embs.size(1)
    labels = torch.cat([
        torch.full((context_len,), -100, device=model.device),
        output_ids.to(model.device)
    ]).unsqueeze(0)
    labels = labels[:, :full_input.size(1)]

    # Forward pass
    print("\n=== Decoder forward pass ===")
    with torch.no_grad():
        outputs = model.decoder(
            inputs_embeds=full_input,
            labels=labels
        )
    print(f"Loss: {outputs.loss.item()}")
    print(f"  loss is nan: {torch.isnan(outputs.loss).item()}")

    # Now test with gradient
    print("\n=== Testing with gradient (training mode) ===")
    model.train()

    loss = model.compute_cpt_loss(text, s=128, o=128, expand_fraction=0.0)
    print(f"CPT Loss: {loss.item()}")
    print(f"  loss is nan: {torch.isnan(loss).item()}")

    # Do backward pass
    print("\n=== Backward pass ===")
    loss.backward()

    # Check gradients
    print("\n=== Checking gradients ===")
    for name, param in model.projector.named_parameters():
        if param.grad is not None:
            has_nan = torch.isnan(param.grad).any().item()
            has_inf = torch.isinf(param.grad).any().item()
            print(f"  {name}: grad nan={has_nan}, inf={has_inf}, "
                  f"grad min={param.grad.min().item():.6f}, max={param.grad.max().item():.6f}")

    # Check decoder gradients
    print("\n=== Decoder gradients (first layer) ===")
    for name, param in list(model.decoder.named_parameters())[:5]:
        if param.grad is not None:
            has_nan = torch.isnan(param.grad).any().item()
            has_inf = torch.isinf(param.grad).any().item()
            print(f"  {name}: grad nan={has_nan}, inf={has_inf}")

    print("\n=== Done ===")

if __name__ == '__main__':
    main()
