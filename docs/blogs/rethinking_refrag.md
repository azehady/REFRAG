Blog URL: https://ai.plainenglish.io/rethinking-rag-based-decoding-refrag-e38ce07a6021

Rethinking RAG-based Decoding (REFRAG)
DhanushKumar
DhanushKumar

Follow
12 min read
·
Oct 12, 2025
3






Traditional Retrieval-Augmented Generation (RAG) feeds all retrieved passages (as token sequences) into the LLM’s context. REFRAG alters this by compressing most retrieved chunks into single embeddings before decoding. Instead of inputting thousands of token embeddings, REFRAG uses a lightweight encoder (e.g. RoBERTa) to map each fixed‐size chunk to a vector, then projects that vector into the LLM’s token-embedding space.This yields three core benefits :

Shorter input: Each chunk becomes one vector, drastically reducing sequence length.
Reuse of computation: The chunk embeddings can be precomputed during retrieval and reused, avoiding redundant encoding.
Sparser attention: The LLM’s cross-attention now works over chunks (one per vector) instead of all tokens, so computation scales with #chunks, not #tokens
REFRAG also preserves autoregressive decoding: it can “compress anywhere” in the context and mix embeddings with real token embeddings under a learned policy.In practice, an RL-based selection policy chooses a few “important” chunks to expand back to full token sequences, while keeping most chunks as single embeddings.This contrasts with standard RAG, where every token of every retrieved passage is fully input to the decoder (wasting compute on irrelevant or redundant text).

Press enter or click to view image in full size

REFRAG’s pipeline. Retrieved context is split into chunks and fed to a lightweight encoder; an RL policy chooses some chunks to expand into full tokens. The decoder (a base LLM) receives the query tokens plus chunk embeddings
The REFRAG model couples a decoder-only LLM (e.g. LLaMA, GPT) with a lightweight encoder (e.g. RoBERTa). Given a query (tokens x1…xq​) and retrieved context (x_{q+1}…x_Txq+1​…xT​), REFRAG splits the context into L chunks of size k each.The encoder processes each chunk CiC_iCi​ to produce a vector c_i = M_{enc}(C_i). A linear projection ϕ then maps each c_i​ into the decoder’s embedding space (producing vectors e_i ^(enc)​ of the same dimension as word embeddings).

The decoder input becomes the original query embeddings {e_1, …, e_q\} plus the set of projected chunk embeddings {e^\text{enc}_1, …, e^\text{enc}_L\}. In effect, the context is condensed: s context tokens become L=s/k embeddings. As Lin et al. note, “the overall input to the decoder will be reduced by a factor of ~k”.An RL policy (a small neural network) then dynamically selects a few chunks whose full token embeddings should be used instead of their single-vector form. This “selective expansion” injects only the most critical text back into the decoder while leaving the rest compressed.

The result is a context with far fewer embeddings in memory and attention, significantly speeding up generation.

End-to-End Workflow
In practice, a REFRAG system operates as follows:

Retrieval: A standard retriever (e.g. FAISS index or DRAGON+) fetches the top-K relevant passages for the query from a large corpus (as in normal RAG).Each passages is split into fixed length chunks.Each passage is split into fixed-length chunks(e.g.s 16–32 chunks)
Chunk Encoding : (Compression): Each chunk is fed into the lightight encoder model.Typically we take [CLS] (or pooled) output as the chunk vector.These are pre-computed or cached for efficiency.
Projection : A learned linear layer projects each chunk vector into the LLMs token embedding space (dimension d_emb).After projection,each chunk is represented by one d_emb — dimensional embedding that the decoder can attend to as if it were a token.
Policy “ Sensing” : An RL-trained policy network examines all chunk embeddings (andpossible the query) and picks a subset(e.g.25%) of chunks to expand .Chunks not selected remain as single embedding(The policy is trained to maximize answer quality; it uses negative next token perplexity as reward signal).
Expansion (if any) : for each chunk the policy chose,we replace its single embedding with the sequence of its original token embeddings.This requires feeding those tokens embeddings into the decoder.
Decoing(Generation): The decoder LLM now receives a mixed sequence: query tokens + some chunk embeddings + expanded tokens.It applies self and cross attention as usual but since many chunks are in one embeddings form,te input sequence is much shorter.The LLM then generates the answer tokens autogressively.
Press enter or click to view image in full size

In summary REFRAG compresses the context chunks to reduce the input size ,senses /selects which chunks actually need full token detail ,then expands those few.This pipeline yields dramatic speedups :: e.g. with a 16× compression rate (k=16), REFRAG achieves ~16.5× faster time-to-first-token than a baseline LLaMA with the full context,all with negligible loss in accuracy.

Below is a simplified PyTorch example showing how one might implement the REFRAG idea for a document QA task. We use Hugging Face Transformers to (1) encode context chunks, (2) project them, and (3) feed to a causal LM via inputs_embeds. This is a proof-of-concept – a full REFRAG system requires training the encoder/projector and RL policy as described above.

import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

# Load models (small examples for illustration)
encoder_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
encoder_model     = AutoModel.from_pretrained("bert-base-uncased")
decoder_tokenizer = AutoTokenizer.from_pretrained("gpt2")
decoder_model     = AutoModelForCausalLM.from_pretrained("gpt2")

# Example retrieved passages (context) for a query
context_docs = [
    "Albert Einstein developed the theory of relativity in 1905.",
    "He received the Nobel Prize in 1921 for his services to theoretical physics.",
    "Later, he introduced the equation E = mc^2 in 1905."
]

# 1. Encode each chunk into a vector (CLS token embedding)
chunk_embeddings = []
for doc in context_docs:
    inputs = encoder_tokenizer(doc, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = encoder_model(**inputs)
    cls_vec = outputs.last_hidden_state[:, 0, :]  # [CLS] token embedding
    chunk_embeddings.append(cls_vec)  # shape [1, hidden_size]

# 2. Project chunk embeddings to decoder embedding size
proj = torch.nn.Linear(chunk_embeddings[0].size(-1), decoder_model.config.n_embd)
projected_chunks = [proj(vec) for vec in chunk_embeddings]  # list of [1, n_embd]

# 3. Prepare query tokens and embeddings
query = "Who won the Nobel Prize for physics in 1921?"
q_inputs = decoder_tokenizer(query, return_tensors="pt")
with torch.no_grad():
    q_embeds = decoder_model.transformer.wte(q_inputs["input_ids"])  # [1, len_q, n_embd]

# 4. Combine query embeddings and projected chunk embeddings
#    (Here we pretend *no* RL expansion: use all chunks as embeddings.)
combined_embeds = torch.cat([q_embeds] + projected_chunks, dim=1)  # [1, total_len, n_embd]

# 5. Generate answer with the decoder using inputs_embeds
generated = decoder_model.generate(inputs_embeds=combined_embeds, max_length=50)
print(decoder_tokenizer.decode(generated[0], skip_special_tokens=True))
In this code, we: (a) use a BERT encoder to compress each passage, (b) use a linear layer to map to GPT-2’s embedding size, and © call generate(..., inputs_embeds=...) on the GPT-2 model. The LLM then treats those vectors like special “pseudo-tokens” in its context. In a full REFRAG setup, one would replace or augment projected_chunks with actual token embeddings for chunks chosen by the RL policy.

Components Explained
Retriever: Typically a dense retriever or vector index (e.g. FAISS) is used to fetch K relevant passages from a corpus. REFRAG does not change this step: you still retrieve text like in RAG. For example, Lin et al. use a DRAGON+ dense retriever over Wikipedia/CommonCrawl (400M passages).
Encoder: A lightweight encoder (e.g. RoBERTa) processes each text chunk. It outputs a fixed-size vector (we often use the [CLS] token or mean pooling) as the chunk embedding. During continual pre-training (CPT), this encoder is trained (with the projection layer) to compress information with minimal loss.
Projection Layer: A learnable linear layer transforms each chunk embedding into the LLM’s token-embedding space (so it has the same dimension as a word vector). This lets the decoder attend to chunk vectors exactly like word embeddings.
Decoder (Generator): A standard decoder-only LLM (e.g. LLaMA, GPT-2) generates the answer autoregressively. In REFRAG, the decoder’s vocabulary and structure are unchanged. It simply sees a shorter input: the query tokens plus our chunk vectors (and any expanded tokens).
Selective Compression Policy: A small policy network (e.g. an MLP) decides which chunks should remain as single embeddings and which should be expanded to full tokens. This is trained with REINFORCE: chunks that, when expanded, reduce perplexity (improve next-token accuracy) are rewarded. Over time the policy learns to keep “easy to compress” context in embedding form and expand only the crucial parts
Continual Pre-training (CPT): Before deploying, REFRAG uses a specialized pre-training recipe to align encoder and decoder. First it learns to reconstruct text from embeddings: freeze the decoder, encode chunks and train the projection so the decoder can recover the original tokens. It then gradually moves to harder tasks via curriculum learning (start with 1 chunk, then 2, up to many). This ensures the encoder truly captures chunk meaning. Finally, the encoder/decoder are fine-tuned on end tasks (RAG QA, dialog, etc.) along with training the RL policy
Decoding: In standard RAG decoding, each output token attends to all query and passage tokens. In REFRAG decoding, the decoder attends to query tokens + L chunk embeddings + any expanded tokens. Because many chunks are single vectors, attention work is greatly reduced. Lin et al. note that RAG contexts exhibit “block-diagonal” sparsity (little cross-talk between unrelated chunks); REFRAG exploits this by skipping most redundant token attention.
LangChain has abstractions like:
Embeddings / embedding models
VectorStore / retriever
LLM (generator)
Chains, or RetrievalQA, etc.
What REFRAG adds / modifies:

Chunking: Split retrieved documents into chunks.
Encoder + Projector: Compress each chunk to a single embedding; project into LLM’s embedding space.
Policy: Decide which chunks to expand (i.e. feed full tokens) vs leave compressed.
Mixed context input for LLM: query + compressed chunk embeddings + expanded full tokens.
Training of encoder, projector, policy so that compression + expansion yields good generation.
LangChain doesn’t by default support feeding arbitrary embeddings mixed with full token embeddings in an LLM context (especially “expandable” policy). But since HF Transformers allow inputs_embeds you can build this.

Components to Build
A ChunkEncoder: takes text chunks, returns embeddings.
A Projector: linear layer mapping chunk embedding space → LLM token embedding space.
A PolicyNetwork: given query embedding + chunk embeddings, outputs a score per chunk (or selection mask) of whether to expand.
A Retriever and VectorStore: fetch candidate documents, optionally splitting into chunks.
A custom Chain / LLM wrapper: that constructs the mixed input, handles expansion, then calls LLM with inputs_embeds.
from typing import List, Tuple
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from langchain.embeddings import Embeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.llms.base import LLM

# 1. ChunkEncoder + Projector + Policy Network

class ChunkEncoder(nn.Module):
    def __init__(self, encoder_model_name: str, chunk_size: int):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_model_name)
        self.encoder = AutoModel.from_pretrained(encoder_model_name)
        self.chunk_size = chunk_size

    def chunkify(self, text: str) -> List[str]:
        # simple split by whitespace / fixed tokens, you could use tokenizer
        toks = self.tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0]
        chunks = []
        for i in range(0, toks.size(0), self.chunk_size):
            chunk_ids = toks[i : i + self.chunk_size]
            chunk_text = self.tokenizer.decode(chunk_ids, skip_special_tokens=True)
            chunks.append(chunk_text)
        return chunks

    def forward(self, chunk_texts: List[str]) -> torch.Tensor:
        # returns embeddings of shape (num_chunks, encoder_hidden_size)
        encodings = self.tokenizer(chunk_texts, padding=True, truncation=True, return_tensors="pt")
        outputs = self.encoder(**encodings)
        # e.g. use .pooler_output or CLS token
        # if model has pooler:
        if hasattr(outputs, "pooler_output"):
            return outputs.pooler_output  # (batch, hidden_size)
        else:
            # fallback: mean pooling
            last = outputs.last_hidden_state  # (batch, seq, hidden)
            return last.mean(dim=1)

class Projector(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, chunk_embs: torch.Tensor) -> torch.Tensor:
        return self.linear(chunk_embs)  # maps into LLM emb dim

class PolicyNetwork(nn.Module):
    def __init__(self, emb_dim: int, hidden_size: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    def forward(self, query_emb: torch.Tensor, chunk_embs: torch.Tensor) -> torch.Tensor:
        # query_emb: (emb_dim,), chunk_embs: (num_chunks, emb_dim)
        # produce a score per chunk
        q = query_emb.unsqueeze(0).expand(chunk_embs.size(0), -1)  # (num_chunks, emb_dim)
        inp = torch.cat([q, chunk_embs], dim=1)
        scores = self.net(inp).squeeze(-1)  # (num_chunks,)
        return scores

# 2. Using LangChain components

class RefragChain:
    def __init__(
        self,
        retriever,                # a LangChain retriever
        chunk_encoder: ChunkEncoder,
        projector: Projector,
        policy: PolicyNetwork,
        llm_model_name: str,
        llm_tokenizer_name: str,
        expand_ratio: float = 0.25,  # fraction of chunks to expand
        max_new_tokens: int = 128
    ):
        self.retriever = retriever
        self.chunk_encoder = chunk_encoder
        self.projector = projector
        self.policy = policy
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_tokenizer_name)
        self.llm = AutoModelForCausalLM.from_pretrained(llm_model_name)
        self.expand_ratio = expand_ratio
        self.max_new_tokens = max_new_tokens

    def answer(self, query: str) -> str:
        # 1. Retrieve documents
        docs: List[Document] = self.retriever.get_relevant_documents(query)

        # 2. Split into chunks
        chunk_texts = []
        chunk_doc_map = []  # to know which chunk came from which doc
        for d in docs:
            chunks = self.chunk_encoder.chunkify(d.page_content)
            for c in chunks:
                chunk_texts.append(c)
                chunk_doc_map.append(d)

        # 3. Encode chunks & project
        chunk_embs = self.chunk_encoder(chunk_texts)  # (C, enc_dim)
        projected = self.projector(chunk_embs)        # (C, llm_emb_dim)

        # 4. Compute query embedding (using chunk_encoder as proxy or separate)
        # optionally, use same encoder
        with torch.no_grad():
            q_enc = self.chunk_encoder([query])  # (1, enc_dim)
        q_proj = self.projector(q_enc).squeeze(0)  # (llm_emb_dim,)

        # 5. Policy: select top-k chunks to expand
        scores = self.policy(q_proj, chunk_embs)  # (C,)
        k = max(1, int(self.expand_ratio * len(chunk_texts)))
        topk_idx = torch.topk(scores, k).indices.tolist()

        # 6. Prepare LLM input embeddings
        #   a) tokenize query
        q_tok = self.llm_tokenizer(query, return_tensors="pt", truncation=True)
        q_tok_ids = q_tok["input_ids"]
        q_embeds = self.llm.get_input_embeddings()(q_tok_ids)  # (1, q_len, llm_emb_dim)

        #   b) For each chunk: if in topk, tokenize fully, else use projected embedding
        chunk_input_embeds_list = []
        for i, c_text in enumerate(chunk_texts):
            if i in topk_idx:
                # expand fully
                tok = self.llm_tokenizer(c_text, return_tensors="pt", truncation=True)
                emb = self.llm.get_input_embeddings()(tok["input_ids"])  # (1, chunk_len, emb_dim)
            else:
                # compressed: treat projection as one “special token embedding”
                emb = projected[i].unsqueeze(0).unsqueeze(1)  # (1,1, emb_dim)
            chunk_input_embeds_list.append(emb)

        # concatenate embeddings: query + all chunk embeddings/expanded
        all_chunk_embeds = torch.cat(chunk_input_embeds_list, dim=1)  # e.g. (1, total_chunkified_length, emb_dim)
        full_input_embeds = torch.cat([q_embeds, all_chunk_embeds], dim=1)

        # 7. Generate
        out = self.llm.generate(
            inputs_embeds=full_input_embeds,
            max_new_tokens=self.max_new_tokens
        )
        answer = self.llm_tokenizer.decode(out[0], skip_special_tokens=True)
        return answer

from langchain.llms.base import LLM
from langchain.schema import LLMResult

class RefragLLM(LLM):
    def __init__(self, refrag_chain: RefragChain):
        self.refrag_chain = refrag_chain

    def _call(self, prompt: str, stop: List[str] = None) -> str:
        return self.refrag_chain.answer(prompt)

    @property
    def _identifying_params(self):
        return {"refrag": True}
Train / load pretrained ChunkEncoder + Projector + Policy: The toy example uses random/untrained components; in REFRAG you’d pre-train so the compressed embeddings retain relevant information.
Align embedding spaces so that projected embeddings behave well when attended by LLM.
RL / reward signal: for training the policy, measuring e.g. how expansion of certain chunks improves generation quality / reduces perplexity.
Curriculum & compression ratios: vary k (chunk size) and expand ratios; ensure performance trade-offs are understood.
Evaluation: Benchmarks and Metrics
REFRAG is evaluated on long-context tasks spanning open-domain QA, multi-choice reasoning, conversation, and summarization. Key evaluation points include:

Datasets: The authors test on RAG benchmarks (e.g. NaturalQuestions, FEVER, TQA, etc.), commonsense reasoning (HellaSwag, Winogrande, etc.), and dialog/summarization tasks. They simulate both strong retriever scenarios (only the true top-K relevant passages) and weak retriever scenarios (lots of candidates, pick some irrelevant).
Baselines: Comparisons include LLaMA-2 (with full context or truncated to match token counts), plus prior long-context methods like CEPE and REPLUG. For RAG QA, they fine-tune a LLaMA model on the same data as REFRAG for a fair comparison.
Metrics: Inference speed is measured by Time-to-First-Token (TTFT) and Time-per-Iterative-Token (TTIT), as well as overall throughput (tokens/sec). Accuracy is measured via perplexity on held-out text and task accuracy (exact match/F1, etc.) on QA tasks.
Results: Across the board, REFRAG achieves massive speedups with no loss of accuracy. For example, with a 16× compression (k=16) on very long context, REFRAG’s TTFT is ~16.5× faster than LLaMA With k=32, TTFT reaches ~32.9× LLaMA (≈30.85× reported) matching the claimed 30.85× speedup. Perplexity and downstream accuracy remain essentially unchanged. In tasks with a weak retriever, REFRAG even outperforms LLaMA (since it can include more context under the same latency budget). Table 3 and Figure 4 of the paper show that REFRAG matches or beats LLaMA on 16 RAG tasks in both strong/weak settings. Ablations also show REFRAG’s RL-driven selective compression outperforms naive heuristics (like dropping low-perplexity chunks)
In summary, REFRAG delivers >30× faster generation (TTFT) with large contexts and no accuracy drop, effectively extending LLM context by up to 16× while keeping latency in check.

Trade-offs, Limitations & Enhancements
Training Overhead: REFRAG requires additional pre-training and fine-tuning. The encoder, projector, and policy must be trained (via reconstruction and curriculum tasks) in addition to any downstream training. This is more complex than plug-and-play RAG.
System Complexity: The system adds new components (encoder, projection layer, policy network), and relies on RL for chunk selection. This increases engineering effort and may require careful tuning (e.g. policy learning rate). The reference implementation highlights many hyperparameters (chunk size k, expand fraction p, learning rates, etc.)
Dependence on Retrieval: While REFRAG helps even when retrieval is imperfect, it still relies on retrieving somewhat relevant chunks. If the retriever fails entirely, compression alone can’t recover the missing knowledge.
Compression Limits: Extremely high compression (too few embeddings) could risk information loss. The RL policy mitigates this by expanding critical parts, but there is a trade-off between speed and fidelity. In experiments, REFRAG maintains accuracy up to 16×–32× compression; beyond that, performance degrades (Figure 10 in the paper shows higher loss with too high compression).
Applicability: REFRAG is tailored for long-context RAG tasks. In scenarios with short prompts or very tight latency (tiny queries), the overhead of running the encoder/policy might outweigh benefits. Its gains are greatest for contexts of thousands of tokens.
Overall, REFRAG is a specialized solution for latency-sensitive, knowledge-intensive tasks. It trades extra model training and system complexity for dramatic inference speedups. As reported, “we propose REFRAG… without requiring modifications to the LLM architecture”, meaning it works with any decoder model once encoder/projector are aligned.

Additional Resources
Original Paper: Lin et al., “REFRAG: Rethinking RAG based Decoding,” . This paper details the methods, experiments, and theory (e.g. scaling with k2k²k2).
GitHub Reference: The Facebook/Meta team has released code (facebookresearch/refrag). An independent compact implementation (Python/Transformers) , which implements FAISS retrieval, encoder+projector, and a REINFORCE policy.
Blog Summaries: Explanations and summaries of REFRAG are available online. For example, TechTalks by Ben Dickson provides an overview (“30× faster without sacrificing quality”), and Medium blogs by Saumajit Saha and Anas Limem cover the core ideas (compressed chunk embeddings, curriculum training, RL selection) in an accessible way
Reproducibility: The authors trained on the SlimPajama dataset (GitHub/HuggingFace) and evaluated on standard benchmarks
A message from our Founder
Hey, Sunil here. I wanted to take a moment to thank you for reading until the end and for being a part of this community.

Did you know that our team run these publications as a volunteer effort to over 3.5m monthly readers? We don’t receive any funding, we do this to support the community. ❤️

If you want to show some love, please take a moment to follow me on LinkedIn, TikTok, Instagram. You can also subscribe to our weekly newsletter.

And before you go, don’t forget to clap and follow the writer️!