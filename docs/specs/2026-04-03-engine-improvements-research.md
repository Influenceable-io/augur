# Augur Engine: Improvements & Optimization Research

**Date**: 2026-04-03
**Status**: Research notes for next phase
**Context**: After completing the 1:1 OASIS port, these are researched improvements to make augur a best-in-class simulation engine.

---

## 1. Performance Optimizations

### 1.1 ONNX Runtime for Real Embeddings

**Current state**: Twitter and TWHiN recsys use hash-based text embeddings as a fallback.

**Improvement**: Use the `ort` crate (ONNX Runtime) with real sentence transformer models:
- **paraphrase-MiniLM-L6-v2** for Twitter recsys (same as OASIS)
- **Twitter/twhin-bert-base** for TWHiN recsys (same as OASIS)

Benefits: 3-5x faster than Python equivalents, 60-80% less memory. The `ort` crate is battle-tested.

**Relevant crates**: `ort` (ONNX Runtime), `tokenizers` (HuggingFace tokenizers), `hf-hub` (model downloading)

**Reference**: [Building Sentence Transformers in Rust](https://dev.to/mayu2008/building-sentence-transformers-in-rust-a-practical-guide-with-burn-onnx-runtime-and-candle-281k), [EmbedAnything](https://github.com/StarlightSearch/EmbedAnything)

### 1.2 Batch LLM Inference via vLLM

**Current state**: Each agent makes individual LLM calls bounded by a semaphore.

**Improvement**: Add native vLLM batch inference support:
- **Continuous batching** — dynamically replaces completed sequences with new ones (23x throughput improvement)
- **Prefix caching** — reuses KV pages for identical prompt prefixes (system prompts shared across agents)
- **Chunked prefill** — splits large prompts so prefill runs in parallel with decoding

This is the single biggest performance unlock for large-scale simulations. OASIS is bottlenecked by sequential LLM calls.

**Implementation**: Add a `VllmBatchBackend` that implements `LlmBackend` but collects all agent prompts per step and submits them as a batch. Use the OpenAI-compatible batch API or vLLM's native batch endpoint.

**Reference**: [vLLM Optimization Guide](https://docs.vllm.ai/en/stable/configuration/optimization/)

### 1.3 Database Optimization

- **Connection pooling**: Replace single `Mutex<Connection>` with `r2d2` connection pool for `rusqlite`. Allows concurrent reads while maintaining single-writer safety.
- **Prepared statement caching**: Pre-compile frequently used SQL statements instead of preparing them each time.
- **Bulk inserts**: Use `INSERT INTO ... VALUES (...), (...), (...)` for batch operations during `reset()` and `update_rec_table`.
- **WAL2 mode**: SQLite's experimental WAL2 mode for even better concurrent read performance.

### 1.4 Parallel Agent Activation

**Current state**: Agents are activated sequentially in `env.step()`.

**Improvement**: Use `tokio::task::JoinSet` or `futures::stream::buffer_unordered` for bounded concurrent execution. This is already partially implemented via the semaphore, but can be improved with:
- Agent-level parallelism with `buffer_unordered(N)`
- Platform-level batching of DB writes (accumulate actions, commit in batch)

---

## 2. Architectural Improvements

### 2.1 Group Agent Support (GA-S³)

**Research**: GA-S³ (ACL 2025) models collections of similar-behaving individuals as a single "group agent," enabling simulation of large-scale network phenomena at manageable computational cost.

**Implementation**: Add a `GroupAgent` that represents N individuals with similar profiles. Instead of N LLM calls, make 1 call for the group and distribute actions probabilistically. This could enable 10x-100x scale improvement.

**Reference**: [GA-S³ Paper](https://aclanthology.org/2025.findings-acl.468/), [GitHub](https://github.com/AI4SS/GAS-3)

### 2.2 Cognitive Architecture (AgentSociety-style)

**Research**: AgentSociety (2025) gives agents human-like "minds" with emotions, needs, motivations, and world cognition. Behaviors (mobility, employment, consumption, social interactions) are driven by internal mental states.

**Implementation**: Extend `SocialAgent` with:
```rust
struct CognitiveState {
    beliefs: Vec<Belief>,           // What the agent believes
    attitudes: HashMap<String, f32>, // Sentiment toward topics (-1.0 to 1.0)
    emotional_state: EmotionalState, // Current emotions
    needs: NeedState,               // Maslow-style hierarchy
    memory: StructuredMemory,       // Episodic + semantic memory
}
```

This makes agents more realistic — they form opinions based on what they've seen, their emotional state affects their behavior, and their memory of past interactions influences future decisions.

**Reference**: [AgentSociety](https://arxiv.org/html/2502.08691v1)

### 2.3 Content Moderation Simulation (MOSAIC)

**Research**: MOSAIC (EMNLP 2025) evaluates content moderation strategies within multi-agent simulations. Agents generate and consume content including misinformation, and different fact-checking mechanisms (community-based, third-party, hybrid) are tested.

**Implementation**: Add moderation system to the Platform:
- `ModerationType` enum: `Community`, `ThirdParty`, `Hybrid`, `None`
- Automatic flagging based on report threshold
- Content visibility reduction for flagged posts
- Agent trust scores that update based on reporting accuracy

**Reference**: [MOSAIC Paper](https://aclanthology.org/2025.emnlp-main.325/)

### 2.4 Multi-Platform Simulation

This was already designed in the prior Egregore Engine spec. The key insight from the prior spec:

> No published system combines (a) LLM-based agent cognition with persistent memory, (b) unified agent identity across multiple social platforms, and (c) platform-specific behavior adaptation from a single cognitive core.

**Implementation**: Add platform trait with Twitter, Reddit, ShortForm (TikTok), and Meatspace (IRL) implementations. Agents have a unified cognitive state that spans all platforms.

---

## 3. Scale Improvements

### 3.1 Million-Agent Scale (MIT AAMAS 2025)

**Research**: MIT Media Lab achieved scaling LLM-guided agent simulations to millions by creating a digital twin of New York City with 8.4 million autonomous agents.

**Key techniques**:
- **Hierarchical agent representation**: Not all agents need full LLM reasoning. Use a pyramid: top 1% get full LLM, next 9% get distilled models, remaining 90% use rule-based heuristics informed by LLM-generated behavior profiles.
- **Spatial partitioning**: Only simulate agents in "active zones" — agents outside interaction range are frozen.
- **Agent archetypes**: Cluster similar agents and share computation.

### 3.2 Distributed Simulation

For truly large-scale simulations (100K+ agents), distribute across machines:
- **Partitioned platform**: Each machine handles a subset of agents
- **Message passing**: Use gRPC or NATS for cross-partition communication
- **Shared state**: Use Redis or a distributed DB for global state (trending, rec tables)

---

## 4. Validation & Quality

### 4.1 Herd Behavior Mitigation

**Known OASIS limitation**: Agents are more susceptible to herd behavior than humans — they're more likely to follow others' opinions.

**Mitigations**:
- Add "stubbornness" parameter to agent profiles (probability of ignoring majority)
- Implement contrarian behavior models
- Add noise to LLM temperature based on agent personality

### 4.2 Benchmark Suite

Build a validation benchmark like GA-S³'s:
- Compare simulation output distributions against real social media data
- Track metrics: sentiment diversity, opinion clustering, engagement patterns
- Regression tests to ensure model changes don't degrade realism

### 4.3 Attitude & Belief Dynamics

**Research**: DeGroot model (weighted opinion averaging) and Hegselmann-Krause model (bounded confidence) provide formal foundations for opinion dynamics.

**Implementation**: Instead of relying solely on LLM for opinion changes, add formal opinion dynamics as a post-processing step:
- Each agent has explicit attitude scores on key topics
- After each interaction, update attitudes using bounded confidence model
- LLM generates behavior based on current attitudes

---

## 5. Immediate Next Steps (Priority Order)

1. **ONNX Runtime integration** — Replace hash embeddings with real sentence transformers (high impact, isolated change)
2. **Batch LLM inference** — VllmBatchBackend for 10-20x throughput improvement
3. **Group agent support** — GA-S³ style for 10-100x scale
4. **Cognitive state model** — Beliefs, attitudes, emotions for more realistic agents
5. **Multi-platform** — Unified agent identity across Twitter/Reddit/etc
6. **gRPC service layer** — Expose engine as a microservice for the Egregore frontend

---

## References

- [OASIS Paper (EMNLP 2024)](https://arxiv.org/html/2411.11581v4)
- [GA-S³: Group Agents for Scalability (ACL 2025)](https://aclanthology.org/2025.findings-acl.468/)
- [MOSAIC: Content Moderation Simulation (EMNLP 2025)](https://aclanthology.org/2025.emnlp-main.325/)
- [AgentSociety: 10K+ Agent Simulation (2025)](https://arxiv.org/html/2502.08691v1)
- [MIT AAMAS 2025: Scaling to Millions](https://www.media.mit.edu/posts/new-paper-on-limits-of-agency-at-aamas-2025/)
- [krABMaga: Rust ABM Framework](https://github.com/krABMaga/krABMaga)
- [Rust for Massive ABM Simulations (Springer)](https://link.springer.com/chapter/10.1007/978-981-15-1078-6_2)
- [vLLM Optimization Guide](https://docs.vllm.ai/en/stable/configuration/optimization/)
- [EmbedAnything: Rust Embeddings](https://github.com/StarlightSearch/EmbedAnything)
- [Validation Challenges in Generative Social Simulation (Springer 2025)](https://link.springer.com/article/10.1007/s10462-025-11412-6)
