# Augur: 1:1 OASIS Port in Rust

**Date**: 2026-04-03
**Status**: Draft
**Repo**: `influenceable-io/augur`
**License**: Apache 2.0 (matching OASIS)

---

## 1. Goal

Port [camel-ai/oasis](https://github.com/camel-ai/oasis) (v0.2.5) to Rust as a standalone library crate called `augur`. This is a **complete, feature-for-feature reimplementation** — every action type, every recommendation algorithm, every database table, every API entry point. No features omitted, no features added.

The purpose is to produce a drop-in replacement that can be swapped for OASIS in downstream projects (like Egregore), and later extended with Rust-native capabilities.

---

## 2. What We're Porting

### 2.1 Module Map

| OASIS Python Module | Augur Rust Module | Description |
|---------------------|-------------------|-------------|
| `oasis/__init__.py` | `src/lib.rs` | Public API: `make()`, re-exports |
| `oasis/social_platform/platform.py` | `src/platform/mod.rs` | Core platform engine (30 action handlers) |
| `oasis/social_platform/channel.py` | `src/channel.rs` | Async message queue (UUID-keyed req/res) |
| `oasis/social_platform/typing.py` | `src/types.rs` | `ActionType`, `RecsysType`, `DefaultPlatformType` enums |
| `oasis/social_platform/recsys.py` | `src/recsys/mod.rs` | 4 recommendation algorithms |
| `oasis/social_platform/database.py` | `src/db/schema.rs` | 17-table SQLite schema |
| `oasis/social_platform/platform_utils.py` | `src/db/queries.rs` | DB helper functions |
| `oasis/social_agent/agent.py` | `src/agent/mod.rs` | `SocialAgent` with LLM integration |
| `oasis/social_agent/agent_action.py` | `src/agent/action.rs` | `SocialAction` (30 action methods) |
| `oasis/social_agent/agent_environment.py` | `src/agent/environment.rs` | `SocialEnvironment` (observation prompts) |
| `oasis/social_agent/agent_graph.py` | `src/agent/graph.rs` | `AgentGraph` (directed graph + agent registry) |
| `oasis/social_agent/agents_generator.py` | `src/agent/generator.rs` | Agent factory functions (Twitter CSV, Reddit JSON) |
| `oasis/environment/env.py` | `src/env.rs` | `OasisEnv` → `AugurEnv` (orchestrator) |
| `oasis/environment/env_action.py` | `src/env_action.rs` | `ManualAction`, `LLMAction` |
| `oasis/environment/make.py` | `src/lib.rs` (`make()`) | Factory function |
| `oasis/clock/clock.py` | `src/clock.rs` | Time simulation with k-factor acceleration |

### 2.2 Action Types (30 total — all ported)

**Content creation:** `CreatePost`, `Repost`, `QuotePost`, `CreateComment`
**Post engagement:** `LikePost`, `UnlikePost`, `DislikePost`, `UndoDislikePost`
**Comment engagement:** `LikeComment`, `UnlikeComment`, `DislikeComment`, `UndoDislikeComment`
**Social graph:** `Follow`, `Unfollow`, `Mute`, `Unmute`
**Discovery:** `Refresh`, `SearchPosts`, `SearchUser`, `Trend`
**Group chat:** `CreateGroup`, `JoinGroup`, `LeaveGroup`, `SendToGroup`, `ListenFromGroup`
**Administrative:** `SignUp`, `ReportPost`, `UpdateRecTable`
**Special:** `DoNothing`, `PurchaseProduct`, `Interview`, `Exit`

### 2.3 Recommendation Algorithms (4 — all ported)

1. **Random** — Random post distribution (no ML)
2. **Reddit** — Hot-score algorithm: `sign * log(max(|s|, 1), 10) + seconds / 45000`
3. **Twitter** — SentenceTransformer (`paraphrase-MiniLM-L6-v2`) cosine similarity between user bio and post content
4. **TWHiN** — Multi-signal: TWHiN-BERT embeddings + time decay + like similarity. Optional OpenAI embeddings fallback.

### 2.4 Database Schema (17 tables — all ported)

**Core:** `user`, `post`, `comment`
**Engagement:** `like`, `dislike`, `comment_like`, `comment_dislike`
**Social:** `follow`, `mute`
**System:** `trace`, `rec`, `report`
**Group chat:** `chat_group`, `group_members`, `group_messages`
**Commerce:** `product`

### 2.5 Agent Features (all ported)

- `perform_action_by_llm()` — LLM observes environment, picks action via tool calling
- `perform_action_by_data()` — Execute scripted action programmatically
- `perform_action_by_hci()` — Human-in-the-loop interactive mode
- `perform_test()` — Group polarization test prompt
- `perform_interview()` — Structured interview with optional memory recording
- `perform_agent_graph_action()` — Update social graph on follow/unfollow
- Agent graph backends: igraph (→ petgraph) and Neo4j

### 2.6 Environment Lifecycle (ported exactly)

```
augur::make(agent_graph, platform, db_path, semaphore) -> AugurEnv
  env.reset()                    // Start platform task, sign up agents
  env.step(actions)              // Update recsys, execute actions concurrently
  env.close()                    // EXIT signal, commit DB, cleanup
```

---

## 3. Rust Architecture

### 3.1 Project Structure

```
augur/
├── Cargo.toml
├── src/
│   ├── lib.rs                   # Public API: make(), re-exports
│   ├── types.rs                 # ActionType, RecsysType, DefaultPlatformType, ActionResult
│   ├── channel.rs               # Channel, AsyncSafeMap (tokio mpsc + DashMap)
│   ├── clock.rs                 # Clock (k-factor time acceleration)
│   ├── env.rs                   # AugurEnv (orchestrator)
│   ├── env_action.rs            # ManualAction, LLMAction
│   ├── platform/
│   │   ├── mod.rs               # Platform struct (30 action handlers)
│   │   └── config.rs            # PlatformConfig
│   ├── db/
│   │   ├── mod.rs               # Database connection management
│   │   ├── schema.rs            # 17-table schema creation
│   │   └── queries.rs           # All SQL queries
│   ├── recsys/
│   │   ├── mod.rs               # RecSys trait + dispatch
│   │   ├── random.rs            # Random distribution
│   │   ├── reddit.rs            # Hot-score algorithm
│   │   ├── twitter.rs           # SentenceTransformer cosine similarity
│   │   └── twhin.rs             # TWHiN-BERT multi-signal
│   ├── agent/
│   │   ├── mod.rs               # SocialAgent struct
│   │   ├── action.rs            # SocialAction (30 action methods via Channel)
│   │   ├── environment.rs       # SocialEnvironment (observation prompts)
│   │   ├── graph.rs             # AgentGraph (petgraph + agent registry)
│   │   └── generator.rs         # generate_twitter_agent_graph, generate_reddit_agent_graph
│   └── llm/
│       ├── mod.rs               # LLM backend trait
│       ├── openai.rs            # OpenAI-compatible client (async-openai)
│       └── tools.rs             # FunctionTool definitions for agent actions
├── tests/
│   ├── platform_tests.rs        # All 30 action handlers
│   ├── recsys_tests.rs          # All 4 algorithms
│   ├── channel_tests.rs         # Message passing
│   ├── agent_tests.rs           # Agent lifecycle
│   ├── env_tests.rs             # Full simulation lifecycle
│   └── integration_tests.rs     # End-to-end simulation
└── examples/
    ├── twitter_simulation.rs    # Twitter platform example
    ├── reddit_simulation.rs     # Reddit platform example
    ├── custom_platform.rs       # Custom platform configuration
    └── group_chat.rs            # Group chat example
```

### 3.2 Crate Dependencies

```toml
[dependencies]
# Async runtime
tokio = { version = "1", features = ["full"] }

# Database
rusqlite = { version = "0.31", features = ["bundled"] }

# Serialization
serde = { version = "1", features = ["derive"] }
serde_json = "1"

# Graph
petgraph = "0.6"

# Async collections
dashmap = "6"

# UUID for message IDs
uuid = { version = "1", features = ["v4"] }

# DateTime
chrono = { version = "0.4", features = ["serde"] }

# LLM client
async-openai = "0.25"

# Embeddings (for Twitter/TWHiN recsys)
ort = { version = "2", features = ["load-dynamic"] }  # ONNX Runtime
tokenizers = "0.20"                                     # HuggingFace tokenizers

# Math (for recsys)
ndarray = "0.16"

# CSV/JSON parsing (for agent generators)
csv = "1"

# Logging
tracing = "0.1"
tracing-subscriber = "0.3"

# Error handling
thiserror = "2"
anyhow = "1"

[dev-dependencies]
tokio-test = "0.4"
tempfile = "3"
```

### 3.3 Core Types

```rust
// src/types.rs

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionType {
    CreatePost,
    Repost,
    QuotePost,
    CreateComment,
    LikePost,
    UnlikePost,
    DislikePost,
    UndoDislikePost,
    LikeComment,
    UnlikeComment,
    DislikeComment,
    UndoDislikeComment,
    Follow,
    Unfollow,
    Mute,
    Unmute,
    Refresh,
    SearchPosts,
    SearchUser,
    Trend,
    CreateGroup,
    JoinGroup,
    LeaveGroup,
    SendToGroup,
    ListenFromGroup,
    SignUp,
    ReportPost,
    UpdateRecTable,
    DoNothing,
    PurchaseProduct,
    Interview,
    Exit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecsysType {
    Twitter,
    Twhin,
    Reddit,
    Random,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DefaultPlatformType {
    Twitter,
    Reddit,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionResult {
    pub success: bool,
    #[serde(flatten)]
    pub data: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserInfo {
    pub user_name: Option<String>,
    pub name: Option<String>,
    pub description: Option<String>,
    pub profile: Option<HashMap<String, serde_json::Value>>,
    pub recsys_type: RecsysType,
    pub is_controllable: bool,
}
```

### 3.4 Channel (Async Message Passing)

Maps OASIS's `Channel` class directly:

```rust
// src/channel.rs

pub struct Channel {
    /// Platform reads from this (agents write requests)
    receive_tx: mpsc::UnboundedSender<(Uuid, (i64, serde_json::Value, ActionType))>,
    receive_rx: Mutex<mpsc::UnboundedReceiver<(Uuid, (i64, serde_json::Value, ActionType))>>,
    /// Agents read from this (platform writes responses)
    send_dict: DashMap<Uuid, (Uuid, i64, ActionResult)>,
    notify: Notify,
}

impl Channel {
    pub fn new() -> Self;

    /// Platform receives next request
    pub async fn receive_from(&self) -> (Uuid, (i64, serde_json::Value, ActionType));

    /// Platform sends response
    pub async fn send_to(&self, message_id: Uuid, agent_id: i64, result: ActionResult);

    /// Agent writes request, gets message_id
    pub async fn write_to_receive_queue(&self, agent_id: i64, message: serde_json::Value, action: ActionType) -> Uuid;

    /// Agent polls for response
    pub async fn read_from_send_queue(&self, message_id: Uuid) -> (Uuid, i64, ActionResult);
}
```

### 3.5 Platform (30 Action Handlers)

```rust
// src/platform/mod.rs

pub struct Platform {
    db: Connection,              // rusqlite
    channel: Arc<Channel>,
    sandbox_clock: Arc<RwLock<Clock>>,
    recsys_type: RecsysType,
    config: PlatformConfig,
}

pub struct PlatformConfig {
    pub refresh_rec_post_count: usize,
    pub max_rec_post_len: usize,
    pub show_score: bool,
    pub allow_self_rating: bool,
}

impl Platform {
    pub fn new(db_path: &str, channel: Arc<Channel>, clock: Arc<RwLock<Clock>>,
               recsys_type: RecsysType, config: PlatformConfig) -> Result<Self>;

    /// Main event loop — receives actions from channel, dispatches, sends results
    pub async fn running(&self);

    // --- 30 action handlers (all async, all return ActionResult) ---
    async fn sign_up(&self, agent_id: i64, user_message: Value) -> ActionResult;
    async fn create_post(&self, agent_id: i64, content: String) -> ActionResult;
    async fn repost(&self, agent_id: i64, post_id: i64) -> ActionResult;
    async fn quote_post(&self, agent_id: i64, quote_message: Value) -> ActionResult;
    async fn create_comment(&self, agent_id: i64, comment_message: Value) -> ActionResult;
    async fn like_post(&self, agent_id: i64, post_id: i64) -> ActionResult;
    async fn unlike_post(&self, agent_id: i64, post_id: i64) -> ActionResult;
    async fn dislike_post(&self, agent_id: i64, post_id: i64) -> ActionResult;
    async fn undo_dislike_post(&self, agent_id: i64, post_id: i64) -> ActionResult;
    async fn like_comment(&self, agent_id: i64, comment_id: i64) -> ActionResult;
    async fn unlike_comment(&self, agent_id: i64, comment_id: i64) -> ActionResult;
    async fn dislike_comment(&self, agent_id: i64, comment_id: i64) -> ActionResult;
    async fn undo_dislike_comment(&self, agent_id: i64, comment_id: i64) -> ActionResult;
    async fn follow(&self, agent_id: i64, followee_id: i64) -> ActionResult;
    async fn unfollow(&self, agent_id: i64, followee_id: i64) -> ActionResult;
    async fn mute(&self, agent_id: i64, mutee_id: i64) -> ActionResult;
    async fn unmute(&self, agent_id: i64, mutee_id: i64) -> ActionResult;
    async fn refresh(&self, agent_id: i64) -> ActionResult;
    async fn search_posts(&self, agent_id: i64, query: String) -> ActionResult;
    async fn search_user(&self, agent_id: i64, query: String) -> ActionResult;
    async fn trend(&self, agent_id: i64) -> ActionResult;
    async fn create_group(&self, agent_id: i64, group_name: String) -> ActionResult;
    async fn join_group(&self, agent_id: i64, group_id: i64) -> ActionResult;
    async fn leave_group(&self, agent_id: i64, group_id: i64) -> ActionResult;
    async fn send_to_group(&self, agent_id: i64, message: Value) -> ActionResult;
    async fn listen_from_group(&self, agent_id: i64) -> ActionResult;
    async fn report_post(&self, agent_id: i64, report_message: Value) -> ActionResult;
    async fn purchase_product(&self, agent_id: i64, purchase_message: Value) -> ActionResult;
    async fn interview(&self, agent_id: i64, interview_data: Value) -> ActionResult;
    async fn do_nothing(&self, agent_id: i64) -> ActionResult;

    // --- Recommendation system ---
    pub async fn update_rec_table(&self);

    // --- Internal helpers ---
    fn check_agent_userid(&self, agent_id: i64) -> Result<i64>;
    fn get_post_type(&self, post_id: i64) -> Result<String>;
    fn add_comments_to_posts(&self, posts: Vec<Value>) -> Vec<Value>;
    fn record_trace(&self, user_id: i64, action: &str, info: &str);
}
```

### 3.6 Agent System

```rust
// src/agent/mod.rs

pub struct SocialAgent {
    pub agent_id: i64,
    pub user_info: UserInfo,
    channel: Arc<Channel>,
    action: SocialAction,
    env: SocialEnvironment,
    model: Option<Arc<dyn LlmBackend>>,
    agent_graph: Option<Arc<RwLock<AgentGraph>>>,
    available_actions: Vec<ActionType>,
    memory: Vec<ChatMessage>,
    max_iteration: usize,
    interview_record: bool,
}

impl SocialAgent {
    pub async fn perform_action_by_llm(&mut self) -> ActionResult;
    pub async fn perform_action_by_data(&mut self, func_name: &str, args: Value) -> ActionResult;
    pub async fn perform_action_by_hci(&mut self) -> ActionResult;
    pub async fn perform_test(&mut self) -> Value;
    pub async fn perform_interview(&mut self, prompt: &str) -> Value;
    pub fn perform_agent_graph_action(&self, action: &str, args: &Value);
}

// src/agent/action.rs

pub struct SocialAction {
    agent_id: i64,
    channel: Arc<Channel>,
}

impl SocialAction {
    pub async fn perform_action(&self, message: Value, action_type: ActionType) -> ActionResult;

    // 30 typed action methods matching OASIS's SocialAction
    pub async fn sign_up(&self, user_name: &str, name: &str, bio: &str) -> ActionResult;
    pub async fn create_post(&self, content: &str) -> ActionResult;
    // ... (all 30)

    pub fn get_function_tools(&self) -> Vec<FunctionTool>;
}

// src/agent/environment.rs

pub struct SocialEnvironment {
    action: SocialAction,
}

impl SocialEnvironment {
    pub async fn get_posts_env(&self) -> String;
    pub async fn get_followers_env(&self) -> String;
    pub async fn get_follows_env(&self) -> String;
    pub async fn get_group_env(&self) -> String;
    pub async fn to_text_prompt(&self, include_posts: bool, include_followers: bool,
                                 include_follows: bool, include_groups: bool) -> String;
}

// src/agent/graph.rs

pub struct AgentGraph {
    graph: petgraph::Graph<i64, (), petgraph::Directed>,
    node_map: HashMap<i64, petgraph::graph::NodeIndex>,
    agent_mappings: HashMap<i64, SocialAgent>,
}

impl AgentGraph {
    pub fn new() -> Self;
    pub fn add_agent(&mut self, agent_id: i64, agent: SocialAgent);
    pub fn remove_agent(&mut self, agent_id: i64);
    pub fn get_agent(&self, agent_id: i64) -> Option<&SocialAgent>;
    pub fn get_agent_mut(&mut self, agent_id: i64) -> Option<&mut SocialAgent>;
    pub fn get_agents(&self) -> Vec<(i64, &SocialAgent)>;
    pub fn add_edge(&mut self, from: i64, to: i64);
    pub fn remove_edge(&mut self, from: i64, to: i64);
    pub fn get_edges(&self) -> Vec<(i64, i64)>;
    pub fn get_num_nodes(&self) -> usize;
    pub fn get_num_edges(&self) -> usize;
    pub fn reset(&mut self);
}
```

### 3.7 Environment (Orchestrator)

```rust
// src/env.rs

pub struct AugurEnv {
    pub agent_graph: Arc<RwLock<AgentGraph>>,
    platform: Arc<Platform>,
    platform_type: DefaultPlatformType,
    channel: Arc<Channel>,
    sandbox_clock: Arc<RwLock<Clock>>,
    platform_task: Option<JoinHandle<()>>,
    llm_semaphore: Arc<Semaphore>,
}

impl AugurEnv {
    pub async fn reset(&mut self) -> Result<()>;

    pub async fn step(&mut self, actions: HashMap<i64, Action>) -> Result<()>;

    pub async fn close(&mut self) -> Result<()>;
}

// src/env_action.rs

pub enum Action {
    Manual(ManualAction),
    Llm(LLMAction),
    Interview { prompt: String },
    Multiple(Vec<Action>),
}

pub struct ManualAction {
    pub action_type: ActionType,
    pub action_args: HashMap<String, serde_json::Value>,
}

pub struct LLMAction;
```

### 3.8 Recommendation System

```rust
// src/recsys/mod.rs

pub trait RecSys: Send + Sync {
    fn recommend(&self, user_table: &[UserRow], post_table: &[PostRow],
                 trace_table: &[TraceRow], rec_matrix: &mut Vec<Vec<i64>>,
                 max_rec_post_len: usize);
}

// src/recsys/random.rs   — rec_sys_random
// src/recsys/reddit.rs    — rec_sys_reddit (hot-score, heap-based top-k)
// src/recsys/twitter.rs   — rec_sys_personalized (SentenceTransformer via ONNX)
// src/recsys/twhin.rs     — rec_sys_personalized_twh (TWHiN-BERT, time decay, like score)
```

### 3.9 LLM Integration

```rust
// src/llm/mod.rs

#[async_trait]
pub trait LlmBackend: Send + Sync {
    async fn chat_completion(&self, messages: Vec<ChatMessage>,
                              tools: Vec<FunctionTool>) -> Result<LlmResponse>;
}

pub struct ChatMessage {
    pub role: Role,
    pub content: String,
}

pub enum Role { System, User, Assistant, Tool }

pub struct LlmResponse {
    pub content: Option<String>,
    pub tool_calls: Vec<ToolCall>,
}

pub struct ToolCall {
    pub id: String,
    pub function_name: String,
    pub arguments: serde_json::Value,
}

pub struct FunctionTool {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,  // JSON Schema
}

// src/llm/openai.rs — OpenAI-compatible implementation via async-openai
// Configurable base_url (works with OpenRouter, vLLM, any compatible endpoint)
```

### 3.10 Clock

```rust
// src/clock.rs

pub struct Clock {
    pub k: u64,                          // Time acceleration factor
    pub real_start_time: DateTime<Utc>,
    pub time_step: u64,
}

impl Clock {
    pub fn new(k: u64) -> Self;
    pub fn time_transfer(&self, now: DateTime<Utc>, start: DateTime<Utc>) -> DateTime<Utc>;
    pub fn get_time_step(&self) -> String;
}
```

### 3.11 Public API

```rust
// src/lib.rs

/// Factory function matching oasis.make()
pub async fn make(
    agent_graph: AgentGraph,
    platform: PlatformOrDefault,
    database_path: &str,
    semaphore: usize,
) -> Result<AugurEnv>;

pub enum PlatformOrDefault {
    Default(DefaultPlatformType),
    Custom(Platform),
}

// Re-exports
pub use types::*;
pub use env::AugurEnv;
pub use env_action::*;
pub use agent::{SocialAgent, AgentGraph, UserInfo};
pub use agent::generator::*;
pub use channel::Channel;
pub use clock::Clock;
pub use platform::Platform;
```

---

## 4. Behavioral Parity Checklist

Every behavior must match OASIS exactly:

- [ ] Duplicate follow prevention (check before insert)
- [ ] Duplicate like prevention (check before insert)
- [ ] Duplicate repost prevention (same user can't repost same post twice)
- [ ] Self-rating toggle (`allow_self_rating` config)
- [ ] `show_score` mode (combined score vs separate like/dislike counts)
- [ ] Trace recording for every action
- [ ] Comment enrichment on refresh (`_add_comments_to_posts`)
- [ ] Refresh combines rec table posts + following posts
- [ ] Trend returns top-K posts by likes in last N days
- [ ] Search supports content, post_id, user_id, username, bio
- [ ] Group chat: create, join, leave, send, listen lifecycle
- [ ] Product purchase (increment sales counter)
- [ ] Interview (record prompt/response, optional memory)
- [ ] Agent graph updates on follow/unfollow
- [ ] Clock acceleration (`k` factor) for both timestep and datetime modes
- [ ] Semaphore-limited concurrent LLM calls
- [ ] Multiple actions per agent per step
- [ ] `update_rec_table` at start of each step
- [ ] EXIT action commits and closes database
- [ ] Agent generators: Twitter (CSV), Reddit (JSON), custom, 1M-scale

---

## 5. Testing Strategy

- **Unit tests** for each of the 30 action handlers (mock DB)
- **Unit tests** for each of the 4 recsys algorithms (known inputs → expected outputs)
- **Integration tests** for full simulation lifecycle (reset → step → close)
- **Channel tests** for concurrent message passing correctness
- **Agent tests** with mock LLM backend (verify prompt construction and tool call parsing)
- **Parity tests** comparing SQLite output from OASIS Python vs Augur Rust on identical inputs

---

## 6. Non-Goals (for this phase)

- No gRPC/HTTP server layer (library crate only)
- No multi-platform extensions
- No cognitive model extensions
- No custom platform types beyond Twitter/Reddit defaults
- No Neo4j backend (petgraph only; Neo4j can be added later)
- No distributed simulation
