use std::sync::Arc;

use augur::agent::action::SocialAction;
use augur::channel::Channel;
use augur::clock::Clock;
use augur::platform::{Platform, PlatformConfig};
use augur::types::*;
use serde_json::json;
use tokio::sync::RwLock;

/// Test harness holding the shared channel, an action for agent 0, and the
/// platform background task handle.
struct TestHarness {
    channel: Arc<Channel>,
    action: SocialAction,
    handle: tokio::task::JoinHandle<()>,
}

impl TestHarness {
    /// Shut down the platform cleanly. Exit does not send a response, so we
    /// write the message without waiting for a reply, then join the task.
    async fn teardown(self) {
        self.channel
            .write_to_receive_queue(0, serde_json::Value::Null, ActionType::Exit)
            .await;
        self.handle.await.unwrap();
    }
}

/// Two-agent test harness.
struct TwoAgentHarness {
    channel: Arc<Channel>,
    action0: SocialAction,
    action1: SocialAction,
    handle: tokio::task::JoinHandle<()>,
}

impl TwoAgentHarness {
    async fn teardown(self) {
        self.channel
            .write_to_receive_queue(0, serde_json::Value::Null, ActionType::Exit)
            .await;
        self.handle.await.unwrap();
    }
}

/// Spin up an in-memory Platform on a background task. Signs up agent 0 as "alice".
async fn setup() -> TestHarness {
    let channel = Arc::new(Channel::new());
    let clock = Arc::new(RwLock::new(Clock::new(1)));
    let config = PlatformConfig::default();
    let platform =
        Platform::new(":memory:", channel.clone(), clock, RecsysType::Random, config).unwrap();
    let platform = Arc::new(platform);
    let p = platform.clone();
    let handle = tokio::spawn(async move { p.running().await });
    let action = SocialAction::new(0, channel.clone());

    let res = action
        .perform_action(
            json!({"user_name": "alice", "name": "Alice", "bio": "Test user"}),
            ActionType::SignUp,
        )
        .await;
    assert!(res.success, "setup sign_up failed: {:?}", res);

    TestHarness {
        channel,
        action,
        handle,
    }
}

/// Spin up with a custom config. Signs up agent 0 as "alice".
async fn setup_with_config(config: PlatformConfig) -> TestHarness {
    let channel = Arc::new(Channel::new());
    let clock = Arc::new(RwLock::new(Clock::new(1)));
    let platform =
        Platform::new(":memory:", channel.clone(), clock, RecsysType::Random, config).unwrap();
    let platform = Arc::new(platform);
    let p = platform.clone();
    let handle = tokio::spawn(async move { p.running().await });
    let action = SocialAction::new(0, channel.clone());

    let res = action
        .perform_action(
            json!({"user_name": "alice", "name": "Alice", "bio": "Test user"}),
            ActionType::SignUp,
        )
        .await;
    assert!(res.success, "setup sign_up failed: {:?}", res);

    TestHarness {
        channel,
        action,
        handle,
    }
}

/// Setup with two agents on the same platform. Signs up agent 0 as "alice"
/// (user_id=1) and agent 1 as "bob" (user_id=2).
async fn setup_two_agents() -> TwoAgentHarness {
    let channel = Arc::new(Channel::new());
    let clock = Arc::new(RwLock::new(Clock::new(1)));
    let config = PlatformConfig::default();
    let platform =
        Platform::new(":memory:", channel.clone(), clock, RecsysType::Random, config).unwrap();
    let platform = Arc::new(platform);
    let p = platform.clone();
    let handle = tokio::spawn(async move { p.running().await });

    let action0 = SocialAction::new(0, channel.clone());
    let action1 = SocialAction::new(1, channel.clone());

    let res = action0
        .perform_action(
            json!({"user_name": "alice", "name": "Alice", "bio": "Test user"}),
            ActionType::SignUp,
        )
        .await;
    assert!(res.success, "agent 0 sign_up failed: {:?}", res);

    let res = action1
        .perform_action(
            json!({"user_name": "bob", "name": "Bob", "bio": "Another user"}),
            ActionType::SignUp,
        )
        .await;
    assert!(res.success, "agent 1 sign_up failed: {:?}", res);

    TwoAgentHarness {
        channel,
        action0,
        action1,
        handle,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
async fn test_create_post() {
    let h = setup().await;

    let res = h.action.create_post("Hello, world!").await;
    assert!(res.success, "create_post failed: {:?}", res);
    let post_id = res.data["post_id"].as_i64().unwrap();
    assert!(post_id > 0, "expected positive post_id, got {}", post_id);

    h.teardown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_follow_and_unfollow() {
    let h = setup_two_agents().await;

    // Agent 0 (user_id=1) follows agent 1 (user_id=2)
    let res = h.action0.follow(2).await;
    assert!(res.success, "follow failed: {:?}", res);

    // Duplicate follow should fail
    let res = h.action0.follow(2).await;
    assert!(!res.success, "duplicate follow should fail");

    // Unfollow
    let res = h.action0.unfollow(2).await;
    assert!(res.success, "unfollow failed: {:?}", res);

    // Double unfollow should fail
    let res = h.action0.unfollow(2).await;
    assert!(!res.success, "double unfollow should fail");

    h.teardown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_like_unlike_post() {
    let h = setup_two_agents().await;

    // Agent 1 creates a post
    let res = h.action1.create_post("Likeable post").await;
    assert!(res.success);
    let post_id = res.data["post_id"].as_i64().unwrap();

    // Agent 0 likes the post
    let res = h.action0.like_post(post_id).await;
    assert!(res.success, "like_post failed: {:?}", res);

    // Duplicate like should fail
    let res = h.action0.like_post(post_id).await;
    assert!(!res.success, "duplicate like should fail");

    // Unlike
    let res = h.action0.unlike_post(post_id).await;
    assert!(res.success, "unlike_post failed: {:?}", res);

    // Unlike again should fail
    let res = h.action0.unlike_post(post_id).await;
    assert!(!res.success, "double unlike should fail");

    h.teardown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_dislike_undo_dislike_post() {
    let h = setup_two_agents().await;

    // Agent 1 creates a post
    let res = h.action1.create_post("Dislikeable post").await;
    assert!(res.success);
    let post_id = res.data["post_id"].as_i64().unwrap();

    // Agent 0 dislikes the post
    let res = h.action0.dislike_post(post_id).await;
    assert!(res.success, "dislike_post failed: {:?}", res);

    // Duplicate dislike should fail
    let res = h.action0.dislike_post(post_id).await;
    assert!(!res.success, "duplicate dislike should fail");

    // Undo dislike
    let res = h.action0.undo_dislike_post(post_id).await;
    assert!(res.success, "undo_dislike_post failed: {:?}", res);

    // Undo dislike again should fail
    let res = h.action0.undo_dislike_post(post_id).await;
    assert!(!res.success, "double undo_dislike should fail");

    h.teardown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_repost() {
    let h = setup_two_agents().await;

    // Agent 1 creates a post
    let res = h.action1.create_post("Original post").await;
    assert!(res.success);
    let post_id = res.data["post_id"].as_i64().unwrap();

    // Agent 0 reposts it
    let res = h.action0.repost(post_id).await;
    assert!(res.success, "repost failed: {:?}", res);
    let repost_id = res.data["post_id"].as_i64().unwrap();
    assert_ne!(repost_id, post_id, "repost should create a new post_id");

    // Duplicate repost should fail
    let res = h.action0.repost(post_id).await;
    assert!(!res.success, "duplicate repost should fail");

    h.teardown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_create_comment_and_engagement() {
    let h = setup_two_agents().await;

    // Agent 0 creates a post
    let res = h.action0.create_post("Post with comments").await;
    assert!(res.success);
    let post_id = res.data["post_id"].as_i64().unwrap();

    // Agent 1 comments on the post
    let res = h
        .action1
        .perform_action(
            json!({"post_id": post_id, "content": "Great post!"}),
            ActionType::CreateComment,
        )
        .await;
    assert!(res.success, "create_comment failed: {:?}", res);
    let comment_id = res.data["comment_id"].as_i64().unwrap();
    assert!(comment_id > 0);

    // Agent 0 likes the comment
    let res = h.action0.like_comment(comment_id).await;
    assert!(res.success, "like_comment failed: {:?}", res);

    // Duplicate like should fail
    let res = h.action0.like_comment(comment_id).await;
    assert!(!res.success, "duplicate like_comment should fail");

    // Unlike the comment
    let res = h.action0.unlike_comment(comment_id).await;
    assert!(res.success, "unlike_comment failed: {:?}", res);

    // Unlike again should fail
    let res = h.action0.unlike_comment(comment_id).await;
    assert!(!res.success, "double unlike_comment should fail");

    // Dislike the comment
    let res = h.action0.dislike_comment(comment_id).await;
    assert!(res.success, "dislike_comment failed: {:?}", res);

    // Duplicate dislike should fail
    let res = h.action0.dislike_comment(comment_id).await;
    assert!(!res.success, "duplicate dislike_comment should fail");

    // Undo dislike
    let res = h.action0.undo_dislike_comment(comment_id).await;
    assert!(res.success, "undo_dislike_comment failed: {:?}", res);

    // Undo dislike again should fail
    let res = h.action0.undo_dislike_comment(comment_id).await;
    assert!(!res.success, "double undo_dislike_comment should fail");

    h.teardown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_mute_unmute() {
    let h = setup_two_agents().await;

    // Agent 0 mutes agent 1 (user_id=2)
    let res = h.action0.mute(2).await;
    assert!(res.success, "mute failed: {:?}", res);

    // Unmute
    let res = h.action0.unmute(2).await;
    assert!(res.success, "unmute failed: {:?}", res);

    // Double unmute should fail
    let res = h.action0.unmute(2).await;
    assert!(!res.success, "double unmute should fail");

    h.teardown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_search_posts() {
    let h = setup().await;

    // Create several posts
    let res = h.action.create_post("Rust is awesome").await;
    assert!(res.success);
    let res = h.action.create_post("Python is great too").await;
    assert!(res.success);
    let res = h.action.create_post("Rust and Python comparison").await;
    assert!(res.success);

    // Search for "Rust"
    let res = h.action.search_posts("Rust").await;
    assert!(res.success, "search_posts failed: {:?}", res);
    let posts = res.data["posts"].as_array().unwrap();
    assert_eq!(
        posts.len(),
        2,
        "expected 2 posts matching 'Rust', got {}",
        posts.len()
    );

    // Search for "Python"
    let res = h.action.search_posts("Python").await;
    assert!(res.success);
    let posts = res.data["posts"].as_array().unwrap();
    assert_eq!(
        posts.len(),
        2,
        "expected 2 posts matching 'Python', got {}",
        posts.len()
    );

    // Search for something absent
    let res = h.action.search_posts("JavaScript").await;
    assert!(res.success);
    let posts = res.data["posts"].as_array().unwrap();
    assert_eq!(posts.len(), 0, "expected 0 posts matching 'JavaScript'");

    h.teardown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_search_user() {
    let h = setup_two_agents().await;

    // Search for "alice"
    let res = h.action0.search_user("alice").await;
    assert!(res.success, "search_user failed: {:?}", res);
    let users = res.data["users"].as_array().unwrap();
    assert_eq!(
        users.len(),
        1,
        "expected 1 user matching 'alice', got {}",
        users.len()
    );
    assert_eq!(users[0]["user_name"].as_str().unwrap(), "alice");

    // Search for "bob"
    let res = h.action0.search_user("bob").await;
    assert!(res.success);
    let users = res.data["users"].as_array().unwrap();
    assert_eq!(users.len(), 1);
    assert_eq!(users[0]["user_name"].as_str().unwrap(), "bob");

    // Search for nonexistent user
    let res = h.action0.search_user("charlie").await;
    assert!(res.success);
    let users = res.data["users"].as_array().unwrap();
    assert_eq!(users.len(), 0);

    h.teardown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_trend() {
    let h = setup().await;

    // Create posts
    h.action.create_post("Trending post 1").await;
    h.action.create_post("Trending post 2").await;
    h.action.create_post("Trending post 3").await;

    // Check trend returns posts
    let res = h.action.trend().await;
    assert!(res.success, "trend failed: {:?}", res);
    let posts = res.data["posts"].as_array().unwrap();
    assert_eq!(
        posts.len(),
        3,
        "expected 3 trending posts, got {}",
        posts.len()
    );

    h.teardown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_quote_post() {
    let h = setup_two_agents().await;

    // Agent 1 creates a post
    let res = h.action1.create_post("Original thought").await;
    assert!(res.success);
    let post_id = res.data["post_id"].as_i64().unwrap();

    // Agent 0 quotes it
    let res = h
        .action0
        .perform_action(
            json!({"post_id": post_id, "quote_content": "I agree with this"}),
            ActionType::QuotePost,
        )
        .await;
    assert!(res.success, "quote_post failed: {:?}", res);
    let quote_post_id = res.data["post_id"].as_i64().unwrap();
    assert_ne!(
        quote_post_id, post_id,
        "quote should create a new post_id"
    );

    h.teardown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_group_chat_lifecycle() {
    let h = setup_two_agents().await;

    // Agent 0 creates a group
    let res = h.action0.create_group("test-group").await;
    assert!(res.success, "create_group failed: {:?}", res);
    let group_id = res.data["group_id"].as_i64().unwrap();
    assert!(group_id > 0);

    // Agent 1 joins the group
    let res = h.action1.join_group(group_id).await;
    assert!(res.success, "join_group failed: {:?}", res);

    // Agent 0 sends a message to the group
    let res = h
        .action0
        .perform_action(
            json!({"group_id": group_id, "message": "Hello group!"}),
            ActionType::SendToGroup,
        )
        .await;
    assert!(res.success, "send_to_group failed: {:?}", res);
    let message_id = res.data["message_id"].as_i64().unwrap();
    assert!(message_id > 0);

    // Agent 1 listens from the group
    let res = h.action1.listen_from_group().await;
    assert!(res.success, "listen_from_group failed: {:?}", res);
    let messages = res.data["messages"].as_array().unwrap();
    assert!(!messages.is_empty(), "expected at least one message");
    assert_eq!(messages[0]["content"].as_str().unwrap(), "Hello group!");

    // Agent 1 leaves the group
    let res = h.action1.leave_group(group_id).await;
    assert!(res.success, "leave_group failed: {:?}", res);

    h.teardown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_report_post() {
    let h = setup_two_agents().await;

    // Agent 1 creates a post
    let res = h.action1.create_post("Offensive content").await;
    assert!(res.success);
    let post_id = res.data["post_id"].as_i64().unwrap();

    // Agent 0 reports the post
    let res = h
        .action0
        .perform_action(
            json!({"post_id": post_id, "report_reason": "Spam"}),
            ActionType::ReportPost,
        )
        .await;
    assert!(res.success, "report_post failed: {:?}", res);
    let report_id = res.data["report_id"].as_i64().unwrap();
    assert!(report_id > 0);

    h.teardown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_do_nothing() {
    let h = setup().await;

    let res = h.action.do_nothing().await;
    assert!(res.success, "do_nothing failed: {:?}", res);

    h.teardown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_self_rating_disabled() {
    let config = PlatformConfig {
        allow_self_rating: false,
        ..PlatformConfig::default()
    };
    let h = setup_with_config(config).await;

    // Create a post (user_id=1, agent_id=0)
    let res = h.action.create_post("My own post").await;
    assert!(res.success);
    let post_id = res.data["post_id"].as_i64().unwrap();

    // Try to like own post — should fail
    let res = h.action.like_post(post_id).await;
    assert!(
        !res.success,
        "self-like should fail when allow_self_rating=false"
    );
    assert!(
        res.data["error"]
            .as_str()
            .unwrap()
            .contains("cannot rate own post"),
        "expected 'cannot rate own post' error, got: {:?}",
        res.data
    );

    // Try to dislike own post — should fail
    let res = h.action.dislike_post(post_id).await;
    assert!(
        !res.success,
        "self-dislike should fail when allow_self_rating=false"
    );
    assert!(
        res.data["error"]
            .as_str()
            .unwrap()
            .contains("cannot rate own post"),
        "expected 'cannot rate own post' error, got: {:?}",
        res.data
    );

    h.teardown().await;
}
