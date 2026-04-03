use std::sync::Arc;

use serde_json::{json, Value};

use crate::channel::Channel;
use crate::llm::FunctionTool;
use crate::types::*;

/// Agent-side action dispatcher. Sends requests through the Channel to the Platform.
///
/// Mirrors OASIS's SocialAction class with all 30 action methods.
#[derive(Clone)]
pub struct SocialAction {
    agent_id: i64,
    channel: Arc<Channel>,
}

impl SocialAction {
    pub fn new(agent_id: i64, channel: Arc<Channel>) -> Self {
        Self { agent_id, channel }
    }

    /// Send an action through the channel and wait for the result.
    pub async fn perform_action(&self, message: Value, action_type: ActionType) -> ActionResult {
        let message_id = self
            .channel
            .write_to_receive_queue(self.agent_id, message, action_type)
            .await;
        let (_, _, result) = self.channel.read_from_send_queue(message_id).await;
        result
    }

    /// Dispatch an action by its string name (used by LLM tool calling).
    pub async fn perform_action_by_name(&self, name: &str, args: &Value) -> ActionResult {
        match name {
            "sign_up" => {
                self.perform_action(args.clone(), ActionType::SignUp).await
            }
            "create_post" => {
                let content = args.get("content").and_then(|v| v.as_str()).unwrap_or("");
                self.create_post(content).await
            }
            "repost" => {
                let post_id = args.get("post_id").and_then(|v| v.as_i64()).unwrap_or(0);
                self.repost(post_id).await
            }
            "quote_post" => {
                self.perform_action(args.clone(), ActionType::QuotePost).await
            }
            "create_comment" => {
                self.perform_action(args.clone(), ActionType::CreateComment).await
            }
            "like_post" => {
                let post_id = args.get("post_id").and_then(|v| v.as_i64()).unwrap_or(0);
                self.like_post(post_id).await
            }
            "unlike_post" => {
                let post_id = args.get("post_id").and_then(|v| v.as_i64()).unwrap_or(0);
                self.unlike_post(post_id).await
            }
            "dislike_post" => {
                let post_id = args.get("post_id").and_then(|v| v.as_i64()).unwrap_or(0);
                self.dislike_post(post_id).await
            }
            "undo_dislike_post" => {
                let post_id = args.get("post_id").and_then(|v| v.as_i64()).unwrap_or(0);
                self.undo_dislike_post(post_id).await
            }
            "like_comment" => {
                let comment_id = args.get("comment_id").and_then(|v| v.as_i64()).unwrap_or(0);
                self.like_comment(comment_id).await
            }
            "unlike_comment" => {
                let comment_id = args.get("comment_id").and_then(|v| v.as_i64()).unwrap_or(0);
                self.unlike_comment(comment_id).await
            }
            "dislike_comment" => {
                let comment_id = args.get("comment_id").and_then(|v| v.as_i64()).unwrap_or(0);
                self.dislike_comment(comment_id).await
            }
            "undo_dislike_comment" => {
                let comment_id = args.get("comment_id").and_then(|v| v.as_i64()).unwrap_or(0);
                self.undo_dislike_comment(comment_id).await
            }
            "follow" => {
                let followee_id = args.get("followee_id").and_then(|v| v.as_i64()).unwrap_or(0);
                self.follow(followee_id).await
            }
            "unfollow" => {
                let followee_id = args.get("followee_id").and_then(|v| v.as_i64()).unwrap_or(0);
                self.unfollow(followee_id).await
            }
            "mute" => {
                let mutee_id = args.get("mutee_id").and_then(|v| v.as_i64()).unwrap_or(0);
                self.mute(mutee_id).await
            }
            "unmute" => {
                let mutee_id = args.get("mutee_id").and_then(|v| v.as_i64()).unwrap_or(0);
                self.unmute(mutee_id).await
            }
            "refresh" => self.refresh().await,
            "search_posts" => {
                let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("");
                self.search_posts(query).await
            }
            "search_user" => {
                let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("");
                self.search_user(query).await
            }
            "trend" => self.trend().await,
            "create_group" => {
                let name = args.get("group_name").and_then(|v| v.as_str()).unwrap_or("");
                self.create_group(name).await
            }
            "join_group" => {
                let group_id = args.get("group_id").and_then(|v| v.as_i64()).unwrap_or(0);
                self.join_group(group_id).await
            }
            "leave_group" => {
                let group_id = args.get("group_id").and_then(|v| v.as_i64()).unwrap_or(0);
                self.leave_group(group_id).await
            }
            "send_to_group" => {
                self.perform_action(args.clone(), ActionType::SendToGroup).await
            }
            "listen_from_group" => self.listen_from_group().await,
            "report_post" => {
                self.perform_action(args.clone(), ActionType::ReportPost).await
            }
            "purchase_product" => {
                self.perform_action(args.clone(), ActionType::PurchaseProduct).await
            }
            "do_nothing" => self.do_nothing().await,
            _ => ActionResult::fail(&format!("unknown action: {}", name)),
        }
    }

    // ========================================================================
    // Typed action methods
    // ========================================================================

    pub async fn create_post(&self, content: &str) -> ActionResult {
        self.perform_action(json!(content), ActionType::CreatePost).await
    }

    pub async fn repost(&self, post_id: i64) -> ActionResult {
        self.perform_action(json!(post_id), ActionType::Repost).await
    }

    pub async fn like_post(&self, post_id: i64) -> ActionResult {
        self.perform_action(json!(post_id), ActionType::LikePost).await
    }

    pub async fn unlike_post(&self, post_id: i64) -> ActionResult {
        self.perform_action(json!(post_id), ActionType::UnlikePost).await
    }

    pub async fn dislike_post(&self, post_id: i64) -> ActionResult {
        self.perform_action(json!(post_id), ActionType::DislikePost).await
    }

    pub async fn undo_dislike_post(&self, post_id: i64) -> ActionResult {
        self.perform_action(json!(post_id), ActionType::UndoDislikePost).await
    }

    pub async fn like_comment(&self, comment_id: i64) -> ActionResult {
        self.perform_action(json!(comment_id), ActionType::LikeComment).await
    }

    pub async fn unlike_comment(&self, comment_id: i64) -> ActionResult {
        self.perform_action(json!(comment_id), ActionType::UnlikeComment).await
    }

    pub async fn dislike_comment(&self, comment_id: i64) -> ActionResult {
        self.perform_action(json!(comment_id), ActionType::DislikeComment).await
    }

    pub async fn undo_dislike_comment(&self, comment_id: i64) -> ActionResult {
        self.perform_action(json!(comment_id), ActionType::UndoDislikeComment).await
    }

    pub async fn follow(&self, followee_id: i64) -> ActionResult {
        self.perform_action(json!(followee_id), ActionType::Follow).await
    }

    pub async fn unfollow(&self, followee_id: i64) -> ActionResult {
        self.perform_action(json!(followee_id), ActionType::Unfollow).await
    }

    pub async fn mute(&self, mutee_id: i64) -> ActionResult {
        self.perform_action(json!(mutee_id), ActionType::Mute).await
    }

    pub async fn unmute(&self, mutee_id: i64) -> ActionResult {
        self.perform_action(json!(mutee_id), ActionType::Unmute).await
    }

    pub async fn refresh(&self) -> ActionResult {
        self.perform_action(Value::Null, ActionType::Refresh).await
    }

    pub async fn search_posts(&self, query: &str) -> ActionResult {
        self.perform_action(json!(query), ActionType::SearchPosts).await
    }

    pub async fn search_user(&self, query: &str) -> ActionResult {
        self.perform_action(json!(query), ActionType::SearchUser).await
    }

    pub async fn trend(&self) -> ActionResult {
        self.perform_action(Value::Null, ActionType::Trend).await
    }

    pub async fn create_group(&self, name: &str) -> ActionResult {
        self.perform_action(json!(name), ActionType::CreateGroup).await
    }

    pub async fn join_group(&self, group_id: i64) -> ActionResult {
        self.perform_action(json!(group_id), ActionType::JoinGroup).await
    }

    pub async fn leave_group(&self, group_id: i64) -> ActionResult {
        self.perform_action(json!(group_id), ActionType::LeaveGroup).await
    }

    pub async fn listen_from_group(&self) -> ActionResult {
        self.perform_action(Value::Null, ActionType::ListenFromGroup).await
    }

    pub async fn do_nothing(&self) -> ActionResult {
        self.perform_action(Value::Null, ActionType::DoNothing).await
    }

    // ========================================================================
    // Function tool definitions for LLM tool calling
    // ========================================================================

    /// Get function tool definitions for the specified available actions.
    pub fn get_function_tools(&self, available_actions: &[ActionType]) -> Vec<FunctionTool> {
        available_actions
            .iter()
            .filter_map(|action| self.action_to_tool(*action))
            .collect()
    }

    fn action_to_tool(&self, action: ActionType) -> Option<FunctionTool> {
        let (name, desc, params) = match action {
            ActionType::CreatePost => (
                "create_post",
                "Create a new post on the platform",
                json!({"type": "object", "properties": {"content": {"type": "string", "description": "The content of the post"}}, "required": ["content"]}),
            ),
            ActionType::Repost => (
                "repost",
                "Repost/share an existing post",
                json!({"type": "object", "properties": {"post_id": {"type": "integer", "description": "ID of the post to repost"}}, "required": ["post_id"]}),
            ),
            ActionType::QuotePost => (
                "quote_post",
                "Quote an existing post with your own commentary",
                json!({"type": "object", "properties": {"post_id": {"type": "integer", "description": "ID of the post to quote"}, "quote_content": {"type": "string", "description": "Your commentary"}}, "required": ["post_id", "quote_content"]}),
            ),
            ActionType::CreateComment => (
                "create_comment",
                "Comment on a post",
                json!({"type": "object", "properties": {"post_id": {"type": "integer", "description": "ID of the post to comment on"}, "content": {"type": "string", "description": "Comment content"}}, "required": ["post_id", "content"]}),
            ),
            ActionType::LikePost => (
                "like_post",
                "Like a post",
                json!({"type": "object", "properties": {"post_id": {"type": "integer", "description": "ID of the post to like"}}, "required": ["post_id"]}),
            ),
            ActionType::UnlikePost => (
                "unlike_post",
                "Remove like from a post",
                json!({"type": "object", "properties": {"post_id": {"type": "integer", "description": "ID of the post to unlike"}}, "required": ["post_id"]}),
            ),
            ActionType::DislikePost => (
                "dislike_post",
                "Dislike a post",
                json!({"type": "object", "properties": {"post_id": {"type": "integer", "description": "ID of the post to dislike"}}, "required": ["post_id"]}),
            ),
            ActionType::UndoDislikePost => (
                "undo_dislike_post",
                "Remove dislike from a post",
                json!({"type": "object", "properties": {"post_id": {"type": "integer", "description": "ID of the post"}}, "required": ["post_id"]}),
            ),
            ActionType::LikeComment => (
                "like_comment",
                "Like a comment",
                json!({"type": "object", "properties": {"comment_id": {"type": "integer", "description": "ID of the comment to like"}}, "required": ["comment_id"]}),
            ),
            ActionType::UnlikeComment => (
                "unlike_comment",
                "Remove like from a comment",
                json!({"type": "object", "properties": {"comment_id": {"type": "integer", "description": "ID of the comment"}}, "required": ["comment_id"]}),
            ),
            ActionType::DislikeComment => (
                "dislike_comment",
                "Dislike a comment",
                json!({"type": "object", "properties": {"comment_id": {"type": "integer", "description": "ID of the comment to dislike"}}, "required": ["comment_id"]}),
            ),
            ActionType::UndoDislikeComment => (
                "undo_dislike_comment",
                "Remove dislike from a comment",
                json!({"type": "object", "properties": {"comment_id": {"type": "integer", "description": "ID of the comment"}}, "required": ["comment_id"]}),
            ),
            ActionType::Follow => (
                "follow",
                "Follow a user",
                json!({"type": "object", "properties": {"followee_id": {"type": "integer", "description": "ID of the user to follow"}}, "required": ["followee_id"]}),
            ),
            ActionType::Unfollow => (
                "unfollow",
                "Unfollow a user",
                json!({"type": "object", "properties": {"followee_id": {"type": "integer", "description": "ID of the user to unfollow"}}, "required": ["followee_id"]}),
            ),
            ActionType::Mute => (
                "mute",
                "Mute a user",
                json!({"type": "object", "properties": {"mutee_id": {"type": "integer", "description": "ID of the user to mute"}}, "required": ["mutee_id"]}),
            ),
            ActionType::Unmute => (
                "unmute",
                "Unmute a user",
                json!({"type": "object", "properties": {"mutee_id": {"type": "integer", "description": "ID of the user to unmute"}}, "required": ["mutee_id"]}),
            ),
            ActionType::Refresh => (
                "refresh",
                "Refresh your feed to see new posts",
                json!({"type": "object", "properties": {}}),
            ),
            ActionType::SearchPosts => (
                "search_posts",
                "Search for posts by content",
                json!({"type": "object", "properties": {"query": {"type": "string", "description": "Search query"}}, "required": ["query"]}),
            ),
            ActionType::SearchUser => (
                "search_user",
                "Search for users",
                json!({"type": "object", "properties": {"query": {"type": "string", "description": "Search query"}}, "required": ["query"]}),
            ),
            ActionType::Trend => (
                "trend",
                "View trending posts",
                json!({"type": "object", "properties": {}}),
            ),
            ActionType::CreateGroup => (
                "create_group",
                "Create a new group chat",
                json!({"type": "object", "properties": {"group_name": {"type": "string", "description": "Name for the group"}}, "required": ["group_name"]}),
            ),
            ActionType::JoinGroup => (
                "join_group",
                "Join a group chat",
                json!({"type": "object", "properties": {"group_id": {"type": "integer", "description": "ID of the group to join"}}, "required": ["group_id"]}),
            ),
            ActionType::LeaveGroup => (
                "leave_group",
                "Leave a group chat",
                json!({"type": "object", "properties": {"group_id": {"type": "integer", "description": "ID of the group to leave"}}, "required": ["group_id"]}),
            ),
            ActionType::SendToGroup => (
                "send_to_group",
                "Send a message to a group chat",
                json!({"type": "object", "properties": {"group_id": {"type": "integer", "description": "ID of the group"}, "message": {"type": "string", "description": "Message content"}}, "required": ["group_id", "message"]}),
            ),
            ActionType::ListenFromGroup => (
                "listen_from_group",
                "Read messages from groups you're in",
                json!({"type": "object", "properties": {}}),
            ),
            ActionType::ReportPost => (
                "report_post",
                "Report a post for policy violation",
                json!({"type": "object", "properties": {"post_id": {"type": "integer", "description": "ID of the post to report"}, "report_reason": {"type": "string", "description": "Reason for reporting"}}, "required": ["post_id", "report_reason"]}),
            ),
            ActionType::PurchaseProduct => (
                "purchase_product",
                "Purchase a product",
                json!({"type": "object", "properties": {"product_name": {"type": "string", "description": "Name of the product"}, "purchase_num": {"type": "integer", "description": "Quantity to purchase"}}, "required": ["product_name"]}),
            ),
            ActionType::DoNothing => (
                "do_nothing",
                "Choose to do nothing this turn",
                json!({"type": "object", "properties": {}}),
            ),
            // These are not exposed as tools
            ActionType::SignUp | ActionType::UpdateRecTable | ActionType::Exit | ActionType::Interview => {
                return None;
            }
        };

        Some(FunctionTool {
            name: name.to_string(),
            description: desc.to_string(),
            parameters: params,
        })
    }
}
