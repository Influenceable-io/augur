use serde::{Deserialize, Serialize};
use std::fmt;

/// All 30 action types supported by the platform, matching OASIS exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionType {
    // Content creation
    CreatePost,
    Repost,
    QuotePost,
    CreateComment,

    // Post engagement
    LikePost,
    UnlikePost,
    DislikePost,
    UndoDislikePost,

    // Comment engagement
    LikeComment,
    UnlikeComment,
    DislikeComment,
    UndoDislikeComment,

    // Social graph
    Follow,
    Unfollow,
    Mute,
    Unmute,

    // Discovery
    Refresh,
    SearchPosts,
    SearchUser,
    Trend,

    // Group chat
    CreateGroup,
    JoinGroup,
    LeaveGroup,
    SendToGroup,
    ListenFromGroup,

    // Administrative
    SignUp,
    ReportPost,
    UpdateRecTable,

    // Special
    DoNothing,
    PurchaseProduct,
    Interview,
    Exit,
}

impl fmt::Display for ActionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = serde_json::to_value(self)
            .ok()
            .and_then(|v| v.as_str().map(String::from))
            .unwrap_or_else(|| format!("{:?}", self));
        write!(f, "{}", s)
    }
}

/// Recommendation system algorithm type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecsysType {
    Twitter,
    #[serde(rename = "twhin-bert")]
    Twhin,
    Reddit,
    Random,
}

/// Default platform presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DefaultPlatformType {
    Twitter,
    Reddit,
}

/// Result returned by every platform action handler.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionResult {
    pub success: bool,
    #[serde(flatten)]
    pub data: serde_json::Value,
}

impl ActionResult {
    pub fn ok(data: serde_json::Value) -> Self {
        Self {
            success: true,
            data,
        }
    }

    pub fn fail(reason: &str) -> Self {
        Self {
            success: false,
            data: serde_json::json!({ "error": reason }),
        }
    }
}
