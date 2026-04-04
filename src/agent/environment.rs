use crate::agent::action::SocialAction;

/// Agent-side environment observation, matching OASIS's SocialEnvironment.
///
/// Provides formatted text prompts describing what the agent "sees" on the platform.
#[derive(Clone)]
pub struct SocialEnvironment {
    action: SocialAction,
}

impl SocialEnvironment {
    pub fn new(action: SocialAction) -> Self {
        Self { action }
    }

    /// Get posts from the agent's feed via refresh.
    pub async fn get_posts_env(&self) -> String {
        let result = self.action.refresh().await;
        if result.success
            && let Some(posts) = result.data.get("posts") {
                return format!("After refreshing, you see some posts: {}", posts);
            }
        "No posts available.".to_string()
    }

    /// Get follower and following counts via a single platform query.
    /// Returns (followers_text, follows_text).
    pub async fn get_social_counts_env(&self) -> (String, String) {
        let result = self.action.search_user(&self.action.agent_id().to_string()).await;
        if result.success {
            if let Some(users) = result.data.get("users").and_then(|v| v.as_array()) {
                if let Some(user) = users.first() {
                    let followers = user.get("num_followers").and_then(|v| v.as_i64()).unwrap_or(0);
                    let followings = user.get("num_followings").and_then(|v| v.as_i64()).unwrap_or(0);
                    return (
                        format!("You have {followers} followers."),
                        format!("You are following {followings} users."),
                    );
                }
            }
        }
        (String::new(), String::new())
    }

    /// Get group chat messages.
    pub async fn get_group_env(&self) -> String {
        let result = self.action.listen_from_group().await;
        if result.success
            && let Some(messages) = result.data.get("messages") {
                return format!(
                    "And there are many group chat channels. Here are recent messages: {}",
                    messages
                );
            }
        "".to_string()
    }

    /// Build a combined text prompt from all environment components.
    pub async fn to_text_prompt(
        &self,
        include_posts: bool,
        include_followers: bool,
        include_follows: bool,
        include_groups: bool,
    ) -> String {
        let mut parts = Vec::new();

        if include_posts {
            let posts = self.get_posts_env().await;
            if !posts.is_empty() {
                parts.push(posts);
            }
        }

        if include_followers || include_follows {
            let (followers, follows) = self.get_social_counts_env().await;
            if include_followers && !followers.is_empty() {
                parts.push(followers);
            }
            if include_follows && !follows.is_empty() {
                parts.push(follows);
            }
        }

        if include_groups {
            let groups = self.get_group_env().await;
            if !groups.is_empty() {
                parts.push(groups);
            }
        }

        parts.join("\n\n")
    }
}
