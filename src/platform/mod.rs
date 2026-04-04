pub mod config;

use std::sync::Arc;

use anyhow::Result;
use rusqlite::{params, Connection};
use serde_json::{json, Value};
use tokio::sync::{Mutex, RwLock};

use crate::channel::Channel;
use crate::clock::Clock;
use crate::db;
use crate::db::queries;
use crate::recsys::{self, TRACE_LIKE_POST_PREFIX};
use crate::types::*;

pub use config::PlatformConfig;

/// Core social media platform simulation engine.
///
/// Processes agent actions via an async Channel, persists all state to SQLite.
/// Mirrors OASIS's Platform class with all 30 action handlers.
pub struct Platform {
    db: Mutex<Connection>,
    channel: Arc<Channel>,
    clock: Arc<RwLock<Clock>>,
    recsys_type: RecsysType,
    config: PlatformConfig,
}

impl Platform {
    pub fn new(
        db_path: &str,
        channel: Arc<Channel>,
        clock: Arc<RwLock<Clock>>,
        recsys_type: RecsysType,
        config: PlatformConfig,
    ) -> Result<Self> {
        let conn = db::open_and_init(db_path)?;
        Ok(Self {
            db: Mutex::new(conn),
            channel,
            clock,
            recsys_type,
            config,
        })
    }

    pub fn sandbox_clock(&self) -> Arc<RwLock<Clock>> {
        self.clock.clone()
    }

    /// Main event loop. Receives actions from the channel, dispatches to handlers, sends results.
    pub async fn running(&self) {
        loop {
            let (message_id, (agent_id, message, action)) = self.channel.receive_from().await;

            if action == ActionType::Exit {
                let db = self.db.lock().await;
                let _ = db.execute_batch("PRAGMA wal_checkpoint(TRUNCATE);");
                tracing::info!("Platform shutting down");
                break;
            }

            let result = self.dispatch(agent_id, message, action).await;
            self.channel.send_to(message_id, agent_id, result).await;
        }
    }

    /// Dispatch an action to the appropriate handler.
    async fn dispatch(&self, agent_id: i64, message: Value, action: ActionType) -> ActionResult {
        match action {
            ActionType::SignUp => self.sign_up(agent_id, message).await,
            ActionType::CreatePost => {
                let content = message.as_str().unwrap_or("").to_string();
                self.create_post(agent_id, content).await
            }
            ActionType::Repost => {
                let post_id = message.as_i64().unwrap_or(0);
                self.repost(agent_id, post_id).await
            }
            ActionType::QuotePost => self.quote_post(agent_id, message).await,
            ActionType::CreateComment => self.create_comment(agent_id, message).await,
            ActionType::LikePost => {
                let post_id = message.as_i64().unwrap_or(0);
                self.like_post(agent_id, post_id).await
            }
            ActionType::UnlikePost => {
                let post_id = message.as_i64().unwrap_or(0);
                self.unlike_post(agent_id, post_id).await
            }
            ActionType::DislikePost => {
                let post_id = message.as_i64().unwrap_or(0);
                self.dislike_post(agent_id, post_id).await
            }
            ActionType::UndoDislikePost => {
                let post_id = message.as_i64().unwrap_or(0);
                self.undo_dislike_post(agent_id, post_id).await
            }
            ActionType::LikeComment => {
                let comment_id = message.as_i64().unwrap_or(0);
                self.like_comment(agent_id, comment_id).await
            }
            ActionType::UnlikeComment => {
                let comment_id = message.as_i64().unwrap_or(0);
                self.unlike_comment(agent_id, comment_id).await
            }
            ActionType::DislikeComment => {
                let comment_id = message.as_i64().unwrap_or(0);
                self.dislike_comment(agent_id, comment_id).await
            }
            ActionType::UndoDislikeComment => {
                let comment_id = message.as_i64().unwrap_or(0);
                self.undo_dislike_comment(agent_id, comment_id).await
            }
            ActionType::Follow => {
                let followee_id = message.as_i64().unwrap_or(0);
                self.follow(agent_id, followee_id).await
            }
            ActionType::Unfollow => {
                let followee_id = message.as_i64().unwrap_or(0);
                self.unfollow(agent_id, followee_id).await
            }
            ActionType::Mute => {
                let mutee_id = message.as_i64().unwrap_or(0);
                self.mute(agent_id, mutee_id).await
            }
            ActionType::Unmute => {
                let mutee_id = message.as_i64().unwrap_or(0);
                self.unmute(agent_id, mutee_id).await
            }
            ActionType::Refresh => self.refresh(agent_id).await,
            ActionType::SearchPosts => {
                let query = message.as_str().unwrap_or("").to_string();
                self.search_posts(agent_id, query).await
            }
            ActionType::SearchUser => {
                let query = message.as_str().unwrap_or("").to_string();
                self.search_user(agent_id, query).await
            }
            ActionType::Trend => self.trend(agent_id).await,
            ActionType::CreateGroup => {
                let name = message.as_str().unwrap_or("").to_string();
                self.create_group(agent_id, name).await
            }
            ActionType::JoinGroup => {
                let group_id = message.as_i64().unwrap_or(0);
                self.join_group(agent_id, group_id).await
            }
            ActionType::LeaveGroup => {
                let group_id = message.as_i64().unwrap_or(0);
                self.leave_group(agent_id, group_id).await
            }
            ActionType::SendToGroup => self.send_to_group(agent_id, message).await,
            ActionType::ListenFromGroup => self.listen_from_group(agent_id).await,
            ActionType::ReportPost => self.report_post(agent_id, message).await,
            ActionType::PurchaseProduct => self.purchase_product(agent_id, message).await,
            ActionType::Interview => self.interview(agent_id, message).await,
            ActionType::DoNothing => self.do_nothing(agent_id).await,
            ActionType::UpdateRecTable => {
                self.update_rec_table().await;
                ActionResult::ok(json!({}))
            }
            ActionType::Exit => unreachable!("Exit handled in running()"),
        }
    }

    /// Get the current timestamp string.
    ///
    /// Return the current (possibly accelerated) time as a `DateTime<Utc>`.
    ///
    /// For Reddit mode (k > 1), uses clock.time_transfer() for accelerated simulated time.
    /// For Twitter mode (k = 1), uses real wall-clock time.
    ///
    /// Uses try_read() to avoid holding the clock lock across await points,
    /// preventing deadlocks when called while the db mutex is held.
    fn now_datetime(&self) -> chrono::DateTime<chrono::Utc> {
        if let Ok(clock) = self.clock.try_read() {
            if clock.k > 1 {
                let now = chrono::Utc::now();
                return clock.time_transfer(now, clock.real_start_time);
            }
        }
        chrono::Utc::now()
    }

    fn now_str(&self) -> String {
        self.now_datetime().format("%Y-%m-%d %H:%M:%S").to_string()
    }

    // ========================================================================
    // User Management
    // ========================================================================

    async fn sign_up(&self, agent_id: i64, user_message: Value) -> ActionResult {
        let db = self.db.lock().await;
        let user_name = user_message.get("user_name").and_then(|v| v.as_str()).unwrap_or("");
        let name = user_message.get("name").and_then(|v| v.as_str()).unwrap_or("");
        let bio = user_message.get("bio").and_then(|v| v.as_str()).unwrap_or("");
        let now = self.now_str();

        match db.execute(
            "INSERT INTO user (agent_id, user_name, name, bio, created_at) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![agent_id, user_name, name, bio, now],
        ) {
            Ok(_) => {
                let user_id = db.last_insert_rowid();
                let _ = queries::record_trace(&db, user_id, "sign_up", &format!("user_id: {}", user_id), &now);
                ActionResult::ok(json!({ "user_id": user_id }))
            }
            Err(e) => ActionResult::fail(&format!("sign_up failed: {}", e)),
        }
    }

    async fn follow(&self, agent_id: i64, followee_id: i64) -> ActionResult {
        let db = self.db.lock().await;
        let user_id = match queries::check_agent_userid(&db, agent_id) {
            Ok(Some(id)) => id,
            _ => return ActionResult::fail("agent not found"),
        };
        let now = self.now_str();

        if queries::exists(&db, "SELECT 1 FROM follow WHERE follower_id = ?1 AND followee_id = ?2 LIMIT 1", &[&user_id, &followee_id]) {
            return ActionResult::fail("already following");
        }

        match db.execute(
            "INSERT INTO follow (follower_id, followee_id, created_at) VALUES (?1, ?2, ?3)",
            params![user_id, followee_id, now],
        ) {
            Ok(_) => {
                let follow_id = db.last_insert_rowid();
                let _ = db.execute(
                    "UPDATE user SET num_followings = num_followings + 1 WHERE user_id = ?1",
                    params![user_id],
                );
                let _ = db.execute(
                    "UPDATE user SET num_followers = num_followers + 1 WHERE user_id = ?1",
                    params![followee_id],
                );
                let _ = queries::record_trace(&db, user_id, "follow", &format!("followee_id: {}", followee_id), &now);
                ActionResult::ok(json!({ "follow_id": follow_id }))
            }
            Err(e) => ActionResult::fail(&format!("follow failed: {}", e)),
        }
    }

    async fn unfollow(&self, agent_id: i64, followee_id: i64) -> ActionResult {
        let db = self.db.lock().await;
        let user_id = match queries::check_agent_userid(&db, agent_id) {
            Ok(Some(id)) => id,
            _ => return ActionResult::fail("agent not found"),
        };
        let now = self.now_str();

        let follow_id: Option<i64> = db
            .query_row(
                "SELECT follow_id FROM follow WHERE follower_id = ?1 AND followee_id = ?2",
                params![user_id, followee_id],
                |row| row.get(0),
            )
            .ok();

        match follow_id {
            Some(fid) => {
                let _ = db.execute("DELETE FROM follow WHERE follow_id = ?1", params![fid]);
                let _ = db.execute(
                    "UPDATE user SET num_followings = num_followings - 1 WHERE user_id = ?1",
                    params![user_id],
                );
                let _ = db.execute(
                    "UPDATE user SET num_followers = num_followers - 1 WHERE user_id = ?1",
                    params![followee_id],
                );
                let _ = queries::record_trace(&db, user_id, "unfollow", &format!("followee_id: {}", followee_id), &now);
                ActionResult::ok(json!({ "follow_id": fid }))
            }
            None => ActionResult::fail("not following"),
        }
    }

    async fn mute(&self, agent_id: i64, mutee_id: i64) -> ActionResult {
        let db = self.db.lock().await;
        let user_id = match queries::check_agent_userid(&db, agent_id) {
            Ok(Some(id)) => id,
            _ => return ActionResult::fail("agent not found"),
        };
        let now = self.now_str();

        if queries::exists(&db, "SELECT 1 FROM mute WHERE muter_id = ?1 AND mutee_id = ?2 LIMIT 1", &[&user_id, &mutee_id]) {
            return ActionResult::fail("already muted");
        }

        match db.execute(
            "INSERT INTO mute (muter_id, mutee_id, created_at) VALUES (?1, ?2, ?3)",
            params![user_id, mutee_id, now],
        ) {
            Ok(_) => {
                let _mute_id = db.last_insert_rowid();
                let _ = queries::record_trace(&db, user_id, "mute", &format!("mutee_id: {}", mutee_id), &now);
                ActionResult::ok(json!({ "mutee_id": mutee_id }))
            }
            Err(e) => ActionResult::fail(&format!("mute failed: {}", e)),
        }
    }

    async fn unmute(&self, agent_id: i64, mutee_id: i64) -> ActionResult {
        let db = self.db.lock().await;
        let user_id = match queries::check_agent_userid(&db, agent_id) {
            Ok(Some(id)) => id,
            _ => return ActionResult::fail("agent not found"),
        };
        let now = self.now_str();

        match db.execute(
            "DELETE FROM mute WHERE muter_id = ?1 AND mutee_id = ?2",
            params![user_id, mutee_id],
        ) {
            Ok(0) => ActionResult::fail("not muted"),
            Ok(_) => {
                let _ = queries::record_trace(&db, user_id, "unmute", &format!("mutee_id: {}", mutee_id), &now);
                ActionResult::ok(json!({ "mutee_id": mutee_id }))
            }
            Err(e) => ActionResult::fail(&format!("unmute failed: {}", e)),
        }
    }

    // ========================================================================
    // Content Creation
    // ========================================================================

    async fn create_post(&self, agent_id: i64, content: String) -> ActionResult {
        let db = self.db.lock().await;
        let user_id = match queries::check_agent_userid(&db, agent_id) {
            Ok(Some(id)) => id,
            _ => return ActionResult::fail("agent not found"),
        };
        let now = self.now_str();

        match db.execute(
            "INSERT INTO post (user_id, content, created_at) VALUES (?1, ?2, ?3)",
            params![user_id, content, now],
        ) {
            Ok(_) => {
                let post_id = db.last_insert_rowid();
                let _ = queries::record_trace(&db, user_id, "create_post", &format!("{TRACE_LIKE_POST_PREFIX}{post_id}"), &now);
                ActionResult::ok(json!({ "post_id": post_id }))
            }
            Err(e) => ActionResult::fail(&format!("create_post failed: {}", e)),
        }
    }

    async fn repost(&self, agent_id: i64, post_id: i64) -> ActionResult {
        let db = self.db.lock().await;
        let user_id = match queries::check_agent_userid(&db, agent_id) {
            Ok(Some(id)) => id,
            _ => return ActionResult::fail("agent not found"),
        };
        let now = self.now_str();

        if queries::exists(&db, "SELECT 1 FROM post WHERE user_id = ?1 AND original_post_id = ?2 AND quote_content IS NULL LIMIT 1", &[&user_id, &post_id]) {
            return ActionResult::fail("already reposted");
        }

        match db.execute(
            "INSERT INTO post (user_id, original_post_id, created_at) VALUES (?1, ?2, ?3)",
            params![user_id, post_id, now],
        ) {
            Ok(_) => {
                let new_post_id = db.last_insert_rowid();
                let _ = db.execute(
                    "UPDATE post SET num_shares = num_shares + 1 WHERE post_id = ?1",
                    params![post_id],
                );
                let _ = queries::record_trace(&db, user_id, "repost", &format!("post_id: {}", new_post_id), &now);
                ActionResult::ok(json!({ "post_id": new_post_id }))
            }
            Err(e) => ActionResult::fail(&format!("repost failed: {}", e)),
        }
    }

    async fn quote_post(&self, agent_id: i64, quote_message: Value) -> ActionResult {
        let db = self.db.lock().await;
        let user_id = match queries::check_agent_userid(&db, agent_id) {
            Ok(Some(id)) => id,
            _ => return ActionResult::fail("agent not found"),
        };
        let post_id = quote_message.get("post_id").and_then(|v| v.as_i64()).unwrap_or(0);
        let quote_content = quote_message.get("quote_content").and_then(|v| v.as_str()).unwrap_or("");
        let now = self.now_str();

        match db.execute(
            "INSERT INTO post (user_id, original_post_id, quote_content, created_at) VALUES (?1, ?2, ?3, ?4)",
            params![user_id, post_id, quote_content, now],
        ) {
            Ok(_) => {
                let new_post_id = db.last_insert_rowid();
                let _ = db.execute(
                    "UPDATE post SET num_shares = num_shares + 1 WHERE post_id = ?1",
                    params![post_id],
                );
                let _ = queries::record_trace(&db, user_id, "quote_post", &format!("post_id: {}", new_post_id), &now);
                ActionResult::ok(json!({ "post_id": new_post_id }))
            }
            Err(e) => ActionResult::fail(&format!("quote_post failed: {}", e)),
        }
    }

    async fn create_comment(&self, agent_id: i64, comment_message: Value) -> ActionResult {
        let db = self.db.lock().await;
        let user_id = match queries::check_agent_userid(&db, agent_id) {
            Ok(Some(id)) => id,
            _ => return ActionResult::fail("agent not found"),
        };
        let post_id = comment_message.get("post_id").and_then(|v| v.as_i64()).unwrap_or(0);
        let content = comment_message.get("content").and_then(|v| v.as_str()).unwrap_or("");
        let now = self.now_str();

        match db.execute(
            "INSERT INTO comment (post_id, user_id, content, created_at) VALUES (?1, ?2, ?3, ?4)",
            params![post_id, user_id, content, now],
        ) {
            Ok(_) => {
                let comment_id = db.last_insert_rowid();
                let _ = queries::record_trace(&db, user_id, "create_comment", &format!("comment_id: {}", comment_id), &now);
                ActionResult::ok(json!({ "comment_id": comment_id }))
            }
            Err(e) => ActionResult::fail(&format!("create_comment failed: {}", e)),
        }
    }

    // ========================================================================
    // Post Engagement
    // ========================================================================

    async fn like_post(&self, agent_id: i64, post_id: i64) -> ActionResult {
        let db = self.db.lock().await;
        let user_id = match queries::check_agent_userid(&db, agent_id) {
            Ok(Some(id)) => id,
            _ => return ActionResult::fail("agent not found"),
        };
        let now = self.now_str();

        if !self.config.allow_self_rating && queries::is_own_post(&db, post_id, user_id) {
            return ActionResult::fail("cannot rate own post");
        }

        if queries::exists(&db, "SELECT 1 FROM \"like\" WHERE user_id = ?1 AND post_id = ?2 LIMIT 1", &[&user_id, &post_id]) {
            return ActionResult::fail("already liked");
        }

        match db.execute(
            "INSERT INTO \"like\" (user_id, post_id, created_at) VALUES (?1, ?2, ?3)",
            params![user_id, post_id, now],
        ) {
            Ok(_) => {
                let like_id = db.last_insert_rowid();
                let _ = db.execute(
                    "UPDATE post SET num_likes = num_likes + 1 WHERE post_id = ?1",
                    params![post_id],
                );
                let _ = queries::record_trace(&db, user_id, "like_post", &format!("{TRACE_LIKE_POST_PREFIX}{post_id}"), &now);
                ActionResult::ok(json!({ "like_id": like_id }))
            }
            Err(e) => ActionResult::fail(&format!("like_post failed: {}", e)),
        }
    }

    async fn unlike_post(&self, agent_id: i64, post_id: i64) -> ActionResult {
        let db = self.db.lock().await;
        let user_id = match queries::check_agent_userid(&db, agent_id) {
            Ok(Some(id)) => id,
            _ => return ActionResult::fail("agent not found"),
        };
        let now = self.now_str();

        let like_id: Option<i64> = db
            .query_row(
                "SELECT like_id FROM \"like\" WHERE user_id = ?1 AND post_id = ?2",
                params![user_id, post_id],
                |row| row.get(0),
            )
            .ok();

        match like_id {
            Some(lid) => {
                let _ = db.execute("DELETE FROM \"like\" WHERE like_id = ?1", params![lid]);
                let _ = db.execute(
                    "UPDATE post SET num_likes = num_likes - 1 WHERE post_id = ?1",
                    params![post_id],
                );
                let _ = queries::record_trace(&db, user_id, "unlike_post", &format!("{TRACE_LIKE_POST_PREFIX}{post_id}"), &now);
                ActionResult::ok(json!({ "like_id": lid }))
            }
            None => ActionResult::fail("not liked"),
        }
    }

    async fn dislike_post(&self, agent_id: i64, post_id: i64) -> ActionResult {
        let db = self.db.lock().await;
        let user_id = match queries::check_agent_userid(&db, agent_id) {
            Ok(Some(id)) => id,
            _ => return ActionResult::fail("agent not found"),
        };
        let now = self.now_str();

        if !self.config.allow_self_rating && queries::is_own_post(&db, post_id, user_id) {
            return ActionResult::fail("cannot rate own post");
        }

        if queries::exists(&db, "SELECT 1 FROM dislike WHERE user_id = ?1 AND post_id = ?2 LIMIT 1", &[&user_id, &post_id]) {
            return ActionResult::fail("already disliked");
        }

        match db.execute(
            "INSERT INTO dislike (user_id, post_id, created_at) VALUES (?1, ?2, ?3)",
            params![user_id, post_id, now],
        ) {
            Ok(_) => {
                let dislike_id = db.last_insert_rowid();
                let _ = db.execute(
                    "UPDATE post SET num_dislikes = num_dislikes + 1 WHERE post_id = ?1",
                    params![post_id],
                );
                let _ = queries::record_trace(&db, user_id, "dislike_post", &format!("{TRACE_LIKE_POST_PREFIX}{post_id}"), &now);
                ActionResult::ok(json!({ "dislike_id": dislike_id }))
            }
            Err(e) => ActionResult::fail(&format!("dislike_post failed: {}", e)),
        }
    }

    async fn undo_dislike_post(&self, agent_id: i64, post_id: i64) -> ActionResult {
        let db = self.db.lock().await;
        let user_id = match queries::check_agent_userid(&db, agent_id) {
            Ok(Some(id)) => id,
            _ => return ActionResult::fail("agent not found"),
        };
        let now = self.now_str();

        let dislike_id: Option<i64> = db
            .query_row(
                "SELECT dislike_id FROM dislike WHERE user_id = ?1 AND post_id = ?2",
                params![user_id, post_id],
                |row| row.get(0),
            )
            .ok();

        match dislike_id {
            Some(did) => {
                let _ = db.execute("DELETE FROM dislike WHERE dislike_id = ?1", params![did]);
                let _ = db.execute(
                    "UPDATE post SET num_dislikes = num_dislikes - 1 WHERE post_id = ?1",
                    params![post_id],
                );
                let _ = queries::record_trace(&db, user_id, "undo_dislike_post", &format!("{TRACE_LIKE_POST_PREFIX}{post_id}"), &now);
                ActionResult::ok(json!({ "dislike_id": did }))
            }
            None => ActionResult::fail("not disliked"),
        }
    }

    // ========================================================================
    // Comment Engagement
    // ========================================================================

    async fn like_comment(&self, agent_id: i64, comment_id: i64) -> ActionResult {
        let db = self.db.lock().await;
        let user_id = match queries::check_agent_userid(&db, agent_id) {
            Ok(Some(id)) => id,
            _ => return ActionResult::fail("agent not found"),
        };
        let now = self.now_str();

        if !self.config.allow_self_rating && queries::is_own_comment(&db, comment_id, user_id) {
            return ActionResult::fail("cannot rate own comment");
        }

        if queries::exists(&db, "SELECT 1 FROM comment_like WHERE user_id = ?1 AND comment_id = ?2 LIMIT 1", &[&user_id, &comment_id]) {
            return ActionResult::fail("already liked");
        }

        match db.execute(
            "INSERT INTO comment_like (user_id, comment_id, created_at) VALUES (?1, ?2, ?3)",
            params![user_id, comment_id, now],
        ) {
            Ok(_) => {
                let like_id = db.last_insert_rowid();
                let _ = db.execute(
                    "UPDATE comment SET num_likes = num_likes + 1 WHERE comment_id = ?1",
                    params![comment_id],
                );
                let _ = queries::record_trace(&db, user_id, "like_comment", &format!("comment_id: {}", comment_id), &now);
                ActionResult::ok(json!({ "comment_like_id": like_id }))
            }
            Err(e) => ActionResult::fail(&format!("like_comment failed: {}", e)),
        }
    }

    async fn unlike_comment(&self, agent_id: i64, comment_id: i64) -> ActionResult {
        let db = self.db.lock().await;
        let user_id = match queries::check_agent_userid(&db, agent_id) {
            Ok(Some(id)) => id,
            _ => return ActionResult::fail("agent not found"),
        };
        let now = self.now_str();

        let like_id: Option<i64> = db
            .query_row(
                "SELECT comment_like_id FROM comment_like WHERE user_id = ?1 AND comment_id = ?2",
                params![user_id, comment_id],
                |row| row.get(0),
            )
            .ok();

        match like_id {
            Some(lid) => {
                let _ = db.execute("DELETE FROM comment_like WHERE comment_like_id = ?1", params![lid]);
                let _ = db.execute(
                    "UPDATE comment SET num_likes = num_likes - 1 WHERE comment_id = ?1",
                    params![comment_id],
                );
                let _ = queries::record_trace(&db, user_id, "unlike_comment", &format!("comment_id: {}", comment_id), &now);
                ActionResult::ok(json!({ "comment_like_id": lid }))
            }
            None => ActionResult::fail("not liked"),
        }
    }

    async fn dislike_comment(&self, agent_id: i64, comment_id: i64) -> ActionResult {
        let db = self.db.lock().await;
        let user_id = match queries::check_agent_userid(&db, agent_id) {
            Ok(Some(id)) => id,
            _ => return ActionResult::fail("agent not found"),
        };
        let now = self.now_str();

        if !self.config.allow_self_rating && queries::is_own_comment(&db, comment_id, user_id) {
            return ActionResult::fail("cannot rate own comment");
        }

        if queries::exists(&db, "SELECT 1 FROM comment_dislike WHERE user_id = ?1 AND comment_id = ?2 LIMIT 1", &[&user_id, &comment_id]) {
            return ActionResult::fail("already disliked");
        }

        match db.execute(
            "INSERT INTO comment_dislike (user_id, comment_id, created_at) VALUES (?1, ?2, ?3)",
            params![user_id, comment_id, now],
        ) {
            Ok(_) => {
                let dislike_id = db.last_insert_rowid();
                let _ = db.execute(
                    "UPDATE comment SET num_dislikes = num_dislikes + 1 WHERE comment_id = ?1",
                    params![comment_id],
                );
                let _ = queries::record_trace(&db, user_id, "dislike_comment", &format!("comment_id: {}", comment_id), &now);
                ActionResult::ok(json!({ "comment_dislike_id": dislike_id }))
            }
            Err(e) => ActionResult::fail(&format!("dislike_comment failed: {}", e)),
        }
    }

    async fn undo_dislike_comment(&self, agent_id: i64, comment_id: i64) -> ActionResult {
        let db = self.db.lock().await;
        let user_id = match queries::check_agent_userid(&db, agent_id) {
            Ok(Some(id)) => id,
            _ => return ActionResult::fail("agent not found"),
        };
        let now = self.now_str();

        let dislike_id: Option<i64> = db
            .query_row(
                "SELECT comment_dislike_id FROM comment_dislike WHERE user_id = ?1 AND comment_id = ?2",
                params![user_id, comment_id],
                |row| row.get(0),
            )
            .ok();

        match dislike_id {
            Some(did) => {
                let _ = db.execute("DELETE FROM comment_dislike WHERE comment_dislike_id = ?1", params![did]);
                let _ = db.execute(
                    "UPDATE comment SET num_dislikes = num_dislikes - 1 WHERE comment_id = ?1",
                    params![comment_id],
                );
                let _ = queries::record_trace(&db, user_id, "undo_dislike_comment", &format!("comment_id: {}", comment_id), &now);
                ActionResult::ok(json!({ "comment_dislike_id": did }))
            }
            None => ActionResult::fail("not disliked"),
        }
    }

    /// Apply show_score transformation: replace num_likes/num_dislikes with score.
    fn apply_show_score(&self, mut posts: Vec<Value>) -> Vec<Value> {
        if !self.config.show_score {
            return posts;
        }
        for post in &mut posts {
            if let Some(obj) = post.as_object_mut() {
                let likes = obj.get("num_likes").and_then(|v| v.as_i64()).unwrap_or(0);
                let dislikes = obj.get("num_dislikes").and_then(|v| v.as_i64()).unwrap_or(0);
                obj.remove("num_likes");
                obj.remove("num_dislikes");
                obj.insert("score".to_string(), json!(likes - dislikes));
            }
        }
        posts
    }

    // ========================================================================
    // Discovery
    // ========================================================================

    async fn refresh(&self, agent_id: i64) -> ActionResult {
        let db = self.db.lock().await;
        let user_id = match queries::check_agent_userid(&db, agent_id) {
            Ok(Some(id)) => id,
            _ => return ActionResult::fail("agent not found"),
        };
        let now = self.now_str();

        // Get recommended posts from rec table
        let mut rec_posts: Vec<Value> = match db.prepare(
            "SELECT p.post_id, p.user_id, p.content, p.created_at, p.num_likes, p.num_dislikes, p.num_shares \
             FROM rec r JOIN post p ON r.post_id = p.post_id \
             WHERE r.user_id = ?1 LIMIT ?2",
        ) {
            Ok(mut stmt) => stmt
                .query_map(params![user_id, self.config.refresh_rec_post_count], |row| {
                    Ok(json!({
                        "post_id": row.get::<_, i64>(0)?,
                        "user_id": row.get::<_, i64>(1)?,
                        "content": row.get::<_, String>(2)?,
                        "created_at": row.get::<_, String>(3)?,
                        "num_likes": row.get::<_, i64>(4)?,
                        "num_dislikes": row.get::<_, i64>(5)?,
                        "num_shares": row.get::<_, i64>(6)?,
                    }))
                })
                .map(|rows| rows.filter_map(|r| r.ok()).collect())
                .unwrap_or_default(),
            Err(e) => {
                return ActionResult::fail(&format!("refresh rec query failed: {}", e));
            }
        };

        // Get posts from followed users
        let following_posts: Vec<Value> = match db.prepare(
            "SELECT p.post_id, p.user_id, p.content, p.created_at, p.num_likes, p.num_dislikes, p.num_shares \
             FROM post p JOIN follow f ON p.user_id = f.followee_id \
             WHERE f.follower_id = ?1 \
             ORDER BY p.created_at DESC LIMIT ?2",
        ) {
            Ok(mut stmt) => stmt
                .query_map(params![user_id, self.config.following_post_count], |row| {
                    Ok(json!({
                        "post_id": row.get::<_, i64>(0)?,
                        "user_id": row.get::<_, i64>(1)?,
                        "content": row.get::<_, String>(2)?,
                        "created_at": row.get::<_, String>(3)?,
                        "num_likes": row.get::<_, i64>(4)?,
                        "num_dislikes": row.get::<_, i64>(5)?,
                        "num_shares": row.get::<_, i64>(6)?,
                    }))
                })
                .map(|rows| rows.filter_map(|r| r.ok()).collect())
                .unwrap_or_default(),
            Err(e) => {
                return ActionResult::fail(&format!("refresh follow query failed: {}", e));
            }
        };

        rec_posts.extend(following_posts);

        // Add comments to posts
        let posts_with_comments = queries::add_comments_to_posts(&db, &rec_posts).unwrap_or(rec_posts);
        let posts_final = self.apply_show_score(posts_with_comments);

        let _ = queries::record_trace(&db, user_id, "refresh", "", &now);
        ActionResult::ok(json!({ "posts": posts_final }))
    }

    async fn search_posts(&self, agent_id: i64, query: String) -> ActionResult {
        let db = self.db.lock().await;
        let user_id = match queries::check_agent_userid(&db, agent_id) {
            Ok(Some(id)) => id,
            _ => return ActionResult::fail("agent not found"),
        };
        let now = self.now_str();
        let search = format!("%{}%", query);

        let posts: Vec<Value> = match db.prepare(
            "SELECT p.post_id, p.user_id, p.content, p.created_at, p.num_likes, p.num_dislikes \
             FROM post p LEFT JOIN user u ON p.user_id = u.user_id \
             WHERE p.content LIKE ?1 OR CAST(p.post_id AS TEXT) = ?2 OR CAST(p.user_id AS TEXT) = ?2 \
             OR u.user_name LIKE ?1 OR u.bio LIKE ?1",
        ) {
            Ok(mut stmt) => stmt
                .query_map(params![search, query], |row| {
                    Ok(json!({
                        "post_id": row.get::<_, i64>(0)?,
                        "user_id": row.get::<_, i64>(1)?,
                        "content": row.get::<_, String>(2)?,
                        "created_at": row.get::<_, String>(3)?,
                        "num_likes": row.get::<_, i64>(4)?,
                        "num_dislikes": row.get::<_, i64>(5)?,
                    }))
                })
                .map(|rows| rows.filter_map(|r| r.ok()).collect())
                .unwrap_or_default(),
            Err(e) => return ActionResult::fail(&format!("search_posts failed: {}", e)),
        };

        let posts = self.apply_show_score(posts);
        let _ = queries::record_trace(&db, user_id, "search_posts", &query, &now);
        ActionResult::ok(json!({ "posts": posts }))
    }

    async fn search_user(&self, agent_id: i64, query: String) -> ActionResult {
        let db = self.db.lock().await;
        let user_id = match queries::check_agent_userid(&db, agent_id) {
            Ok(Some(id)) => id,
            _ => return ActionResult::fail("agent not found"),
        };
        let now = self.now_str();
        let search = format!("%{}%", query);

        let users: Vec<Value> = match db.prepare(
            "SELECT user_id, user_name, name, bio, created_at, num_followings, num_followers \
             FROM user WHERE user_name LIKE ?1 OR name LIKE ?1 OR bio LIKE ?1 \
             OR CAST(user_id AS TEXT) = ?2 OR CAST(agent_id AS TEXT) = ?2 \
             ORDER BY CASE WHEN CAST(user_id AS TEXT) = ?2 OR CAST(agent_id AS TEXT) = ?2 THEN 0 ELSE 1 END",
        ) {
            Ok(mut stmt) => stmt
                .query_map(params![search, query], |row| {
                    Ok(json!({
                        "user_id": row.get::<_, i64>(0)?,
                        "user_name": row.get::<_, String>(1)?,
                        "name": row.get::<_, String>(2)?,
                        "bio": row.get::<_, String>(3)?,
                        "created_at": row.get::<_, String>(4)?,
                        "num_followings": row.get::<_, i64>(5)?,
                        "num_followers": row.get::<_, i64>(6)?,
                    }))
                })
                .map(|rows| rows.filter_map(|r| r.ok()).collect())
                .unwrap_or_default(),
            Err(e) => return ActionResult::fail(&format!("search_user failed: {}", e)),
        };

        let _ = queries::record_trace(&db, user_id, "search_user", &query, &now);
        ActionResult::ok(json!({ "users": users }))
    }

    async fn trend(&self, _agent_id: i64) -> ActionResult {
        let db = self.db.lock().await;
        let top_k = self.config.trend_top_k;
        let num_days = self.config.trend_num_days;

        // Filter to posts within trend_num_days using simulated time (matches OASIS)
        let cutoff = (self.now_datetime() - chrono::Duration::days(num_days as i64))
            .format("%Y-%m-%d %H:%M:%S")
            .to_string();

        let posts: Vec<Value> = match db.prepare(
            "SELECT post_id, user_id, content, created_at, num_likes, num_dislikes \
             FROM post WHERE created_at >= ?1 ORDER BY num_likes DESC LIMIT ?2",
        ) {
            Ok(mut stmt) => stmt
                .query_map(params![cutoff, top_k], |row| {
                    Ok(json!({
                        "post_id": row.get::<_, i64>(0)?,
                        "user_id": row.get::<_, i64>(1)?,
                        "content": row.get::<_, String>(2)?,
                        "created_at": row.get::<_, String>(3)?,
                        "num_likes": row.get::<_, i64>(4)?,
                        "num_dislikes": row.get::<_, i64>(5)?,
                    }))
                })
                .map(|rows| rows.filter_map(|r| r.ok()).collect())
                .unwrap_or_default(),
            Err(e) => return ActionResult::fail(&format!("trend failed: {}", e)),
        };

        let posts = self.apply_show_score(posts);
        ActionResult::ok(json!({ "posts": posts }))
    }

    // ========================================================================
    // Group Chat
    // ========================================================================

    async fn create_group(&self, agent_id: i64, group_name: String) -> ActionResult {
        let db = self.db.lock().await;
        let now = self.now_str();

        match db.execute(
            "INSERT INTO chat_group (name, created_at) VALUES (?1, ?2)",
            params![group_name, now],
        ) {
            Ok(_) => {
                let group_id = db.last_insert_rowid();
                // Creator automatically joins
                let _ = db.execute(
                    "INSERT INTO group_members (group_id, agent_id) VALUES (?1, ?2)",
                    params![group_id, agent_id],
                );
                ActionResult::ok(json!({ "group_id": group_id }))
            }
            Err(e) => ActionResult::fail(&format!("create_group failed: {}", e)),
        }
    }

    async fn join_group(&self, agent_id: i64, group_id: i64) -> ActionResult {
        let db = self.db.lock().await;

        match db.execute(
            "INSERT OR IGNORE INTO group_members (group_id, agent_id) VALUES (?1, ?2)",
            params![group_id, agent_id],
        ) {
            Ok(_) => ActionResult::ok(json!({})),
            Err(e) => ActionResult::fail(&format!("join_group failed: {}", e)),
        }
    }

    async fn leave_group(&self, agent_id: i64, group_id: i64) -> ActionResult {
        let db = self.db.lock().await;

        match db.execute(
            "DELETE FROM group_members WHERE group_id = ?1 AND agent_id = ?2",
            params![group_id, agent_id],
        ) {
            Ok(_) => ActionResult::ok(json!({})),
            Err(e) => ActionResult::fail(&format!("leave_group failed: {}", e)),
        }
    }

    async fn send_to_group(&self, agent_id: i64, message: Value) -> ActionResult {
        let db = self.db.lock().await;
        let group_id = message.get("group_id").and_then(|v| v.as_i64()).unwrap_or(0);
        let content = message.get("message").and_then(|v| v.as_str()).unwrap_or("");

        match db.execute(
            "INSERT INTO group_messages (group_id, sender_id, content) VALUES (?1, ?2, ?3)",
            params![group_id, agent_id, content],
        ) {
            Ok(_) => {
                let message_id = db.last_insert_rowid();
                ActionResult::ok(json!({ "message_id": message_id }))
            }
            Err(e) => ActionResult::fail(&format!("send_to_group failed: {}", e)),
        }
    }

    async fn listen_from_group(&self, agent_id: i64) -> ActionResult {
        let db = self.db.lock().await;

        let messages: Vec<Value> = {
            let mut stmt = db
                .prepare(
                    "SELECT gm.message_id, gm.group_id, gm.sender_id, gm.content, gm.sent_at, cg.name \
                     FROM group_messages gm \
                     JOIN group_members mem ON gm.group_id = mem.group_id \
                     JOIN chat_group cg ON gm.group_id = cg.group_id \
                     WHERE mem.agent_id = ?1 \
                     ORDER BY gm.sent_at DESC",
                )
                .unwrap();
            stmt.query_map(params![agent_id], |row| {
                Ok(json!({
                    "message_id": row.get::<_, i64>(0)?,
                    "group_id": row.get::<_, i64>(1)?,
                    "sender_id": row.get::<_, i64>(2)?,
                    "content": row.get::<_, String>(3)?,
                    "sent_at": row.get::<_, String>(4)?,
                    "group_name": row.get::<_, String>(5)?,
                }))
            })
            .unwrap()
            .filter_map(|r| r.ok())
            .collect()
        };

        ActionResult::ok(json!({ "messages": messages }))
    }

    // ========================================================================
    // Administrative / Special
    // ========================================================================

    async fn report_post(&self, agent_id: i64, report_message: Value) -> ActionResult {
        let db = self.db.lock().await;
        let user_id = match queries::check_agent_userid(&db, agent_id) {
            Ok(Some(id)) => id,
            _ => return ActionResult::fail("agent not found"),
        };
        let post_id = report_message.get("post_id").and_then(|v| v.as_i64()).unwrap_or(0);
        let reason = report_message.get("report_reason").and_then(|v| v.as_str()).unwrap_or("");
        let now = self.now_str();

        match db.execute(
            "INSERT INTO report (user_id, post_id, report_reason, created_at) VALUES (?1, ?2, ?3, ?4)",
            params![user_id, post_id, reason, now],
        ) {
            Ok(_) => {
                let report_id = db.last_insert_rowid();
                let _ = db.execute(
                    "UPDATE post SET num_reports = num_reports + 1 WHERE post_id = ?1",
                    params![post_id],
                );

                // Auto-delete post if reports meet or exceed threshold (matches OASIS)
                let deleted = db.execute(
                    "DELETE FROM post WHERE post_id = ?1 AND num_reports >= ?2",
                    params![post_id, self.config.report_threshold as i64],
                ).unwrap_or(0);
                if deleted > 0 {
                    tracing::info!(post_id, threshold = self.config.report_threshold, "Post auto-deleted: report threshold exceeded");
                }

                let _ = queries::record_trace(&db, user_id, "report_post", &format!("{TRACE_LIKE_POST_PREFIX}{post_id}"), &now);
                ActionResult::ok(json!({ "report_id": report_id }))
            }
            Err(e) => ActionResult::fail(&format!("report_post failed: {}", e)),
        }
    }

    async fn purchase_product(&self, _agent_id: i64, purchase_message: Value) -> ActionResult {
        let db = self.db.lock().await;
        let product_name = purchase_message.get("product_name").and_then(|v| v.as_str()).unwrap_or("");
        let purchase_num = purchase_message.get("purchase_num").and_then(|v| v.as_i64()).unwrap_or(1);

        match db.execute(
            "UPDATE product SET sales = sales + ?1 WHERE product_name = ?2",
            params![purchase_num, product_name],
        ) {
            Ok(_) => ActionResult::ok(json!({})),
            Err(e) => ActionResult::fail(&format!("purchase_product failed: {}", e)),
        }
    }

    async fn interview(&self, agent_id: i64, interview_data: Value) -> ActionResult {
        let db = self.db.lock().await;
        let now = self.now_str();

        if let Some(user_id) = queries::check_agent_userid(&db, agent_id).ok().flatten() {
            let _ = queries::record_trace(
                &db,
                user_id,
                "interview",
                &serde_json::to_string(&interview_data).unwrap_or_default(),
                &now,
            );
        }

        ActionResult::ok(json!({ "interview_id": uuid::Uuid::new_v4().to_string() }))
    }

    async fn do_nothing(&self, agent_id: i64) -> ActionResult {
        let db = self.db.lock().await;
        let now = self.now_str();

        if let Some(user_id) = queries::check_agent_userid(&db, agent_id).ok().flatten() {
            let _ = queries::record_trace(&db, user_id, "do_nothing", "", &now);
        }

        ActionResult::ok(json!({}))
    }

    // ========================================================================
    // Recommendation System
    // ========================================================================

    pub async fn update_rec_table(&self) {
        let db = self.db.lock().await;
        recsys::update_rec_table(&db, self.recsys_type, self.config.max_rec_post_len);
    }

    /// Register a product on the platform (matches OASIS's sign_up_product).
    ///
    /// Called during setup, not as an agent action. Products must be registered
    /// before agents can use PurchaseProduct.
    pub async fn sign_up_product(&self, product_id: i64, product_name: &str) -> ActionResult {
        let db = self.db.lock().await;
        match db.execute(
            "INSERT OR IGNORE INTO product (product_id, product_name, sales) VALUES (?1, ?2, 0)",
            params![product_id, product_name],
        ) {
            Ok(_) => ActionResult::ok(json!({ "product_id": product_id })),
            Err(e) => ActionResult::fail(&format!("sign_up_product failed: {}", e)),
        }
    }
}
