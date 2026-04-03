pub mod random;
pub mod reddit;
pub mod twitter;
pub mod twhin;

use rusqlite::{params, Connection};

use crate::types::RecsysType;

/// Row from the user table for recsys processing.
#[derive(Debug, Clone)]
pub struct UserRow {
    pub user_id: i64,
    pub bio: String,
}

/// Row from the post table for recsys processing.
#[derive(Debug, Clone)]
pub struct PostRow {
    pub post_id: i64,
    pub user_id: i64,
    pub content: String,
    pub created_at: String,
    pub num_likes: i64,
    pub num_dislikes: i64,
}

/// Row from the trace table for recsys processing.
#[derive(Debug, Clone)]
pub struct TraceRow {
    pub user_id: i64,
    pub action: String,
    pub info: String,
}

/// Update the recommendation table using the specified algorithm.
pub fn update_rec_table(conn: &Connection, recsys_type: RecsysType, max_rec_post_len: usize) {
    // Fetch data from database
    let user_table = fetch_user_table(conn);
    let post_table = fetch_post_table(conn);
    let trace_table = fetch_trace_table(conn);

    if user_table.is_empty() || post_table.is_empty() {
        return;
    }

    // Compute recommendations: Vec<(user_id, post_id)>
    let recommendations = match recsys_type {
        RecsysType::Random => random::recommend(&user_table, &post_table, max_rec_post_len),
        RecsysType::Reddit => reddit::recommend(&post_table, max_rec_post_len),
        RecsysType::Twitter => twitter::recommend(&user_table, &post_table, &trace_table, max_rec_post_len),
        RecsysType::Twhin => twhin::recommend(&user_table, &post_table, &trace_table, max_rec_post_len),
    };

    // Clear and rebuild rec table
    let _ = conn.execute("DELETE FROM rec", []);
    for (user_id, post_id) in &recommendations {
        let _ = conn.execute(
            "INSERT OR IGNORE INTO rec (user_id, post_id) VALUES (?1, ?2)",
            params![user_id, post_id],
        );
    }
}

fn fetch_user_table(conn: &Connection) -> Vec<UserRow> {
    let mut stmt = conn
        .prepare("SELECT user_id, COALESCE(bio, '') FROM user")
        .unwrap();
    stmt.query_map([], |row| {
        Ok(UserRow {
            user_id: row.get(0)?,
            bio: row.get(1)?,
        })
    })
    .unwrap()
    .filter_map(|r| r.ok())
    .collect()
}

fn fetch_post_table(conn: &Connection) -> Vec<PostRow> {
    let mut stmt = conn
        .prepare("SELECT post_id, user_id, COALESCE(content, ''), COALESCE(created_at, ''), num_likes, num_dislikes FROM post")
        .unwrap();
    stmt.query_map([], |row| {
        Ok(PostRow {
            post_id: row.get(0)?,
            user_id: row.get(1)?,
            content: row.get(2)?,
            created_at: row.get(3)?,
            num_likes: row.get(4)?,
            num_dislikes: row.get(5)?,
        })
    })
    .unwrap()
    .filter_map(|r| r.ok())
    .collect()
}

fn fetch_trace_table(conn: &Connection) -> Vec<TraceRow> {
    let mut stmt = conn
        .prepare("SELECT user_id, action, COALESCE(info, '') FROM trace")
        .unwrap();
    stmt.query_map([], |row| {
        Ok(TraceRow {
            user_id: row.get(0)?,
            action: row.get(1)?,
            info: row.get(2)?,
        })
    })
    .unwrap()
    .filter_map(|r| r.ok())
    .collect()
}
