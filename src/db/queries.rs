use anyhow::Result;
use rusqlite::{params, Connection};
use serde_json::Value;

/// Check if an agent_id has a corresponding user_id, return it.
pub fn check_agent_userid(conn: &Connection, agent_id: i64) -> Result<Option<i64>> {
    let mut stmt = conn.prepare("SELECT user_id FROM user WHERE agent_id = ?1")?;
    let result = stmt
        .query_row(params![agent_id], |row| row.get::<_, i64>(0))
        .ok();
    Ok(result)
}

/// Get the type of a post: "common", "repost", or "quote".
pub fn get_post_type(conn: &Connection, post_id: i64) -> Result<String> {
    let mut stmt = conn.prepare(
        "SELECT original_post_id, quote_content FROM post WHERE post_id = ?1",
    )?;
    let (original_id, quote_content): (Option<i64>, Option<String>) =
        stmt.query_row(params![post_id], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })?;

    match (original_id, quote_content) {
        (None, _) => Ok("common".to_string()),
        (Some(_), Some(_)) => Ok("quote".to_string()),
        (Some(_), None) => Ok("repost".to_string()),
    }
}

/// Fetch all posts with their comments attached.
pub fn add_comments_to_posts(conn: &Connection, posts: &[Value]) -> Result<Vec<Value>> {
    let mut result = Vec::new();
    for post in posts {
        let mut post = post.clone();
        if let Some(post_id) = post.get("post_id").and_then(|v| v.as_i64()) {
            let mut stmt = conn.prepare(
                "SELECT comment_id, post_id, user_id, content, created_at, num_likes, num_dislikes \
                 FROM comment WHERE post_id = ?1 ORDER BY created_at ASC",
            )?;
            let comments: Vec<Value> = stmt
                .query_map(params![post_id], |row| {
                    Ok(serde_json::json!({
                        "comment_id": row.get::<_, i64>(0)?,
                        "post_id": row.get::<_, i64>(1)?,
                        "user_id": row.get::<_, i64>(2)?,
                        "content": row.get::<_, String>(3)?,
                        "created_at": row.get::<_, String>(4)?,
                        "num_likes": row.get::<_, i64>(5)?,
                        "num_dislikes": row.get::<_, i64>(6)?,
                    }))
                })?
                .filter_map(|r| r.ok())
                .collect();
            post["comments"] = Value::Array(comments);
        }
        result.push(post);
    }
    Ok(result)
}

/// Record an action in the trace table.
pub fn record_trace(
    conn: &Connection,
    user_id: i64,
    action: &str,
    info: &str,
    created_at: &str,
) -> Result<()> {
    conn.execute(
        "INSERT OR IGNORE INTO trace (user_id, created_at, action, info) VALUES (?1, ?2, ?3, ?4)",
        params![user_id, created_at, action, info],
    )?;
    Ok(())
}

/// Fetch table contents as a vector of JSON values.
pub fn fetch_table(conn: &Connection, table_name: &str) -> Result<Vec<Value>> {
    let mut stmt = conn.prepare(&format!("SELECT * FROM \"{}\"", table_name))?;
    let column_count = stmt.column_count();
    let column_names: Vec<String> = (0..column_count)
        .map(|i| stmt.column_name(i).unwrap_or("?").to_string())
        .collect();

    let rows: Vec<Value> = stmt
        .query_map([], |row| {
            let mut obj = serde_json::Map::new();
            for (i, name) in column_names.iter().enumerate() {
                let val: Value = match row.get_ref(i) {
                    Ok(rusqlite::types::ValueRef::Null) => Value::Null,
                    Ok(rusqlite::types::ValueRef::Integer(n)) => Value::from(n),
                    Ok(rusqlite::types::ValueRef::Real(f)) => {
                        serde_json::Number::from_f64(f)
                            .map(Value::Number)
                            .unwrap_or(Value::Null)
                    }
                    Ok(rusqlite::types::ValueRef::Text(s)) => {
                        Value::String(String::from_utf8_lossy(s).to_string())
                    }
                    Ok(rusqlite::types::ValueRef::Blob(b)) => {
                        Value::String(format!("<blob {} bytes>", b.len()))
                    }
                    Err(_) => Value::Null,
                };
                obj.insert(name.clone(), val);
            }
            Ok(Value::Object(obj))
        })?
        .filter_map(|r| r.ok())
        .collect();

    Ok(rows)
}

/// Print all database table contents (debug utility matching OASIS's print_db_contents).
pub fn print_db_contents(conn: &Connection) -> Result<()> {
    let tables: Vec<String> = conn
        .prepare("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")?
        .query_map([], |row| row.get(0))?
        .filter_map(|r| r.ok())
        .collect();

    for table in &tables {
        let rows = fetch_table(conn, table)?;
        println!("\n=== {} ({} rows) ===", table, rows.len());
        for row in &rows {
            println!("{}", serde_json::to_string_pretty(row)?);
        }
    }

    Ok(())
}
