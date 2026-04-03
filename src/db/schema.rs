use anyhow::Result;
use rusqlite::Connection;

/// Create all 17 tables matching the OASIS SQLite schema exactly.
pub fn create_all_tables(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "
        -- User accounts
        CREATE TABLE IF NOT EXISTS user (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_id INTEGER UNIQUE,
            user_name TEXT,
            name TEXT,
            bio TEXT,
            created_at DATETIME,
            num_followings INTEGER DEFAULT 0,
            num_followers INTEGER DEFAULT 0
        );

        -- Posts
        CREATE TABLE IF NOT EXISTS post (
            post_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            original_post_id INTEGER,
            content TEXT DEFAULT '',
            quote_content TEXT,
            created_at DATETIME,
            num_likes INTEGER DEFAULT 0,
            num_dislikes INTEGER DEFAULT 0,
            num_shares INTEGER DEFAULT 0,
            num_reports INTEGER DEFAULT 0,
            FOREIGN KEY(user_id) REFERENCES user(user_id),
            FOREIGN KEY(original_post_id) REFERENCES post(post_id)
        );

        -- Comments on posts
        CREATE TABLE IF NOT EXISTS comment (
            comment_id INTEGER PRIMARY KEY AUTOINCREMENT,
            post_id INTEGER,
            user_id INTEGER,
            content TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            num_likes INTEGER DEFAULT 0,
            num_dislikes INTEGER DEFAULT 0,
            FOREIGN KEY(post_id) REFERENCES post(post_id),
            FOREIGN KEY(user_id) REFERENCES user(user_id)
        );

        -- Follow relationships
        CREATE TABLE IF NOT EXISTS follow (
            follow_id INTEGER PRIMARY KEY AUTOINCREMENT,
            follower_id INTEGER,
            followee_id INTEGER,
            created_at DATETIME,
            FOREIGN KEY(follower_id) REFERENCES user(user_id),
            FOREIGN KEY(followee_id) REFERENCES user(user_id)
        );

        -- Mute relationships
        CREATE TABLE IF NOT EXISTS mute (
            mute_id INTEGER PRIMARY KEY AUTOINCREMENT,
            muter_id INTEGER,
            mutee_id INTEGER,
            created_at DATETIME,
            FOREIGN KEY(muter_id) REFERENCES user(user_id),
            FOREIGN KEY(mutee_id) REFERENCES user(user_id)
        );

        -- Post likes
        CREATE TABLE IF NOT EXISTS \"like\" (
            like_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            post_id INTEGER,
            created_at DATETIME,
            FOREIGN KEY(user_id) REFERENCES user(user_id),
            FOREIGN KEY(post_id) REFERENCES post(post_id)
        );

        -- Post dislikes
        CREATE TABLE IF NOT EXISTS dislike (
            dislike_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            post_id INTEGER,
            created_at DATETIME,
            FOREIGN KEY(user_id) REFERENCES user(user_id),
            FOREIGN KEY(post_id) REFERENCES post(post_id)
        );

        -- Comment likes
        CREATE TABLE IF NOT EXISTS comment_like (
            comment_like_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            comment_id INTEGER,
            created_at DATETIME,
            FOREIGN KEY(user_id) REFERENCES user(user_id),
            FOREIGN KEY(comment_id) REFERENCES comment(comment_id)
        );

        -- Comment dislikes
        CREATE TABLE IF NOT EXISTS comment_dislike (
            comment_dislike_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            comment_id INTEGER,
            created_at DATETIME,
            FOREIGN KEY(user_id) REFERENCES user(user_id),
            FOREIGN KEY(comment_id) REFERENCES comment(comment_id)
        );

        -- Content reports
        CREATE TABLE IF NOT EXISTS report (
            report_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            post_id INTEGER,
            report_reason TEXT,
            created_at DATETIME,
            FOREIGN KEY(user_id) REFERENCES user(user_id),
            FOREIGN KEY(post_id) REFERENCES post(post_id)
        );

        -- Activity trace log
        CREATE TABLE IF NOT EXISTS trace (
            user_id INTEGER,
            created_at DATETIME,
            action TEXT,
            info TEXT,
            PRIMARY KEY(user_id, created_at, action, info),
            FOREIGN KEY(user_id) REFERENCES user(user_id)
        );

        -- Recommendation buffer
        CREATE TABLE IF NOT EXISTS rec (
            user_id INTEGER,
            post_id INTEGER,
            PRIMARY KEY(user_id, post_id),
            FOREIGN KEY(user_id) REFERENCES user(user_id),
            FOREIGN KEY(post_id) REFERENCES post(post_id)
        );

        -- Chat groups
        CREATE TABLE IF NOT EXISTS chat_group (
            group_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        -- Group membership
        CREATE TABLE IF NOT EXISTS group_members (
            group_id INTEGER NOT NULL,
            agent_id INTEGER NOT NULL,
            joined_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(group_id, agent_id),
            FOREIGN KEY(group_id) REFERENCES chat_group(group_id)
        );

        -- Group messages
        CREATE TABLE IF NOT EXISTS group_messages (
            message_id INTEGER PRIMARY KEY AUTOINCREMENT,
            group_id INTEGER NOT NULL,
            sender_id INTEGER NOT NULL,
            content TEXT NOT NULL,
            sent_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(group_id) REFERENCES chat_group(group_id),
            FOREIGN KEY(sender_id) REFERENCES user(agent_id)
        );

        -- Products (e-commerce)
        CREATE TABLE IF NOT EXISTS product (
            product_id INTEGER PRIMARY KEY,
            product_name TEXT,
            sales INTEGER DEFAULT 0
        );
        ",
    )?;

    Ok(())
}
