use augur::db;
use rusqlite::params;

#[test]
fn test_schema_creates_all_tables() {
    let conn = db::open_and_init(":memory:").expect("failed to open in-memory DB");

    let tables: Vec<String> = conn
        .prepare(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name",
        )
        .unwrap()
        .query_map([], |row| row.get(0))
        .unwrap()
        .filter_map(|r| r.ok())
        .collect();

    let expected = vec![
        "chat_group",
        "comment",
        "comment_dislike",
        "comment_like",
        "dislike",
        "follow",
        "group_members",
        "group_messages",
        "like",
        "mute",
        "post",
        "product",
        "rec",
        "report",
        "trace",
        "user",
    ];

    for table in &expected {
        assert!(
            tables.contains(&table.to_string()),
            "Missing table: {table}"
        );
    }
    assert_eq!(tables.len(), expected.len(), "Unexpected table count: {tables:?}");
}

#[test]
fn test_schema_idempotent() {
    let conn = db::open_and_init(":memory:").expect("failed to open in-memory DB");
    // Running create_all_tables again should not fail
    db::schema::create_all_tables(&conn).expect("second create_all_tables call should not fail");
}

#[test]
fn test_queries_check_agent_userid() {
    let conn = db::open_and_init(":memory:").unwrap();

    // No user yet
    let result = db::queries::check_agent_userid(&conn, 42).unwrap();
    assert!(result.is_none(), "Expected None for nonexistent agent");

    // Insert a user with agent_id = 42
    conn.execute(
        "INSERT INTO user (agent_id, user_name, name, bio, created_at) VALUES (?1, ?2, ?3, ?4, ?5)",
        params![42, "testuser", "Test User", "A bio", "2026-01-01T00:00:00"],
    )
    .unwrap();

    let result = db::queries::check_agent_userid(&conn, 42).unwrap();
    assert!(result.is_some(), "Expected Some for existing agent");
    assert_eq!(result.unwrap(), 1, "First inserted user should have user_id = 1");
}

#[test]
fn test_queries_record_trace() {
    let conn = db::open_and_init(":memory:").unwrap();

    // Insert a user first (foreign key target)
    conn.execute(
        "INSERT INTO user (agent_id, user_name, name, bio, created_at) VALUES (?1, ?2, ?3, ?4, ?5)",
        params![1, "agent1", "Agent One", "bio", "2026-01-01T00:00:00"],
    )
    .unwrap();

    db::queries::record_trace(&conn, 1, "login", "first login", "2026-01-01T00:00:00").unwrap();

    let rows = db::queries::fetch_table(&conn, "trace").unwrap();
    assert_eq!(rows.len(), 1, "Expected 1 trace row");
    assert_eq!(rows[0]["action"], "login");
    assert_eq!(rows[0]["info"], "first login");
    assert_eq!(rows[0]["user_id"], 1);
}

#[test]
fn test_queries_fetch_table() {
    let conn = db::open_and_init(":memory:").unwrap();

    // Insert a user
    conn.execute(
        "INSERT INTO user (agent_id, user_name, name, bio, created_at) VALUES (?1, ?2, ?3, ?4, ?5)",
        params![7, "fetchme", "Fetch Me", "a bio", "2026-01-01T00:00:00"],
    )
    .unwrap();

    let rows = db::queries::fetch_table(&conn, "user").unwrap();
    assert_eq!(rows.len(), 1, "Expected 1 user row");
    assert_eq!(rows[0]["user_name"], "fetchme");
    assert_eq!(rows[0]["agent_id"], 7);
}
