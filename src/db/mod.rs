pub mod schema;
pub mod queries;

use anyhow::Result;
use rusqlite::Connection;

/// Create a new database connection and initialize the schema.
pub fn open_and_init(path: &str) -> Result<Connection> {
    let conn = if path == ":memory:" {
        Connection::open_in_memory()?
    } else {
        Connection::open(path)?
    };

    // Enable WAL mode for concurrent reads
    conn.execute_batch("PRAGMA journal_mode=WAL;")?;

    schema::create_all_tables(&conn)?;

    Ok(conn)
}
