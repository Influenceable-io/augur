use std::cmp::Ordering;

use super::PostRow;

/// Reddit hot-score algorithm.
///
/// Formula: `sign * log(max(|s|, 1), 10) + seconds / 45000`
/// where `s = num_likes - num_dislikes`
pub fn recommend(
    post_table: &[PostRow],
    max_rec_post_len: usize,
) -> Vec<(i64, i64)> {
    // Score all posts
    let mut scored: Vec<(i64, f64)> = post_table
        .iter()
        .map(|post| {
            let score = calculate_hot_score(post.num_likes, post.num_dislikes, &post.created_at);
            (post.post_id, score)
        })
        .collect();

    // Sort by score descending, take top-k
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    let top_k: Vec<i64> = scored.iter().take(max_rec_post_len).map(|(id, _)| *id).collect();

    // Same recommendations for all users (Reddit doesn't personalize)
    // Return empty user_id=0 as placeholder; caller maps to all users
    top_k.iter().map(|&post_id| (0_i64, post_id)).collect()
}

/// Calculate Reddit's hot score.
///
/// Matches OASIS's implementation exactly:
/// ```
/// s = num_likes - num_dislikes
/// order = log(max(abs(s), 1), 10)
/// sign = 1 if s > 0, -1 if s < 0, 0 if s == 0
/// seconds = epoch_seconds - 1134028003
/// hot_score = sign * order + seconds / 45000
/// ```
fn calculate_hot_score(num_likes: i64, num_dislikes: i64, created_at: &str) -> f64 {
    let s = num_likes - num_dislikes;

    let order = (s.unsigned_abs().max(1) as f64).log10();

    let sign = if s > 0 {
        1.0
    } else if s < 0 {
        -1.0
    } else {
        0.0
    };

    // Parse created_at to epoch seconds
    let epoch_seconds = chrono::NaiveDateTime::parse_from_str(created_at, "%Y-%m-%d %H:%M:%S")
        .map(|dt| dt.and_utc().timestamp() as f64)
        .unwrap_or(0.0);

    let seconds = epoch_seconds - 1_134_028_003.0;

    let score = sign * order + seconds / 45000.0;
    (score * 10_000_000.0).round() / 10_000_000.0 // Round to 7 decimal places
}
