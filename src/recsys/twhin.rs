use std::cmp::Ordering;
use std::collections::HashMap;

use super::embeddings::{cosine_similarity, simple_text_embeddings};
use super::{PostRow, TraceRow, UserRow};

/// TWHiN-BERT personalized recommendations with multi-signal scoring.
///
/// Matches OASIS's rec_sys_personalized_twh:
/// 1. Time decay: `log((271.8 - elapsed_steps) / 100)`
/// 2. Profile similarity: cosine similarity between user profile and post
/// 3. Like similarity: avg cosine similarity with user's last 5 liked posts
/// 4. Final score: cosine_similarity * date_score + like_score
pub fn recommend(
    user_table: &[UserRow],
    post_table: &[PostRow],
    trace_table: &[TraceRow],
    max_rec_post_len: usize,
) -> Vec<(i64, i64)> {
    if user_table.is_empty() || post_table.is_empty() {
        return Vec::new();
    }

    let embedding_dim = 128;

    // Encode user bios
    let user_texts: Vec<String> = user_table.iter().map(|u| u.bio.clone()).collect();
    let user_embeddings = simple_text_embeddings(&user_texts, embedding_dim);

    // Encode post contents
    let post_texts: Vec<String> = post_table.iter().map(|p| p.content.clone()).collect();
    let post_embeddings = simple_text_embeddings(&post_texts, embedding_dim);

    // Build liked-post index per user from trace table
    let mut user_liked_posts: HashMap<i64, Vec<usize>> = HashMap::new();
    for trace in trace_table {
        if trace.action == "like_post"
            && let Some(pid_str) = trace.info.strip_prefix("post_id: ")
                && let Ok(pid) = pid_str.trim().parse::<i64>()
                    && let Some(post_idx) = post_table.iter().position(|p| p.post_id == pid) {
                        user_liked_posts
                            .entry(trace.user_id)
                            .or_default()
                            .push(post_idx);
                    }
    }

    let mut recommendations = Vec::new();

    for (user_idx, user) in user_table.iter().enumerate() {
        let user_emb = user_embeddings.row(user_idx).to_owned();

        let mut scored: Vec<(usize, f64)> = Vec::new();

        for (post_idx, post) in post_table.iter().enumerate() {
            // Skip own posts
            if post.user_id == user.user_id {
                continue;
            }

            let post_emb = post_embeddings.row(post_idx).to_owned();

            // 1. Content similarity
            let content_sim = cosine_similarity(&user_emb, &post_emb) as f64;

            // 2. Time decay score
            // In OASIS: date_score = log((271.8 - elapsed) / 100)
            // Use post index as a proxy for recency (higher index = newer)
            let recency = post_table.len() as f64 - post_idx as f64;
            let date_score = ((271.8 - recency.min(270.0)) / 100.0).max(0.01).ln();

            // 3. Like similarity (average cosine sim with last 5 liked posts)
            let like_score = if let Some(liked) = user_liked_posts.get(&user.user_id) {
                let recent_liked: Vec<_> = liked.iter().rev().take(5).collect();
                if recent_liked.is_empty() {
                    0.0
                } else {
                    recent_liked
                        .iter()
                        .map(|&&li| {
                            let liked_emb = post_embeddings.row(li).to_owned();
                            cosine_similarity(&post_emb, &liked_emb) as f64
                        })
                        .sum::<f64>()
                        / recent_liked.len() as f64
                }
            } else {
                0.0
            };

            // 4. Combined score: content_sim * date_score + like_score
            let final_score = content_sim * date_score + like_score;
            scored.push((post_idx, final_score));
        }

        // Sort by score descending
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        for (post_idx, _) in scored.iter().take(max_rec_post_len) {
            recommendations.push((user.user_id, post_table[*post_idx].post_id));
        }
    }

    recommendations
}
