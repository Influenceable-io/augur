use std::cmp::Ordering;
use std::collections::HashMap;

use super::embeddings::{cosine_similarity, embed_users_and_posts, EmbeddingModelType};
use super::{PostRow, TraceRow, UserRow, TRACE_LIKE_POST_PREFIX};

/// Maximum posts to consider before fine-grained scoring (matches OASIS coarse filter).
const COARSE_FILTER_LIMIT: usize = 4000;

/// TWHiN-BERT personalized recommendations with multi-signal scoring.
///
/// Matches OASIS's rec_sys_personalized_twh:
/// 1. Coarse filter: take most recent 4000 posts
/// 2. Time decay: `log((271.8 - elapsed_steps) / 100)`
/// 3. Profile similarity: cosine similarity between user profile and post
/// 4. Like similarity: avg cosine similarity with user's last 5 liked posts
/// 5. Final score: cosine_similarity * date_score + like_score
pub fn recommend(
    user_table: &[UserRow],
    post_table: &[PostRow],
    trace_table: &[TraceRow],
    max_rec_post_len: usize,
) -> Vec<(i64, i64)> {
    if user_table.is_empty() || post_table.is_empty() {
        return Vec::new();
    }

    // Coarse filter: keep only the most recent COARSE_FILTER_LIMIT posts (matches OASIS)
    let post_table = if post_table.len() > COARSE_FILTER_LIMIT {
        &post_table[post_table.len() - COARSE_FILTER_LIMIT..]
    } else {
        post_table
    };

    let (user_embeddings, post_embeddings) = embed_users_and_posts(user_table, post_table, EmbeddingModelType::TwhinBert);

    // Build post_id → index lookup for O(1) resolution in trace scanning
    let post_idx_map: HashMap<i64, usize> = post_table
        .iter()
        .enumerate()
        .map(|(i, p)| (p.post_id, i))
        .collect();

    // Build liked-post index per user from trace table
    let mut user_liked_posts: HashMap<i64, Vec<usize>> = HashMap::new();
    for trace in trace_table {
        if trace.action == "like_post"
            && let Some(pid_str) = trace.info.strip_prefix(TRACE_LIKE_POST_PREFIX)
            && let Ok(pid) = pid_str.trim().parse::<i64>()
            && let Some(&post_idx) = post_idx_map.get(&pid)
        {
            user_liked_posts
                .entry(trace.user_id)
                .or_default()
                .push(post_idx);
        }
    }

    let mut recommendations = Vec::new();

    for (user_idx, user) in user_table.iter().enumerate() {
        let user_emb = user_embeddings.row(user_idx);

        let mut scored: Vec<(usize, f64)> = Vec::new();

        for (post_idx, post) in post_table.iter().enumerate() {
            if post.user_id == user.user_id {
                continue;
            }

            let post_emb = post_embeddings.row(post_idx);

            // 1. Content similarity
            let content_sim = cosine_similarity(user_emb, post_emb) as f64;

            // 2. Time decay: log((271.8 - elapsed) / 100)
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
                            cosine_similarity(post_emb, post_embeddings.row(li)) as f64
                        })
                        .sum::<f64>()
                        / recent_liked.len() as f64
                }
            } else {
                0.0
            };

            // 4. Combined score
            let final_score = content_sim * date_score + like_score;
            scored.push((post_idx, final_score));
        }

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        for (post_idx, _) in scored.iter().take(max_rec_post_len) {
            recommendations.push((user.user_id, post_table[*post_idx].post_id));
        }
    }

    recommendations
}
