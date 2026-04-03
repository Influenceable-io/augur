use std::cmp::Ordering;

use super::embeddings::{cosine_similarity_matrix, simple_text_embeddings};
use super::{PostRow, TraceRow, UserRow};

/// Twitter-style personalized recommendations using cosine similarity.
///
/// Matches OASIS's rec_sys_personalized:
/// 1. Encode user bios as embeddings
/// 2. Encode post contents as embeddings
/// 3. Compute cosine similarity matrix
/// 4. Filter out user's own posts
/// 5. Return top-k for each user
pub fn recommend(
    user_table: &[UserRow],
    post_table: &[PostRow],
    _trace_table: &[TraceRow],
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

    // Compute similarity matrix: users x posts
    let sim_matrix = cosine_similarity_matrix(&user_embeddings, &post_embeddings);

    // For each user, get top-k posts (excluding own posts)
    let mut recommendations = Vec::new();
    for (user_idx, user) in user_table.iter().enumerate() {
        let mut scored: Vec<(usize, f32)> = (0..post_table.len())
            .map(|post_idx| (post_idx, sim_matrix[[user_idx, post_idx]]))
            .filter(|(post_idx, _)| post_table[*post_idx].user_id != user.user_id)
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        for (post_idx, _) in scored.iter().take(max_rec_post_len) {
            recommendations.push((user.user_id, post_table[*post_idx].post_id));
        }
    }

    recommendations
}
