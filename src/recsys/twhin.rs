use super::{PostRow, TraceRow, UserRow};

/// TWHiN-BERT personalized recommendations with multi-signal scoring.
///
/// In OASIS, this uses TWHiN-BERT (Twitter/twhin-bert-base) or OpenAI embeddings.
///
/// Algorithm:
/// 1. Time decay: `log((271.8 - (current_time - created_at)) / 100)`
/// 2. Profile similarity: cosine similarity between user profile and post content
/// 3. Like similarity (optional): average cosine similarity with user's last 5 liked posts
/// 4. Final score: cosine_similarity * date_score + like_score
pub fn recommend(
    user_table: &[UserRow],
    post_table: &[PostRow],
    _trace_table: &[TraceRow],
    max_rec_post_len: usize,
) -> Vec<(i64, i64)> {
    // TODO: Implement TWHiN-BERT embeddings via ONNX Runtime
    // For now, fall back to random recommendation as a placeholder
    // This will be replaced with the full embedding pipeline

    tracing::warn!("TWHiN recsys: ONNX embeddings not yet implemented, falling back to random");
    super::random::recommend(
        user_table,
        post_table,
        max_rec_post_len,
    )
}
