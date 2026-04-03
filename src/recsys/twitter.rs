use super::{PostRow, TraceRow, UserRow};

/// Twitter-style personalized recommendations using cosine similarity.
///
/// In OASIS, this uses SentenceTransformer (paraphrase-MiniLM-L6-v2) for embeddings.
/// In augur, we use ONNX Runtime to load the same model.
///
/// Algorithm:
/// 1. Encode user bios with sentence transformer
/// 2. Encode post contents
/// 3. Compute cosine similarity: dot_product / (user_norms * post_norms)
/// 4. Filter out user's own posts
/// 5. Return top-k for each user
pub fn recommend(
    user_table: &[UserRow],
    post_table: &[PostRow],
    _trace_table: &[TraceRow],
    max_rec_post_len: usize,
) -> Vec<(i64, i64)> {
    // TODO: Implement ONNX-based sentence transformer embeddings
    // For now, fall back to random recommendation as a placeholder
    // This will be replaced with the full embedding pipeline

    tracing::warn!("Twitter recsys: ONNX embeddings not yet implemented, falling back to random");
    super::random::recommend(
        user_table,
        post_table,
        max_rec_post_len,
    )
}
