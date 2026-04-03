use ndarray::{Array1, Array2, ArrayView1, Axis};

use super::{PostRow, UserRow};

/// Default embedding dimension for hash-based embeddings.
pub const EMBEDDING_DIM: usize = 128;

/// Cosine similarity between two vectors.
pub fn cosine_similarity(a: ArrayView1<f32>, b: ArrayView1<f32>) -> f32 {
    let dot = a.dot(&b);
    let norm_a = a.dot(&a).sqrt();
    let norm_b = b.dot(&b).sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// Cosine similarity matrix: each row of `a` against each row of `b`.
/// Returns matrix of shape (a.nrows(), b.nrows()).
pub fn cosine_similarity_matrix(a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
    let a_norms = a.map_axis(Axis(1), |row| row.dot(&row).sqrt());
    let b_norms = b.map_axis(Axis(1), |row| row.dot(&row).sqrt());

    let dot_products = a.dot(&b.t());

    // Broadcast: outer product of norms gives the (U, P) denominator matrix
    let norms = &a_norms.insert_axis(Axis(1)) * &b_norms.insert_axis(Axis(0));
    let mut result = dot_products / norms;
    result.mapv_inplace(|v| if v.is_nan() { 0.0 } else { v });
    result
}

/// Embed user bios and post contents into the same vector space.
///
/// Shared prologue used by both Twitter and TWHiN recsys.
pub fn embed_users_and_posts(
    user_table: &[UserRow],
    post_table: &[PostRow],
) -> (Array2<f32>, Array2<f32>) {
    let user_texts: Vec<String> = user_table.iter().map(|u| u.bio.clone()).collect();
    let user_embeddings = simple_text_embeddings(&user_texts, EMBEDDING_DIM);

    let post_texts: Vec<String> = post_table.iter().map(|p| p.content.clone()).collect();
    let post_embeddings = simple_text_embeddings(&post_texts, EMBEDDING_DIM);

    (user_embeddings, post_embeddings)
}

/// Simple hash-based text embedding as a fallback when ONNX Runtime is not available.
///
/// When ONNX Runtime is available, this should be replaced with real
/// SentenceTransformer or TWHiN-BERT embeddings.
pub fn simple_text_embeddings(texts: &[String], dim: usize) -> Array2<f32> {
    let mut result: Array2<f32> = Array2::zeros((texts.len(), dim));
    for (i, text) in texts.iter().enumerate() {
        for word in text.split_whitespace() {
            let lower = word.to_lowercase();
            let hash = lower
                .bytes()
                .fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
            let idx = (hash % dim as u64) as usize;
            result[[i, idx]] += 1.0;
        }
        // L2 normalize
        let norm = result.row(i).dot(&result.row(i)).sqrt();
        if norm > 0.0 {
            for j in 0..dim {
                result[[i, j]] /= norm;
            }
        }
    }
    result
}
