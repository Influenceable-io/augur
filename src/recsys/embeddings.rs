use ndarray::{Array1, Array2, Axis};

/// Cosine similarity between two vectors.
pub fn cosine_similarity(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    let dot = a.dot(b);
    let norm_a = a.dot(a).sqrt();
    let norm_b = b.dot(b).sqrt();
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

    let mut result = Array2::zeros(dot_products.raw_dim());
    for i in 0..result.nrows() {
        for j in 0..result.ncols() {
            let norm = a_norms[i] * b_norms[j];
            result[[i, j]] = if norm > 0.0 {
                dot_products[[i, j]] / norm
            } else {
                0.0
            };
        }
    }
    result
}

/// Simple hash-based text embedding as a fallback when ONNX Runtime is not available.
///
/// Each text gets a vector based on word frequency hashing.
/// This mimics the structure of real sentence embeddings (normalized vectors
/// in a fixed-dimension space) but uses a simple hash function instead of
/// a neural network. The cosine similarity between these vectors
/// correlates with word overlap, which is a reasonable approximation for
/// content-based filtering.
///
/// When ONNX Runtime is available, this should be replaced with real
/// SentenceTransformer or TWHiN-BERT embeddings.
pub fn simple_text_embeddings(texts: &[String], dim: usize) -> Array2<f32> {
    let mut result: Array2<f32> = Array2::zeros((texts.len(), dim));
    for (i, text) in texts.iter().enumerate() {
        let words: Vec<&str> = text.split_whitespace().collect();
        for word in &words {
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
