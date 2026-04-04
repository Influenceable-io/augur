use std::sync::OnceLock;

use ndarray::{Array2, ArrayView1, Axis};

use super::onnx_embedder::OnnxEmbedder;
use super::{PostRow, UserRow};

/// Default embedding dimension for hash-based fallback embeddings.
const HASH_EMBEDDING_DIM: usize = 128;

/// Default HuggingFace model for Twitter-style recsys (matches OASIS's all-MiniLM-L6-v2).
const MINILM_REPO: &str = "sentence-transformers/all-MiniLM-L6-v2";
const MINILM_ONNX: &str = "onnx/model.onnx";

/// Default HuggingFace model for TWHiN recsys.
/// TWHiN-BERT doesn't have an ONNX export on HuggingFace, so we fall back to MiniLM.
/// Users can override via AUGUR_TWHIN_MODEL_REPO and AUGUR_TWHIN_ONNX_FILE env vars,
/// or export TWHiN-BERT to ONNX with: `optimum-cli export onnx --model Twitter/twhin-bert-base ./twhin-onnx/`
const TWHIN_REPO_DEFAULT: &str = MINILM_REPO;
const TWHIN_ONNX_DEFAULT: &str = MINILM_ONNX;

/// Which embedding model to use.
#[derive(Debug, Clone, Copy)]
pub enum EmbeddingModelType {
    /// all-MiniLM-L6-v2 (384-dim) — used by Twitter-style personalized recsys.
    MiniLM,
    /// TWHiN-BERT (768-dim) — used by TWHiN personalized recsys.
    /// Falls back to MiniLM if TWHiN-BERT ONNX is not available.
    TwhinBert,
}

static MINILM_EMBEDDER: OnceLock<Option<OnnxEmbedder>> = OnceLock::new();
static TWHIN_EMBEDDER: OnceLock<Option<OnnxEmbedder>> = OnceLock::new();

/// Check if ONNX embeddings are enabled via AUGUR_ONNX_EMBEDDINGS env var.
///
/// Set `AUGUR_ONNX_EMBEDDINGS=1` to enable real SentenceTransformer embeddings.
/// Models are auto-downloaded from HuggingFace on first use (~90MB).
/// Without this env var, fast hash-based embeddings are used (no download required).
fn onnx_enabled() -> bool {
    std::env::var("AUGUR_ONNX_EMBEDDINGS")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

fn init_embedder(
    repo_env: &str,
    onnx_env: &str,
    default_repo: &str,
    default_onnx: &str,
    label: &str,
) -> Option<OnnxEmbedder> {
    if !onnx_enabled() {
        return None;
    }

    let repo = std::env::var(repo_env).unwrap_or_else(|_| default_repo.to_string());
    let onnx = std::env::var(onnx_env).unwrap_or_else(|_| default_onnx.to_string());

    match OnnxEmbedder::from_pretrained(&repo, &onnx) {
        Ok(e) => {
            tracing::info!("{label} ONNX embedder loaded successfully");
            Some(e)
        }
        Err(e) => {
            tracing::warn!("{label} ONNX embedder unavailable, using hash fallback: {e}");
            None
        }
    }
}

fn get_minilm_embedder() -> &'static Option<OnnxEmbedder> {
    MINILM_EMBEDDER.get_or_init(|| {
        init_embedder("AUGUR_MINILM_MODEL_REPO", "AUGUR_MINILM_ONNX_FILE", MINILM_REPO, MINILM_ONNX, "MiniLM")
    })
}

fn get_twhin_embedder() -> &'static Option<OnnxEmbedder> {
    TWHIN_EMBEDDER.get_or_init(|| {
        init_embedder("AUGUR_TWHIN_MODEL_REPO", "AUGUR_TWHIN_ONNX_FILE", TWHIN_REPO_DEFAULT, TWHIN_ONNX_DEFAULT, "TWHiN")
    })
}

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
/// Tries ONNX-based embeddings first (SentenceTransformer / TWHiN-BERT),
/// falling back to hash-based embeddings if ONNX Runtime is unavailable.
pub fn embed_users_and_posts(
    user_table: &[UserRow],
    post_table: &[PostRow],
    model_type: EmbeddingModelType,
) -> (Array2<f32>, Array2<f32>) {
    let user_texts: Vec<String> = user_table.iter().map(|u| u.bio.clone()).collect();
    let post_texts: Vec<String> = post_table.iter().map(|p| p.content.clone()).collect();

    let embedder = match model_type {
        EmbeddingModelType::MiniLM => get_minilm_embedder(),
        EmbeddingModelType::TwhinBert => get_twhin_embedder(),
    };

    if let Some(onnx) = embedder {
        match (onnx.encode(&user_texts), onnx.encode(&post_texts)) {
            (Ok(user_emb), Ok(post_emb)) => return (user_emb, post_emb),
            (Err(e), _) | (_, Err(e)) => {
                tracing::warn!("ONNX encoding failed, falling back to hash: {e}");
            }
        }
    }

    // Fallback: hash-based embeddings
    let user_embeddings = simple_text_embeddings(&user_texts, HASH_EMBEDDING_DIM);
    let post_embeddings = simple_text_embeddings(&post_texts, HASH_EMBEDDING_DIM);
    (user_embeddings, post_embeddings)
}

/// Simple hash-based text embedding as a fallback when ONNX Runtime is not available.
///
/// Projects each word into a fixed-dimension vector via hashing, then L2-normalizes.
/// Not semantically meaningful — used only when real transformer embeddings are unavailable.
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
