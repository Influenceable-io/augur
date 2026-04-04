use std::path::Path;
use std::sync::Mutex;

use anyhow::{Context, Result};
use ndarray::{Array2, Axis};

const MAX_SEQ_LEN: usize = 512;
const ENCODE_BATCH_SIZE: usize = 64;

/// ONNX-based text embedder using SentenceTransformer-compatible models.
///
/// Supports models exported to ONNX format with input_ids + attention_mask inputs.
/// For sentence-transformers models (2 outputs), uses the sentence_embedding output.
/// For generic BERT models (1 output), applies mean pooling with attention mask.
pub struct OnnxEmbedder {
    session: Mutex<ort::session::Session>,
    tokenizer: Mutex<tokenizers::Tokenizer>,
    num_outputs: usize,
}

impl OnnxEmbedder {
    /// Load from a HuggingFace Hub model repository.
    ///
    /// Downloads the ONNX model and tokenizer from HuggingFace, caching in
    /// `~/.cache/huggingface/hub/`. Matches OASIS's SentenceTransformer auto-download.
    ///
    /// Checks ONNX Runtime availability FIRST to avoid downloading model files
    /// when the runtime isn't installed (common with `load-dynamic` feature).
    pub fn from_pretrained(repo_id: &str, onnx_file: &str) -> Result<Self> {
        // Canary check: verify ONNX Runtime is loadable before downloading model files.
        // With `load-dynamic`, this fails fast if the shared library isn't installed.
        let _ = ort::session::Session::builder()
            .map_err(|e| anyhow::anyhow!("ONNX Runtime not available: {e}"))?;

        tracing::info!(repo_id, onnx_file, "Downloading ONNX model from HuggingFace");

        let api = hf_hub::api::sync::Api::new()
            .context("Failed to initialize HuggingFace Hub API")?;
        let repo = api.model(repo_id.to_string());

        let model_path = repo
            .get(onnx_file)
            .with_context(|| format!("Failed to download {repo_id}/{onnx_file}"))?;
        let tokenizer_path = repo
            .get("tokenizer.json")
            .with_context(|| format!("Failed to download {repo_id}/tokenizer.json"))?;

        Self::from_files(&model_path, &tokenizer_path)
    }

    /// Load from local file paths.
    pub fn from_files(model_path: &Path, tokenizer_path: &Path) -> Result<Self> {
        use ort::session::{Session, builder::GraphOptimizationLevel};

        // ort::Error<SessionBuilder> doesn't implement Send+Sync, so we convert manually
        let session = Session::builder()
            .map_err(|e| anyhow::anyhow!("ort session builder: {e}"))?
            .with_optimization_level(GraphOptimizationLevel::Level1)
            .map_err(|e| anyhow::anyhow!("ort optimization level: {e}"))?
            .with_intra_threads(4)
            .map_err(|e| anyhow::anyhow!("ort intra threads: {e}"))?
            .commit_from_file(model_path)
            .map_err(|e| anyhow::anyhow!("Failed to load ONNX model from {}: {e}", model_path.display()))?;

        let num_outputs = session.outputs().len();

        let mut tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;

        tokenizer.with_padding(Some(tokenizers::PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            ..tokenizers::PaddingParams::default()
        }));

        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: MAX_SEQ_LEN,
                ..tokenizers::TruncationParams::default()
            }))
            .map_err(|e| anyhow::anyhow!("Failed to set truncation: {e}"))?;

        tracing::info!(
            model = model_path.display().to_string(),
            num_outputs,
            "ONNX embedder loaded"
        );

        Ok(Self {
            session: Mutex::new(session),
            tokenizer: Mutex::new(tokenizer),
            num_outputs,
        })
    }

    /// Encode texts into L2-normalized embeddings.
    ///
    /// Processes in batches of ENCODE_BATCH_SIZE to limit memory usage.
    /// Returns Array2<f32> of shape [texts.len(), embedding_dim].
    pub fn encode(&self, texts: &[String]) -> Result<Array2<f32>> {
        if texts.is_empty() {
            return Ok(Array2::zeros((0, 0)));
        }

        let mut all_rows: Vec<Vec<f32>> = Vec::with_capacity(texts.len());

        for chunk in texts.chunks(ENCODE_BATCH_SIZE) {
            let batch_embeddings = self.encode_batch(chunk)?;
            all_rows.extend(batch_embeddings);
        }

        let dim = all_rows.first().map(|r| r.len()).unwrap_or(1);
        let flat: Vec<f32> = all_rows.into_iter().flatten().collect();
        let mut result = Array2::from_shape_vec((texts.len(), dim), flat)?;

        // L2 normalize
        for mut row in result.axis_iter_mut(Axis(0)) {
            let norm = row.dot(&row).sqrt();
            if norm > 0.0 {
                row /= norm;
            }
        }

        Ok(result)
    }

    /// Encode a single batch of texts (up to ENCODE_BATCH_SIZE).
    fn encode_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let str_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let batch_size = texts.len();

        // Tokenize
        let encodings = {
            let tokenizer = self.tokenizer.lock().unwrap();
            tokenizer
                .encode_batch(str_refs, true)
                .map_err(|e| anyhow::anyhow!("Tokenization failed: {e}"))?
        };

        let seq_len = encodings[0].len();

        let ids: Vec<i64> = encodings
            .iter()
            .flat_map(|e| e.get_ids().iter().map(|&i| i as i64))
            .collect();
        let mask: Vec<i64> = encodings
            .iter()
            .flat_map(|e| e.get_attention_mask().iter().map(|&i| i as i64))
            .collect();

        // Create input tensors
        let ids_tensor =
            ort::value::TensorRef::from_array_view(([batch_size, seq_len], &*ids))?;
        let mask_tensor =
            ort::value::TensorRef::from_array_view(([batch_size, seq_len], &*mask))?;

        // Run inference
        let mut session = self.session.lock().unwrap();
        let outputs = session.run(ort::inputs! {
            "input_ids" => ids_tensor,
            "attention_mask" => mask_tensor,
        })?;

        // Extract embeddings based on model type
        if self.num_outputs > 1 {
            // Sentence-transformers model: output 1 is sentence_embedding [batch, dim]
            self.extract_sentence_embedding(&outputs)
        } else {
            // Generic BERT: output 0 is last_hidden_state [batch, seq, dim]
            self.extract_with_mean_pooling(&outputs, &mask, batch_size, seq_len)
        }
    }

    /// Extract pre-pooled sentence embeddings (sentence-transformers ONNX models).
    fn extract_sentence_embedding(
        &self,
        outputs: &ort::session::SessionOutputs<'_>,
    ) -> Result<Vec<Vec<f32>>> {
        let output = &outputs[1];
        let tensor = output.try_extract_tensor::<f32>()?;
        let shape = tensor.0;
        let data = tensor.1;

        let dim = shape[1] as usize;
        let rows: Vec<Vec<f32>> = data.chunks(dim).map(|c: &[f32]| c.to_vec()).collect();

        Ok(rows)
    }

    /// Extract token-level embeddings and apply mean pooling with attention mask.
    fn extract_with_mean_pooling(
        &self,
        outputs: &ort::session::SessionOutputs<'_>,
        mask: &[i64],
        batch_size: usize,
        seq_len: usize,
    ) -> Result<Vec<Vec<f32>>> {
        let output = &outputs[0];
        let tensor = output.try_extract_tensor::<f32>()?;
        let data = tensor.1;
        let dim = tensor.0[2] as usize;

        let mut rows = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let mut emb = vec![0.0f32; dim];
            let mut count = 0.0f32;
            for j in 0..seq_len {
                if mask[i * seq_len + j] == 1 {
                    let offset = i * seq_len * dim + j * dim;
                    for k in 0..dim {
                        emb[k] += data[offset + k];
                    }
                    count += 1.0;
                }
            }
            if count > 0.0 {
                for v in &mut emb {
                    *v /= count;
                }
            }
            rows.push(emb);
        }

        Ok(rows)
    }
}
