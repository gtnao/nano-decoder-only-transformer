use crate::embedding::Embedding;
use crate::layer_norm::LayerNorm;
use crate::linear::Linear;
use crate::positional_encoding::positional_encoding;
use crate::tensor::Tensor;
use crate::transformer_block::TransformerBlock;

/// Decoder-Only Transformer
/// token_ids -> Embedding + PosEnc -> N x Block -> LayerNorm -> Linear -> logits
pub struct Transformer {
    pub token_embedding: Embedding,
    pub blocks: Vec<TransformerBlock>,
    pub ln_final: LayerNorm,
    pub lm_head: Linear, // [d_model, vocab_size]
}

impl Transformer {
    pub fn new(
        token_embedding: Embedding,
        blocks: Vec<TransformerBlock>,
        ln_final: LayerNorm,
        lm_head: Linear,
    ) -> Self {
        Self {
            token_embedding,
            blocks,
            ln_final,
            lm_head,
        }
    }

    pub fn rand(
        vocab_size: usize,
        d_model: usize,
        n_heads: usize,
        d_ff: usize,
        n_layers: usize,
    ) -> Self {
        let token_embedding = Embedding::rand(vocab_size, d_model);
        let blocks = (0..n_layers)
            .map(|_| TransformerBlock::rand(d_model, n_heads, d_ff))
            .collect();
        let ln_final = LayerNorm::new(d_model);
        let lm_head = Linear::rand(d_model, vocab_size);
        Self {
            token_embedding,
            blocks,
            ln_final,
            lm_head,
        }
    }

    /// token_ids: &[usize] -> logits: Tensor [seq_len, vocab_size]
    pub fn forward(&self, token_ids: &[usize]) -> Tensor {
        let seq_len = token_ids.len();
        let d_model = self.token_embedding.weight.shape[1];

        // Token embedding + positional encoding
        let tok_emb = self.token_embedding.forward(token_ids);
        let pos_enc = positional_encoding(seq_len, d_model);
        let mut x = tok_emb.add(&pos_enc);

        // N transformer blocks
        for block in &self.blocks {
            x = block.forward(&x);
        }

        // Final layer norm + projection to vocab
        let x = self.ln_final.forward(&x);
        self.lm_head.forward(&x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tiny_model() -> Transformer {
        // vocab=10, d_model=8, n_heads=2, d_ff=16, n_layers=2
        Transformer::rand(10, 8, 2, 16, 2)
    }

    #[test]
    fn test_forward_shape() {
        let model = make_tiny_model();
        let logits = model.forward(&[0, 1, 2]);
        assert_eq!(logits.shape, vec![3, 10]); // [seq_len, vocab_size]
    }

    #[test]
    fn test_forward_single_token() {
        let model = make_tiny_model();
        let logits = model.forward(&[5]);
        assert_eq!(logits.shape, vec![1, 10]);
    }

    #[test]
    fn test_forward_deterministic() {
        let model = make_tiny_model();
        let out1 = model.forward(&[0, 1, 2]);
        let out2 = model.forward(&[0, 1, 2]);
        assert_eq!(out1.data, out2.data);
    }

    #[test]
    fn test_forward_different_inputs_differ() {
        let model = make_tiny_model();
        let out1 = model.forward(&[0, 1, 2]);
        let out2 = model.forward(&[3, 4, 5]);
        assert_ne!(out1.data, out2.data);
    }

    #[test]
    fn test_forward_logits_are_finite() {
        let model = make_tiny_model();
        let logits = model.forward(&[0, 1, 2, 3, 4]);
        assert!(logits.data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_rand_structure() {
        let model = make_tiny_model();
        assert_eq!(model.token_embedding.weight.shape, vec![10, 8]);
        assert_eq!(model.blocks.len(), 2);
        assert_eq!(model.ln_final.gamma.shape, vec![8]);
        assert_eq!(model.lm_head.weight.shape, vec![10, 8]);
    }
}
