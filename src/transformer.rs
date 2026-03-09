use crate::embedding::Embedding;
use crate::layer_norm::LayerNorm;
use crate::linear::Linear;
use crate::optimizer::{Adam, clip_grad_norm};
use crate::positional_encoding::positional_encoding;
use crate::tensor::Tensor;
use crate::transformer_block::{TransformerBlock, TransformerBlockGradients};

pub struct TransformerGradients {
    pub d_embedding_weight: Tensor,
    pub block_grads: Vec<TransformerBlockGradients>,
    pub d_ln_final_gamma: Tensor,
    pub d_ln_final_beta: Tensor,
    pub d_lm_head_weight: Tensor,
    pub d_lm_head_bias: Tensor,
}

impl TransformerGradients {
    /// Clip all gradients by global L2 norm. Returns original norm.
    pub fn clip_norm(&mut self, max_norm: f32) -> f32 {
        let mut slices: Vec<&mut [f32]> = Vec::new();

        slices.push(&mut self.d_embedding_weight.data);

        for bg in &mut self.block_grads {
            slices.push(&mut bg.d_ln1_gamma.data);
            slices.push(&mut bg.d_ln1_beta.data);
            slices.push(&mut bg.mha_grads.d_wq_weight.data);
            slices.push(&mut bg.mha_grads.d_wq_bias.data);
            slices.push(&mut bg.mha_grads.d_wk_weight.data);
            slices.push(&mut bg.mha_grads.d_wk_bias.data);
            slices.push(&mut bg.mha_grads.d_wv_weight.data);
            slices.push(&mut bg.mha_grads.d_wv_bias.data);
            slices.push(&mut bg.mha_grads.d_wo_weight.data);
            slices.push(&mut bg.mha_grads.d_wo_bias.data);
            slices.push(&mut bg.d_ln2_gamma.data);
            slices.push(&mut bg.d_ln2_beta.data);
            slices.push(&mut bg.ffn_grads.d_l1_weight.data);
            slices.push(&mut bg.ffn_grads.d_l1_bias.data);
            slices.push(&mut bg.ffn_grads.d_l2_weight.data);
            slices.push(&mut bg.ffn_grads.d_l2_bias.data);
        }

        slices.push(&mut self.d_ln_final_gamma.data);
        slices.push(&mut self.d_ln_final_beta.data);
        slices.push(&mut self.d_lm_head_weight.data);
        slices.push(&mut self.d_lm_head_bias.data);

        clip_grad_norm(&mut slices, max_norm)
    }
}

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
        Self::rand_with_dropout(vocab_size, d_model, n_heads, d_ff, n_layers, 0.0)
    }

    pub fn rand_with_dropout(
        vocab_size: usize,
        d_model: usize,
        n_heads: usize,
        d_ff: usize,
        n_layers: usize,
        dropout_rate: f32,
    ) -> Self {
        let token_embedding = Embedding::rand(vocab_size, d_model);
        let blocks = (0..n_layers)
            .map(|_| TransformerBlock::rand_with_dropout(d_model, n_heads, d_ff, dropout_rate))
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
        self.forward_with_training(token_ids, false)
    }

    /// Forward with explicit training flag (enables dropout).
    pub fn forward_with_training(&self, token_ids: &[usize], training: bool) -> Tensor {
        let seq_len = token_ids.len();
        let d_model = self.token_embedding.weight.shape[1];

        // Token embedding + positional encoding
        let tok_emb = self.token_embedding.forward(token_ids);
        let pos_enc = positional_encoding(seq_len, d_model);
        let mut x = tok_emb.add(&pos_enc);

        // N transformer blocks
        for block in &self.blocks {
            x = block.forward(&x, training);
        }

        // Final layer norm + projection to vocab
        let x = self.ln_final.forward(&x);
        self.lm_head.forward(&x)
    }

    /// Backward pass for the full Transformer.
    /// d_logits: [seq_len, vocab_size] gradient from loss
    /// token_ids: cached from forward
    pub fn backward(&self, d_logits: &Tensor, token_ids: &[usize]) -> TransformerGradients {
        let seq_len = token_ids.len();
        let d_model = self.token_embedding.weight.shape[1];

        // Recompute forward to get intermediates
        let tok_emb = self.token_embedding.forward(token_ids);
        let pos_enc = positional_encoding(seq_len, d_model);
        let x0 = tok_emb.add(&pos_enc);

        // Store input to each block (recompute without dropout;
        // dropout masks are cached in blocks and applied in backward)
        let mut block_inputs = Vec::with_capacity(self.blocks.len());
        let mut x = x0.clone();
        for block in &self.blocks {
            block_inputs.push(x.clone());
            x = block.forward(&x, false);
        }
        let x_before_ln = x;
        let x_after_ln = self.ln_final.forward(&x_before_ln);

        // Backward through lm_head
        let (d_after_ln, d_lm_head_weight, d_lm_head_bias) =
            self.lm_head.backward(d_logits, &x_after_ln);

        // Backward through ln_final
        let (mut d_x, d_ln_final_gamma, d_ln_final_beta) =
            self.ln_final.backward(&d_after_ln, &x_before_ln);

        // Backward through blocks (reverse order)
        let mut block_grads = Vec::with_capacity(self.blocks.len());
        for (block, block_input) in self.blocks.iter().zip(block_inputs.iter()).rev() {
            let grads = block.backward(&d_x, block_input);
            d_x = grads.d_x.clone();
            block_grads.push(grads);
        }
        block_grads.reverse(); // restore original order

        // Backward through embedding (pos_enc is fixed, no gradient)
        let d_embedding_weight = self.token_embedding.backward(&d_x, token_ids);

        TransformerGradients {
            d_embedding_weight,
            block_grads,
            d_ln_final_gamma,
            d_ln_final_beta,
            d_lm_head_weight,
            d_lm_head_bias,
        }
    }

    /// Return sizes of all parameter groups (for Adam initialization).
    pub fn param_sizes(&self) -> Vec<usize> {
        let mut sizes = Vec::new();
        // Embedding
        sizes.push(self.token_embedding.weight.data.len());
        // Blocks
        for block in &self.blocks {
            // LN1: gamma, beta
            sizes.push(block.ln1.gamma.data.len());
            sizes.push(block.ln1.beta.data.len());
            // MHA: wq, wk, wv, wo (weight + bias each)
            sizes.push(block.mha.wq.weight.data.len());
            sizes.push(block.mha.wq.bias.data.len());
            sizes.push(block.mha.wk.weight.data.len());
            sizes.push(block.mha.wk.bias.data.len());
            sizes.push(block.mha.wv.weight.data.len());
            sizes.push(block.mha.wv.bias.data.len());
            sizes.push(block.mha.wo.weight.data.len());
            sizes.push(block.mha.wo.bias.data.len());
            // LN2: gamma, beta
            sizes.push(block.ln2.gamma.data.len());
            sizes.push(block.ln2.beta.data.len());
            // FFN: linear1, linear2 (weight + bias each)
            sizes.push(block.ffn.linear1.weight.data.len());
            sizes.push(block.ffn.linear1.bias.data.len());
            sizes.push(block.ffn.linear2.weight.data.len());
            sizes.push(block.ffn.linear2.bias.data.len());
        }
        // Final LN: gamma, beta
        sizes.push(self.ln_final.gamma.data.len());
        sizes.push(self.ln_final.beta.data.len());
        // LM Head: weight, bias
        sizes.push(self.lm_head.weight.data.len());
        sizes.push(self.lm_head.bias.data.len());
        sizes
    }

    /// Apply gradients using Adam optimizer.
    pub fn apply_gradients(&mut self, grads: &TransformerGradients, adam: &mut Adam) {
        adam.begin_step();
        let mut idx = 0;

        // Embedding
        adam.update(idx, &mut self.token_embedding.weight.data, &grads.d_embedding_weight.data);
        idx += 1;

        // Blocks
        for (block, bg) in self.blocks.iter_mut().zip(grads.block_grads.iter()) {
            // LN1
            adam.update(idx, &mut block.ln1.gamma.data, &bg.d_ln1_gamma.data); idx += 1;
            adam.update(idx, &mut block.ln1.beta.data, &bg.d_ln1_beta.data); idx += 1;
            // MHA
            adam.update(idx, &mut block.mha.wq.weight.data, &bg.mha_grads.d_wq_weight.data); idx += 1;
            adam.update(idx, &mut block.mha.wq.bias.data, &bg.mha_grads.d_wq_bias.data); idx += 1;
            adam.update(idx, &mut block.mha.wk.weight.data, &bg.mha_grads.d_wk_weight.data); idx += 1;
            adam.update(idx, &mut block.mha.wk.bias.data, &bg.mha_grads.d_wk_bias.data); idx += 1;
            adam.update(idx, &mut block.mha.wv.weight.data, &bg.mha_grads.d_wv_weight.data); idx += 1;
            adam.update(idx, &mut block.mha.wv.bias.data, &bg.mha_grads.d_wv_bias.data); idx += 1;
            adam.update(idx, &mut block.mha.wo.weight.data, &bg.mha_grads.d_wo_weight.data); idx += 1;
            adam.update(idx, &mut block.mha.wo.bias.data, &bg.mha_grads.d_wo_bias.data); idx += 1;
            // LN2
            adam.update(idx, &mut block.ln2.gamma.data, &bg.d_ln2_gamma.data); idx += 1;
            adam.update(idx, &mut block.ln2.beta.data, &bg.d_ln2_beta.data); idx += 1;
            // FFN
            adam.update(idx, &mut block.ffn.linear1.weight.data, &bg.ffn_grads.d_l1_weight.data); idx += 1;
            adam.update(idx, &mut block.ffn.linear1.bias.data, &bg.ffn_grads.d_l1_bias.data); idx += 1;
            adam.update(idx, &mut block.ffn.linear2.weight.data, &bg.ffn_grads.d_l2_weight.data); idx += 1;
            adam.update(idx, &mut block.ffn.linear2.bias.data, &bg.ffn_grads.d_l2_bias.data); idx += 1;
        }

        // Final LN
        adam.update(idx, &mut self.ln_final.gamma.data, &grads.d_ln_final_gamma.data); idx += 1;
        adam.update(idx, &mut self.ln_final.beta.data, &grads.d_ln_final_beta.data); idx += 1;
        // LM Head
        adam.update(idx, &mut self.lm_head.weight.data, &grads.d_lm_head_weight.data); idx += 1;
        adam.update(idx, &mut self.lm_head.bias.data, &grads.d_lm_head_bias.data);
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

    // ==================== backward ====================

    #[test]
    fn test_backward_shapes() {
        let model = make_tiny_model();
        let token_ids = &[0, 1, 2];
        let d_logits = Tensor::new(vec![0.1; 3 * 10], vec![3, 10]);
        let grads = model.backward(&d_logits, token_ids);

        assert_eq!(grads.d_embedding_weight.shape, vec![10, 8]);
        assert_eq!(grads.block_grads.len(), 2);
        assert_eq!(grads.d_ln_final_gamma.shape, vec![8]);
        assert_eq!(grads.d_ln_final_beta.shape, vec![8]);
        assert_eq!(grads.d_lm_head_weight.shape, vec![10, 8]);
        assert_eq!(grads.d_lm_head_bias.shape, vec![10]);
    }

    #[test]
    fn test_backward_numerical_d_lm_head_weight() {
        // Use a tiny model: vocab=4, d_model=4, n_heads=2, d_ff=8, n_layers=1
        let model = Transformer::rand(4, 4, 2, 8, 1);
        let token_ids = &[0, 1];
        let d_logits = Tensor::new(
            vec![0.1, -0.2, 0.3, 0.0, 0.5, -0.1, 0.2, -0.4],
            vec![2, 4],
        );
        let grads = model.backward(&d_logits, token_ids);

        let eps = 1e-4;
        let out_size = 8; // 2x4
        // Check first 4 elements of d_lm_head_weight
        for i in 0..4 {
            let mut model_plus = Transformer::rand(4, 4, 2, 8, 1);
            // Copy all params
            model_plus.token_embedding = Embedding::new(model.token_embedding.weight.clone());
            model_plus.blocks = vec![TransformerBlock::new(
                LayerNorm::new(4),
                crate::multi_head_attention::MultiHeadAttention::new(
                    Linear::new(model.blocks[0].mha.wq.weight.clone(), model.blocks[0].mha.wq.bias.clone()),
                    Linear::new(model.blocks[0].mha.wk.weight.clone(), model.blocks[0].mha.wk.bias.clone()),
                    Linear::new(model.blocks[0].mha.wv.weight.clone(), model.blocks[0].mha.wv.bias.clone()),
                    Linear::new(model.blocks[0].mha.wo.weight.clone(), model.blocks[0].mha.wo.bias.clone()),
                    2,
                ),
                LayerNorm::new(4),
                crate::feed_forward::FeedForward::new(
                    Linear::new(model.blocks[0].ffn.linear1.weight.clone(), model.blocks[0].ffn.linear1.bias.clone()),
                    Linear::new(model.blocks[0].ffn.linear2.weight.clone(), model.blocks[0].ffn.linear2.bias.clone()),
                ),
            )];
            model_plus.ln_final = LayerNorm::new(4);
            model_plus.lm_head = Linear::new(
                model.lm_head.weight.clone(),
                model.lm_head.bias.clone(),
            );
            model_plus.lm_head.weight.data[i] += eps;

            let mut model_minus = Transformer::rand(4, 4, 2, 8, 1);
            model_minus.token_embedding = Embedding::new(model.token_embedding.weight.clone());
            model_minus.blocks = vec![TransformerBlock::new(
                LayerNorm::new(4),
                crate::multi_head_attention::MultiHeadAttention::new(
                    Linear::new(model.blocks[0].mha.wq.weight.clone(), model.blocks[0].mha.wq.bias.clone()),
                    Linear::new(model.blocks[0].mha.wk.weight.clone(), model.blocks[0].mha.wk.bias.clone()),
                    Linear::new(model.blocks[0].mha.wv.weight.clone(), model.blocks[0].mha.wv.bias.clone()),
                    Linear::new(model.blocks[0].mha.wo.weight.clone(), model.blocks[0].mha.wo.bias.clone()),
                    2,
                ),
                LayerNorm::new(4),
                crate::feed_forward::FeedForward::new(
                    Linear::new(model.blocks[0].ffn.linear1.weight.clone(), model.blocks[0].ffn.linear1.bias.clone()),
                    Linear::new(model.blocks[0].ffn.linear2.weight.clone(), model.blocks[0].ffn.linear2.bias.clone()),
                ),
            )];
            model_minus.ln_final = LayerNorm::new(4);
            model_minus.lm_head = Linear::new(
                model.lm_head.weight.clone(),
                model.lm_head.bias.clone(),
            );
            model_minus.lm_head.weight.data[i] -= eps;

            let y_plus = model_plus.forward(token_ids);
            let y_minus = model_minus.forward(token_ids);
            let mut numerical = 0.0;
            for j in 0..out_size {
                numerical += (y_plus.data[j] - y_minus.data[j]) / (2.0 * eps) * d_logits.data[j];
            }
            assert!(
                (grads.d_lm_head_weight.data[i] - numerical).abs() < 1e-2,
                "d_lm_head_weight[{}]: analytical {} vs numerical {}",
                i,
                grads.d_lm_head_weight.data[i],
                numerical
            );
        }
    }

    #[test]
    fn test_backward_gradients_nonzero() {
        // Sanity check: gradients should not be all zeros
        let model = make_tiny_model();
        let token_ids = &[0, 1, 2];
        let d_logits = Tensor::new(vec![0.1; 3 * 10], vec![3, 10]);
        let grads = model.backward(&d_logits, token_ids);

        let emb_sum: f32 = grads.d_embedding_weight.data.iter().map(|v| v.abs()).sum();
        assert!(emb_sum > 0.0, "d_embedding_weight should not be all zeros");

        let lm_sum: f32 = grads.d_lm_head_weight.data.iter().map(|v| v.abs()).sum();
        assert!(lm_sum > 0.0, "d_lm_head_weight should not be all zeros");
    }
}
