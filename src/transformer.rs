use crate::embedding::Embedding;
use crate::feed_forward::FFNGradients;
use crate::layer_norm::LayerNorm;
use crate::linear::Linear;
use crate::multi_head_attention::MHAGradients;
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
    /// Create zero-initialized gradients matching the model structure.
    pub fn zeros_like(model: &Transformer) -> Self {
        let vocab_size = model.token_embedding.weight.shape[0];
        let d_model = model.token_embedding.weight.shape[1];

        let block_grads = model.blocks.iter().map(|block| {
            let d_ff = block.ffn.linear1.weight.shape[0];
            TransformerBlockGradients {
                d_x: Tensor::zeros(vec![1, 1]), // unused placeholder
                mha_grads: MHAGradients {
                    d_x: Tensor::zeros(vec![1, 1]),
                    d_wq_weight: Tensor::zeros(vec![d_model, d_model]),
                    d_wq_bias: Tensor::zeros(vec![d_model]),
                    d_wk_weight: Tensor::zeros(vec![d_model, d_model]),
                    d_wk_bias: Tensor::zeros(vec![d_model]),
                    d_wv_weight: Tensor::zeros(vec![d_model, d_model]),
                    d_wv_bias: Tensor::zeros(vec![d_model]),
                    d_wo_weight: Tensor::zeros(vec![d_model, d_model]),
                    d_wo_bias: Tensor::zeros(vec![d_model]),
                },
                ffn_grads: FFNGradients {
                    d_x: Tensor::zeros(vec![1, 1]),
                    d_l1_weight: Tensor::zeros(vec![d_ff, d_model]),
                    d_l1_bias: Tensor::zeros(vec![d_ff]),
                    d_l2_weight: Tensor::zeros(vec![d_model, d_ff]),
                    d_l2_bias: Tensor::zeros(vec![d_model]),
                },
                d_ln1_gamma: Tensor::zeros(vec![d_model]),
                d_ln1_beta: Tensor::zeros(vec![d_model]),
                d_ln2_gamma: Tensor::zeros(vec![d_model]),
                d_ln2_beta: Tensor::zeros(vec![d_model]),
            }
        }).collect();

        TransformerGradients {
            d_embedding_weight: Tensor::zeros(vec![vocab_size, d_model]),
            block_grads,
            d_ln_final_gamma: Tensor::zeros(vec![d_model]),
            d_ln_final_beta: Tensor::zeros(vec![d_model]),
            d_lm_head_weight: Tensor::zeros(vec![vocab_size, d_model]),
            d_lm_head_bias: Tensor::zeros(vec![vocab_size]),
        }
    }

    /// Accumulate gradients from another set (in-place addition).
    pub fn accumulate(&mut self, other: &TransformerGradients) {
        self.d_embedding_weight.add_inplace(&other.d_embedding_weight);
        self.d_ln_final_gamma.add_inplace(&other.d_ln_final_gamma);
        self.d_ln_final_beta.add_inplace(&other.d_ln_final_beta);
        self.d_lm_head_weight.add_inplace(&other.d_lm_head_weight);
        self.d_lm_head_bias.add_inplace(&other.d_lm_head_bias);

        for (sg, og) in self.block_grads.iter_mut().zip(other.block_grads.iter()) {
            sg.d_ln1_gamma.add_inplace(&og.d_ln1_gamma);
            sg.d_ln1_beta.add_inplace(&og.d_ln1_beta);
            sg.d_ln2_gamma.add_inplace(&og.d_ln2_gamma);
            sg.d_ln2_beta.add_inplace(&og.d_ln2_beta);
            sg.mha_grads.d_wq_weight.add_inplace(&og.mha_grads.d_wq_weight);
            sg.mha_grads.d_wq_bias.add_inplace(&og.mha_grads.d_wq_bias);
            sg.mha_grads.d_wk_weight.add_inplace(&og.mha_grads.d_wk_weight);
            sg.mha_grads.d_wk_bias.add_inplace(&og.mha_grads.d_wk_bias);
            sg.mha_grads.d_wv_weight.add_inplace(&og.mha_grads.d_wv_weight);
            sg.mha_grads.d_wv_bias.add_inplace(&og.mha_grads.d_wv_bias);
            sg.mha_grads.d_wo_weight.add_inplace(&og.mha_grads.d_wo_weight);
            sg.mha_grads.d_wo_bias.add_inplace(&og.mha_grads.d_wo_bias);
            sg.ffn_grads.d_l1_weight.add_inplace(&og.ffn_grads.d_l1_weight);
            sg.ffn_grads.d_l1_bias.add_inplace(&og.ffn_grads.d_l1_bias);
            sg.ffn_grads.d_l2_weight.add_inplace(&og.ffn_grads.d_l2_weight);
            sg.ffn_grads.d_l2_bias.add_inplace(&og.ffn_grads.d_l2_bias);
        }
    }

    /// Scale all gradients by a scalar (for batch averaging).
    pub fn scale(&mut self, s: f32) {
        self.d_embedding_weight.scale_inplace(s);
        self.d_ln_final_gamma.scale_inplace(s);
        self.d_ln_final_beta.scale_inplace(s);
        self.d_lm_head_weight.scale_inplace(s);
        self.d_lm_head_bias.scale_inplace(s);

        for bg in &mut self.block_grads {
            bg.d_ln1_gamma.scale_inplace(s);
            bg.d_ln1_beta.scale_inplace(s);
            bg.d_ln2_gamma.scale_inplace(s);
            bg.d_ln2_beta.scale_inplace(s);
            bg.mha_grads.d_wq_weight.scale_inplace(s);
            bg.mha_grads.d_wq_bias.scale_inplace(s);
            bg.mha_grads.d_wk_weight.scale_inplace(s);
            bg.mha_grads.d_wk_bias.scale_inplace(s);
            bg.mha_grads.d_wv_weight.scale_inplace(s);
            bg.mha_grads.d_wv_bias.scale_inplace(s);
            bg.mha_grads.d_wo_weight.scale_inplace(s);
            bg.mha_grads.d_wo_bias.scale_inplace(s);
            bg.ffn_grads.d_l1_weight.scale_inplace(s);
            bg.ffn_grads.d_l1_bias.scale_inplace(s);
            bg.ffn_grads.d_l2_weight.scale_inplace(s);
            bg.ffn_grads.d_l2_bias.scale_inplace(s);
        }
    }

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

    /// Collect all parameter data into a flat Vec in deterministic order.
    fn collect_params(&self) -> Vec<f32> {
        let mut params = Vec::new();
        params.extend_from_slice(&self.token_embedding.weight.data);
        for block in &self.blocks {
            params.extend_from_slice(&block.ln1.gamma.data);
            params.extend_from_slice(&block.ln1.beta.data);
            params.extend_from_slice(&block.mha.wq.weight.data);
            params.extend_from_slice(&block.mha.wq.bias.data);
            params.extend_from_slice(&block.mha.wk.weight.data);
            params.extend_from_slice(&block.mha.wk.bias.data);
            params.extend_from_slice(&block.mha.wv.weight.data);
            params.extend_from_slice(&block.mha.wv.bias.data);
            params.extend_from_slice(&block.mha.wo.weight.data);
            params.extend_from_slice(&block.mha.wo.bias.data);
            params.extend_from_slice(&block.ln2.gamma.data);
            params.extend_from_slice(&block.ln2.beta.data);
            params.extend_from_slice(&block.ffn.linear1.weight.data);
            params.extend_from_slice(&block.ffn.linear1.bias.data);
            params.extend_from_slice(&block.ffn.linear2.weight.data);
            params.extend_from_slice(&block.ffn.linear2.bias.data);
        }
        params.extend_from_slice(&self.ln_final.gamma.data);
        params.extend_from_slice(&self.ln_final.beta.data);
        params.extend_from_slice(&self.lm_head.weight.data);
        params.extend_from_slice(&self.lm_head.bias.data);
        params
    }

    /// Load parameter data from a flat slice in the same order as collect_params.
    fn load_params(&mut self, data: &[f32]) {
        let mut offset = 0;
        let mut read = |dst: &mut [f32]| {
            dst.copy_from_slice(&data[offset..offset + dst.len()]);
            offset += dst.len();
        };
        read(&mut self.token_embedding.weight.data);
        for block in &mut self.blocks {
            read(&mut block.ln1.gamma.data);
            read(&mut block.ln1.beta.data);
            read(&mut block.mha.wq.weight.data);
            read(&mut block.mha.wq.bias.data);
            read(&mut block.mha.wk.weight.data);
            read(&mut block.mha.wk.bias.data);
            read(&mut block.mha.wv.weight.data);
            read(&mut block.mha.wv.bias.data);
            read(&mut block.mha.wo.weight.data);
            read(&mut block.mha.wo.bias.data);
            read(&mut block.ln2.gamma.data);
            read(&mut block.ln2.beta.data);
            read(&mut block.ffn.linear1.weight.data);
            read(&mut block.ffn.linear1.bias.data);
            read(&mut block.ffn.linear2.weight.data);
            read(&mut block.ffn.linear2.bias.data);
        }
        read(&mut self.ln_final.gamma.data);
        read(&mut self.ln_final.beta.data);
        read(&mut self.lm_head.weight.data);
        read(&mut self.lm_head.bias.data);
    }

    /// Save model to a binary file.
    /// Format: [vocab_size, d_model, n_heads, d_ff, n_layers] as u32, then f32 params.
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        use std::io::Write;
        let vocab_size = self.token_embedding.weight.shape[0] as u32;
        let d_model = self.token_embedding.weight.shape[1] as u32;
        let n_heads = self.blocks[0].mha.n_heads as u32;
        let d_ff = self.blocks[0].ffn.linear1.weight.shape[0] as u32;
        let n_layers = self.blocks.len() as u32;

        let mut file = std::fs::File::create(path)?;
        // Write header
        for v in &[vocab_size, d_model, n_heads, d_ff, n_layers] {
            file.write_all(&v.to_le_bytes())?;
        }
        // Write params
        let params = self.collect_params();
        for v in &params {
            file.write_all(&v.to_le_bytes())?;
        }
        Ok(())
    }

    /// Load model from a binary file.
    pub fn load(path: &str) -> std::io::Result<Self> {
        use std::io::Read;
        let mut file = std::fs::File::open(path)?;

        // Read header
        let mut buf = [0u8; 4];
        let mut read_u32 = |f: &mut std::fs::File| -> std::io::Result<u32> {
            f.read_exact(&mut buf)?;
            Ok(u32::from_le_bytes(buf))
        };
        let vocab_size = read_u32(&mut file)? as usize;
        let d_model = read_u32(&mut file)? as usize;
        let n_heads = read_u32(&mut file)? as usize;
        let d_ff = read_u32(&mut file)? as usize;
        let n_layers = read_u32(&mut file)? as usize;

        // Create model with correct structure
        let mut model = Self::rand(vocab_size, d_model, n_heads, d_ff, n_layers);

        // Read params
        let mut raw = Vec::new();
        file.read_to_end(&mut raw)?;
        let params: Vec<f32> = raw
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        model.load_params(&params);
        Ok(model)
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

    #[test]
    fn test_save_load_roundtrip() {
        let model = make_tiny_model();
        let path = "/tmp/test_model.bin";

        model.save(path).unwrap();
        let loaded = Transformer::load(path).unwrap();

        // Check structure matches
        assert_eq!(
            model.token_embedding.weight.shape,
            loaded.token_embedding.weight.shape
        );
        assert_eq!(model.blocks.len(), loaded.blocks.len());

        // Check params match
        let orig_params = model.collect_params();
        let loaded_params = loaded.collect_params();
        assert_eq!(orig_params.len(), loaded_params.len());
        for (a, b) in orig_params.iter().zip(loaded_params.iter()) {
            assert_eq!(a, b);
        }

        // Check forward produces same output
        let out1 = model.forward(&[0, 1, 2]);
        let out2 = loaded.forward(&[0, 1, 2]);
        assert_eq!(out1.data, out2.data);

        std::fs::remove_file(path).ok();
    }
}
