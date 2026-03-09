use std::cell::RefCell;
use crate::dropout::{dropout, dropout_backward};
use crate::feed_forward::{FeedForward, FFNGradients};
use crate::layer_norm::LayerNorm;
use crate::multi_head_attention::{MultiHeadAttention, MHAGradients};
use crate::tensor::Tensor;

pub struct TransformerBlockGradients {
    pub d_x: Tensor,
    pub mha_grads: MHAGradients,
    pub ffn_grads: FFNGradients,
    pub d_ln1_gamma: Tensor,
    pub d_ln1_beta: Tensor,
    pub d_ln2_gamma: Tensor,
    pub d_ln2_beta: Tensor,
}

/// Pre-LayerNorm Transformer Block:
/// x = x + dropout(MHA(LayerNorm1(x)))
/// x = x + dropout(FFN(LayerNorm2(x)))
pub struct TransformerBlock {
    pub ln1: LayerNorm,
    pub mha: MultiHeadAttention,
    pub ln2: LayerNorm,
    pub ffn: FeedForward,
    pub dropout_rate: f32,
    // Cached dropout masks from forward (for backward)
    dropout_masks: RefCell<Option<(Vec<bool>, Vec<bool>)>>,
}

impl TransformerBlock {
    pub fn new(ln1: LayerNorm, mha: MultiHeadAttention, ln2: LayerNorm, ffn: FeedForward) -> Self {
        Self { ln1, mha, ln2, ffn, dropout_rate: 0.0, dropout_masks: RefCell::new(None) }
    }

    pub fn rand(d_model: usize, n_heads: usize, d_ff: usize) -> Self {
        Self {
            ln1: LayerNorm::new(d_model),
            mha: MultiHeadAttention::rand(d_model, n_heads),
            ln2: LayerNorm::new(d_model),
            ffn: FeedForward::rand(d_model, d_ff),
            dropout_rate: 0.0,
            dropout_masks: RefCell::new(None),
        }
    }

    pub fn rand_with_dropout(d_model: usize, n_heads: usize, d_ff: usize, dropout_rate: f32) -> Self {
        Self {
            ln1: LayerNorm::new(d_model),
            mha: MultiHeadAttention::rand(d_model, n_heads),
            ln2: LayerNorm::new(d_model),
            ffn: FeedForward::rand(d_model, d_ff),
            dropout_rate,
            dropout_masks: RefCell::new(None),
        }
    }

    /// x: [seq_len, d_model] -> [seq_len, d_model]
    /// training: if true, apply dropout and cache masks
    pub fn forward(&self, x: &Tensor, training: bool) -> Tensor {
        let normed1 = self.ln1.forward(x);
        let attn_out = self.mha.forward(&normed1, true);
        let (attn_out, mask1) = if training && self.dropout_rate > 0.0 {
            dropout(&attn_out, self.dropout_rate)
        } else {
            let mask = vec![true; attn_out.data.len()];
            (attn_out, mask)
        };
        let x1 = x.add(&attn_out);

        let normed2 = self.ln2.forward(&x1);
        let ffn_out = self.ffn.forward(&normed2);
        let (ffn_out, mask2) = if training && self.dropout_rate > 0.0 {
            dropout(&ffn_out, self.dropout_rate)
        } else {
            let mask = vec![true; ffn_out.data.len()];
            (ffn_out, mask)
        };

        if training && self.dropout_rate > 0.0 {
            *self.dropout_masks.borrow_mut() = Some((mask1, mask2));
        }

        x1.add(&ffn_out)
    }

    /// Backward pass for Transformer Block.
    pub fn backward(&self, d_output: &Tensor, x: &Tensor) -> TransformerBlockGradients {
        let masks = self.dropout_masks.borrow().clone();

        // Recompute forward intermediates (without dropout for determinism)
        let normed1 = self.ln1.forward(x);
        let attn_out_raw = self.mha.forward(&normed1, true);
        let attn_out = if let Some((ref mask1, _)) = masks {
            // Apply cached mask to get same output as forward
            let rate = self.dropout_rate;
            let scale = 1.0 / (1.0 - rate);
            let data: Vec<f32> = attn_out_raw.data.iter().zip(mask1.iter())
                .map(|(&v, &keep)| if keep { v * scale } else { 0.0 })
                .collect();
            Tensor::new(data, attn_out_raw.shape.clone())
        } else {
            attn_out_raw
        };
        let x1 = x.add(&attn_out);

        let normed2 = self.ln2.forward(&x1);

        // --- Second residual: output = x1 + dropout(ffn(ln2(x1))) ---
        let mut d_ffn_out = d_output.clone();
        if let Some((_, ref mask2)) = masks {
            d_ffn_out = dropout_backward(&d_ffn_out, mask2, self.dropout_rate);
        }
        let ffn_grads = self.ffn.backward(&d_ffn_out, &normed2);
        let (d_x1_from_ln2, d_ln2_gamma, d_ln2_beta) = self.ln2.backward(&ffn_grads.d_x, &x1);
        let d_x1 = d_output.add(&d_x1_from_ln2);

        // --- First residual: x1 = x + dropout(mha(ln1(x))) ---
        let mut d_attn_out = d_x1.clone();
        if let Some((ref mask1, _)) = masks {
            d_attn_out = dropout_backward(&d_attn_out, mask1, self.dropout_rate);
        }
        let mha_grads = self.mha.backward(&d_attn_out, &normed1, true);
        let (d_x_from_ln1, d_ln1_gamma, d_ln1_beta) = self.ln1.backward(&mha_grads.d_x, x);
        let d_x = d_x1.add(&d_x_from_ln1);

        TransformerBlockGradients {
            d_x,
            mha_grads,
            ffn_grads,
            d_ln1_gamma,
            d_ln1_beta,
            d_ln2_gamma,
            d_ln2_beta,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rand() {
        let block = TransformerBlock::rand(8, 2, 32);
        assert_eq!(block.ln1.gamma.shape, vec![8]);
        assert_eq!(block.mha.wq.weight.shape, vec![8, 8]);
        assert_eq!(block.ln2.gamma.shape, vec![8]);
        assert_eq!(block.ffn.linear1.weight.shape, vec![32, 8]);
    }

    #[test]
    fn test_forward_shape() {
        let block = TransformerBlock::rand(8, 2, 32);
        let x = Tensor::new(vec![0.1; 3 * 8], vec![3, 8]);
        let out = block.forward(&x, false);
        assert_eq!(out.shape, vec![3, 8]);
    }

    #[test]
    fn test_residual_connection() {
        let block = TransformerBlock::rand(8, 2, 32);
        let x = Tensor::new(vec![1.0; 3 * 8], vec![3, 8]);
        let out = block.forward(&x, false);
        let sum: f32 = out.data.iter().map(|v| v.abs()).sum();
        assert!(sum > 0.0);
    }

    #[test]
    fn test_forward_deterministic() {
        let block = TransformerBlock::rand(8, 2, 32);
        let x = Tensor::new(vec![0.5; 3 * 8], vec![3, 8]);
        let out1 = block.forward(&x, false);
        let out2 = block.forward(&x, false);
        assert_eq!(out1.data, out2.data);
    }

    #[test]
    fn test_forward_different_inputs_differ() {
        let block = TransformerBlock::rand(8, 2, 32);
        let x1 = Tensor::new(vec![0.1; 3 * 8], vec![3, 8]);
        let x2 = Tensor::new(vec![0.9; 3 * 8], vec![3, 8]);
        let out1 = block.forward(&x1, false);
        let out2 = block.forward(&x2, false);
        assert_ne!(out1.data, out2.data);
    }

    #[test]
    fn test_forward_with_dropout() {
        let block = TransformerBlock::rand_with_dropout(8, 2, 32, 0.1);
        let x = Tensor::new(vec![0.1; 3 * 8], vec![3, 8]);
        let out = block.forward(&x, true);
        assert_eq!(out.shape, vec![3, 8]);
        // Masks should be cached
        assert!(block.dropout_masks.borrow().is_some());
    }

    #[test]
    fn test_forward_no_dropout_in_eval() {
        let block = TransformerBlock::rand_with_dropout(8, 2, 32, 0.1);
        let x = Tensor::new(vec![0.5; 3 * 8], vec![3, 8]);
        // Inference mode: deterministic
        let out1 = block.forward(&x, false);
        let out2 = block.forward(&x, false);
        assert_eq!(out1.data, out2.data);
    }

    // ==================== backward ====================

    #[test]
    fn test_backward_shapes() {
        let block = TransformerBlock::rand(4, 2, 8);
        let x = Tensor::new(vec![0.1; 2 * 4], vec![2, 4]);
        let d_out = Tensor::new(vec![0.1; 2 * 4], vec![2, 4]);
        let grads = block.backward(&d_out, &x);
        assert_eq!(grads.d_x.shape, vec![2, 4]);
        assert_eq!(grads.d_ln1_gamma.shape, vec![4]);
        assert_eq!(grads.d_ln2_gamma.shape, vec![4]);
    }

    #[test]
    fn test_backward_numerical_d_x() {
        let block = TransformerBlock::rand(4, 2, 8);
        let x = Tensor::new(
            vec![0.1, -0.2, 0.3, 0.4, 0.5, 0.1, -0.3, 0.2],
            vec![2, 4],
        );
        let d_out = Tensor::new(
            vec![0.5, -0.3, 0.1, 0.2, -0.1, 0.4, -0.2, 0.6],
            vec![2, 4],
        );
        // Use dropout_rate=0 so numerical gradient check works
        let grads = block.backward(&d_out, &x);

        let eps = 1e-4;
        let out_size = 8;
        for i in 0..8 {
            let mut x_plus = x.clone();
            x_plus.data[i] += eps;
            let mut x_minus = x.clone();
            x_minus.data[i] -= eps;
            let y_plus = block.forward(&x_plus, false);
            let y_minus = block.forward(&x_minus, false);
            let mut numerical = 0.0;
            for j in 0..out_size {
                numerical += (y_plus.data[j] - y_minus.data[j]) / (2.0 * eps) * d_out.data[j];
            }
            assert!(
                (grads.d_x.data[i] - numerical).abs() < 1e-2,
                "d_x[{}]: analytical {} vs numerical {}",
                i,
                grads.d_x.data[i],
                numerical
            );
        }
    }
}
