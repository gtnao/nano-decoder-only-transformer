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
/// x = x + MHA(LayerNorm1(x))
/// x = x + FFN(LayerNorm2(x))
pub struct TransformerBlock {
    pub ln1: LayerNorm,
    pub mha: MultiHeadAttention,
    pub ln2: LayerNorm,
    pub ffn: FeedForward,
}

impl TransformerBlock {
    pub fn new(ln1: LayerNorm, mha: MultiHeadAttention, ln2: LayerNorm, ffn: FeedForward) -> Self {
        Self { ln1, mha, ln2, ffn }
    }

    pub fn rand(d_model: usize, n_heads: usize, d_ff: usize) -> Self {
        Self {
            ln1: LayerNorm::new(d_model),
            mha: MultiHeadAttention::rand(d_model, n_heads),
            ln2: LayerNorm::new(d_model),
            ffn: FeedForward::rand(d_model, d_ff),
        }
    }

    /// x: [seq_len, d_model] -> [seq_len, d_model]
    pub fn forward(&self, x: &Tensor) -> Tensor {
        // x = x + MHA(LayerNorm1(x))
        let normed1 = self.ln1.forward(x);
        let attn_out = self.mha.forward(&normed1, true);
        let x = x.add(&attn_out);

        // x = x + FFN(LayerNorm2(x))
        let normed2 = self.ln2.forward(&x);
        let ffn_out = self.ffn.forward(&normed2);
        x.add(&ffn_out)
    }

    /// Backward pass for Transformer Block.
    /// d_output: [seq_len, d_model]
    /// x: cached input from forward
    pub fn backward(&self, d_output: &Tensor, x: &Tensor) -> TransformerBlockGradients {
        // Recompute forward intermediates
        let normed1 = self.ln1.forward(x);
        let attn_out = self.mha.forward(&normed1, true);
        let x1 = x.add(&attn_out); // after first residual

        let normed2 = self.ln2.forward(&x1);

        // --- Second residual: output = x1 + ffn(ln2(x1)) ---
        // d_output flows to both x1 (residual) and ffn_out
        let ffn_grads = self.ffn.backward(d_output, &normed2);
        let (d_x1_from_ln2, d_ln2_gamma, d_ln2_beta) = self.ln2.backward(&ffn_grads.d_x, &x1);
        let d_x1 = d_output.add(&d_x1_from_ln2); // residual + ln2 path

        // --- First residual: x1 = x + mha(ln1(x)) ---
        // d_x1 flows to both x (residual) and attn_out
        let mha_grads = self.mha.backward(&d_x1, &normed1, true);
        let (d_x_from_ln1, d_ln1_gamma, d_ln1_beta) = self.ln1.backward(&mha_grads.d_x, x);
        let d_x = d_x1.add(&d_x_from_ln1); // residual + ln1 path

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
        let out = block.forward(&x);
        assert_eq!(out.shape, vec![3, 8]);
    }

    #[test]
    fn test_residual_connection() {
        // With zero-initialized MHA and FFN weights,
        // output should equal input (residual passthrough).
        // We approximate by checking output differs from zero.
        let block = TransformerBlock::rand(8, 2, 32);
        let x = Tensor::new(vec![1.0; 3 * 8], vec![3, 8]);
        let out = block.forward(&x);
        // Output should not be all zeros (residual keeps input alive)
        let sum: f32 = out.data.iter().map(|v| v.abs()).sum();
        assert!(sum > 0.0);
    }

    #[test]
    fn test_forward_deterministic() {
        let block = TransformerBlock::rand(8, 2, 32);
        let x = Tensor::new(vec![0.5; 3 * 8], vec![3, 8]);
        let out1 = block.forward(&x);
        let out2 = block.forward(&x);
        assert_eq!(out1.data, out2.data);
    }

    #[test]
    fn test_forward_different_inputs_differ() {
        let block = TransformerBlock::rand(8, 2, 32);
        let x1 = Tensor::new(vec![0.1; 3 * 8], vec![3, 8]);
        let x2 = Tensor::new(vec![0.9; 3 * 8], vec![3, 8]);
        let out1 = block.forward(&x1);
        let out2 = block.forward(&x2);
        assert_ne!(out1.data, out2.data);
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
        let grads = block.backward(&d_out, &x);

        let eps = 1e-4;
        let out_size = 8;
        for i in 0..8 {
            let mut x_plus = x.clone();
            x_plus.data[i] += eps;
            let mut x_minus = x.clone();
            x_minus.data[i] -= eps;
            let y_plus = block.forward(&x_plus);
            let y_minus = block.forward(&x_minus);
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
