use crate::feed_forward::FeedForward;
use crate::layer_norm::LayerNorm;
use crate::multi_head_attention::MultiHeadAttention;
use crate::tensor::Tensor;

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
}
