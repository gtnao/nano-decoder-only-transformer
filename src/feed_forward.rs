use crate::activation::gelu;
use crate::linear::Linear;
use crate::tensor::Tensor;

/// Position-wise Feed-Forward Network
/// FFN(x) = GELU(x @ W1 + b1) @ W2 + b2
pub struct FeedForward {
    pub linear1: Linear, // [d_model, d_ff]
    pub linear2: Linear, // [d_ff, d_model]
}

impl FeedForward {
    pub fn new(linear1: Linear, linear2: Linear) -> Self {
        Self { linear1, linear2 }
    }

    pub fn rand(d_model: usize, d_ff: usize) -> Self {
        Self {
            linear1: Linear::rand(d_model, d_ff),
            linear2: Linear::rand(d_ff, d_model),
        }
    }

    /// x: [seq_len, d_model] -> [seq_len, d_model]
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let hidden = self.linear1.forward(x);
        let activated = gelu(&hidden);
        self.linear2.forward(&activated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rand_shapes() {
        let ffn = FeedForward::rand(8, 32);
        assert_eq!(ffn.linear1.weight.shape, vec![32, 8]);
        assert_eq!(ffn.linear2.weight.shape, vec![8, 32]);
    }

    #[test]
    fn test_forward_shape() {
        let ffn = FeedForward::rand(8, 32);
        let x = Tensor::new(vec![0.1; 3 * 8], vec![3, 8]);
        let out = ffn.forward(&x);
        assert_eq!(out.shape, vec![3, 8]);
    }

    #[test]
    fn test_forward_known_values() {
        // d_model=2, d_ff=3
        // W1: [3, 2], b1: [3], W2: [2, 3], b2: [2]
        let l1 = Linear::new(
            Tensor::new(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], vec![3, 2]),
            Tensor::new(vec![0.0, 0.0, 0.0], vec![3]),
        );
        let l2 = Linear::new(
            Tensor::new(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0], vec![2, 3]),
            Tensor::new(vec![0.0, 0.0], vec![2]),
        );
        let ffn = FeedForward::new(l1, l2);

        // x = [[1, 2]]
        // hidden = x @ W1^T = [1*1+2*0, 1*0+2*1, 1*1+2*1] = [1, 2, 3]
        // after GELU: [gelu(1), gelu(2), gelu(3)]
        // output = gelu_hidden @ W2^T
        // W2^T picks: out[0] = gelu(1), out[1] = gelu(2)
        let x = Tensor::new(vec![1.0, 2.0], vec![1, 2]);
        let out = ffn.forward(&x);
        assert_eq!(out.shape, vec![1, 2]);

        let expected_0 = gelu(&Tensor::new(vec![1.0], vec![1])).data[0];
        let expected_1 = gelu(&Tensor::new(vec![2.0], vec![1])).data[0];
        assert!((out.data[0] - expected_0).abs() < 1e-5);
        assert!((out.data[1] - expected_1).abs() < 1e-5);
    }

    #[test]
    fn test_forward_deterministic() {
        let ffn = FeedForward::rand(8, 32);
        let x = Tensor::new(vec![0.5; 3 * 8], vec![3, 8]);
        let out1 = ffn.forward(&x);
        let out2 = ffn.forward(&x);
        assert_eq!(out1.data, out2.data);
    }

    #[test]
    fn test_forward_position_wise() {
        // Each position (row) is processed independently
        // Same input row => same output row
        let ffn = FeedForward::rand(4, 16);
        let row = vec![0.5; 4];
        let x = Tensor::new([row.clone(), row.clone(), row.clone()].concat(), vec![3, 4]);
        let out = ffn.forward(&x);
        // All 3 rows should be identical
        assert_eq!(&out.data[0..4], &out.data[4..8]);
        assert_eq!(&out.data[4..8], &out.data[8..12]);
    }
}
