use crate::activation::{gelu, gelu_backward};
use crate::linear::Linear;
use crate::tensor::Tensor;

pub struct FFNGradients {
    pub d_x: Tensor,
    pub d_l1_weight: Tensor,
    pub d_l1_bias: Tensor,
    pub d_l2_weight: Tensor,
    pub d_l2_bias: Tensor,
}

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

    /// Backward pass for FFN.
    /// d_output: [seq_len, d_model]
    /// x: cached input from forward
    pub fn backward(&self, d_output: &Tensor, x: &Tensor) -> FFNGradients {
        // Recompute forward intermediates
        let hidden = self.linear1.forward(x);
        let activated = gelu(&hidden);

        // Backward through linear2
        let (d_activated, d_l2_weight, d_l2_bias) = self.linear2.backward(d_output, &activated);

        // Backward through GELU
        let d_hidden = gelu_backward(&d_activated, &hidden);

        // Backward through linear1
        let (d_x, d_l1_weight, d_l1_bias) = self.linear1.backward(&d_hidden, x);

        FFNGradients {
            d_x,
            d_l1_weight,
            d_l1_bias,
            d_l2_weight,
            d_l2_bias,
        }
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

    // ==================== backward ====================

    #[test]
    fn test_backward_shapes() {
        // d_model=4, d_ff=8
        let ffn = FeedForward::rand(4, 8);
        let x = Tensor::new(vec![0.1; 2 * 4], vec![2, 4]);
        let d_out = Tensor::new(vec![0.1; 2 * 4], vec![2, 4]);
        let grads = ffn.backward(&d_out, &x);
        assert_eq!(grads.d_x.shape, vec![2, 4]);
        assert_eq!(grads.d_l1_weight.shape, vec![8, 4]);
        assert_eq!(grads.d_l1_bias.shape, vec![8]);
        assert_eq!(grads.d_l2_weight.shape, vec![4, 8]);
        assert_eq!(grads.d_l2_bias.shape, vec![4]);
    }

    #[test]
    fn test_backward_numerical_d_x() {
        let ffn = FeedForward::rand(4, 8);
        let x = Tensor::new(
            vec![0.1, -0.2, 0.3, 0.4, 0.5, 0.1, -0.3, 0.2],
            vec![2, 4],
        );
        let d_out = Tensor::new(
            vec![0.5, -0.3, 0.1, 0.2, -0.1, 0.4, -0.2, 0.6],
            vec![2, 4],
        );
        let grads = ffn.backward(&d_out, &x);

        let eps = 1e-4;
        let out_size = 8;
        for i in 0..8 {
            let mut x_plus = x.clone();
            x_plus.data[i] += eps;
            let mut x_minus = x.clone();
            x_minus.data[i] -= eps;
            let y_plus = ffn.forward(&x_plus);
            let y_minus = ffn.forward(&x_minus);
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

    #[test]
    fn test_backward_numerical_d_l1_weight() {
        let ffn = FeedForward::rand(4, 8);
        let x = Tensor::new(
            vec![0.1, -0.2, 0.3, 0.4, 0.5, 0.1, -0.3, 0.2],
            vec![2, 4],
        );
        let d_out = Tensor::new(
            vec![0.5, -0.3, 0.1, 0.2, -0.1, 0.4, -0.2, 0.6],
            vec![2, 4],
        );
        let grads = ffn.backward(&d_out, &x);

        let eps = 1e-4;
        let out_size = 8;
        // Check a few elements
        for i in 0..4 {
            let mut ffn_plus = FeedForward::new(
                Linear::new(ffn.linear1.weight.clone(), ffn.linear1.bias.clone()),
                Linear::new(ffn.linear2.weight.clone(), ffn.linear2.bias.clone()),
            );
            ffn_plus.linear1.weight.data[i] += eps;
            let mut ffn_minus = FeedForward::new(
                Linear::new(ffn.linear1.weight.clone(), ffn.linear1.bias.clone()),
                Linear::new(ffn.linear2.weight.clone(), ffn.linear2.bias.clone()),
            );
            ffn_minus.linear1.weight.data[i] -= eps;

            let y_plus = ffn_plus.forward(&x);
            let y_minus = ffn_minus.forward(&x);
            let mut numerical = 0.0;
            for j in 0..out_size {
                numerical += (y_plus.data[j] - y_minus.data[j]) / (2.0 * eps) * d_out.data[j];
            }
            assert!(
                (grads.d_l1_weight.data[i] - numerical).abs() < 1e-2,
                "d_l1_weight[{}]: analytical {} vs numerical {}",
                i,
                grads.d_l1_weight.data[i],
                numerical
            );
        }
    }
}
