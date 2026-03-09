use rand::RngExt;

use crate::tensor::Tensor;

pub struct Linear {
    pub weight: Tensor, // [out_features, in_features]
    pub bias: Tensor,   // [out_features]
}

impl Linear {
    /// Create with explicit weights (for testing)
    pub fn new(weight: Tensor, bias: Tensor) -> Self {
        assert_eq!(weight.shape.len(), 2, "weight must be 2D");
        let out_features = weight.shape[0];
        assert_eq!(bias.shape, vec![out_features], "bias shape mismatch");
        Self { weight, bias }
    }

    /// Create with random weights (Xavier initialization)
    pub fn rand(in_features: usize, out_features: usize) -> Self {
        let mut rng = rand::rng();
        let limit = (6.0 / (in_features + out_features) as f64).sqrt() as f32;
        let weight_data: Vec<f32> = (0..out_features * in_features)
            .map(|_| rng.random_range(-limit..limit))
            .collect();
        let bias_data = vec![0.0; out_features];
        Self {
            weight: Tensor::new(weight_data, vec![out_features, in_features]),
            bias: Tensor::new(bias_data, vec![out_features]),
        }
    }

    /// y = x @ W^T + b
    /// input: [batch, in_features] -> output: [batch, out_features]
    pub fn forward(&self, input: &Tensor) -> Tensor {
        assert_eq!(input.shape.len(), 2, "input must be 2D");
        let wt = self.weight.transpose();
        let out = input.matmul(&wt);
        // add bias to each row
        let batch = out.shape[0];
        let out_features = out.shape[1];
        let bias_broadcast = Tensor::new(
            self.bias.data.repeat(batch),
            vec![batch, out_features],
        );
        out.add(&bias_broadcast)
    }

    /// Backward pass for Linear layer.
    /// d_output: [batch, out_features]
    /// input: [batch, in_features] (cached from forward)
    /// Returns: (d_input, d_weight, d_bias)
    pub fn backward(&self, d_output: &Tensor, input: &Tensor) -> (Tensor, Tensor, Tensor) {
        // y = x @ W^T + b
        // d_input = d_output @ W          [batch, in]
        // d_weight = d_output^T @ input   [out, in]
        // d_bias = sum(d_output, axis=0)  [out]
        let d_input = d_output.matmul(&self.weight);
        let d_weight = d_output.transpose().matmul(input);

        let batch = d_output.shape[0];
        let out_features = d_output.shape[1];
        let mut d_bias_data = vec![0.0_f32; out_features];
        for b in 0..batch {
            for o in 0..out_features {
                d_bias_data[o] += d_output.data[b * out_features + o];
            }
        }
        let d_bias = Tensor::new(d_bias_data, vec![out_features]);

        (d_input, d_weight, d_bias)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let w = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = Tensor::new(vec![0.1, 0.2], vec![2]);
        let linear = Linear::new(w, b);
        assert_eq!(linear.weight.shape, vec![2, 3]);
        assert_eq!(linear.bias.shape, vec![2]);
    }

    #[test]
    #[should_panic]
    fn test_new_weight_not_2d() {
        let w = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let b = Tensor::new(vec![0.1], vec![1]);
        Linear::new(w, b);
    }

    #[test]
    #[should_panic]
    fn test_new_bias_shape_mismatch() {
        let w = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = Tensor::new(vec![0.1, 0.2, 0.3], vec![3]); // should be [2]
        Linear::new(w, b);
    }

    #[test]
    fn test_forward() {
        // W: [2, 3], b: [2]
        // x: [1, 3] (single sample)
        //
        // y = x @ W^T + b
        // W^T: [3, 2]
        // x @ W^T: [1, 3] @ [3, 2] = [1, 2]
        //
        // x = [1, 2, 3]
        // W = [[1, 2, 3],
        //      [4, 5, 6]]
        // b = [0.1, 0.2]
        //
        // y[0] = 1*1 + 2*2 + 3*3 + 0.1 = 14.1
        // y[1] = 1*4 + 2*5 + 3*6 + 0.2 = 32.2
        let w = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = Tensor::new(vec![0.1, 0.2], vec![2]);
        let linear = Linear::new(w, b);

        let x = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let y = linear.forward(&x);
        assert_eq!(y.shape, vec![1, 2]);
        assert!((y.get(&[0, 0]) - 14.1).abs() < 1e-5);
        assert!((y.get(&[0, 1]) - 32.2).abs() < 1e-5);
    }

    #[test]
    fn test_forward_batch() {
        // x: [2, 3] (two samples)
        let w = Tensor::new(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0], vec![2, 3]);
        let b = Tensor::new(vec![0.0, 0.0], vec![2]);
        let linear = Linear::new(w, b);

        // identity-like: picks first two elements
        let x = Tensor::new(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], vec![2, 3]);
        let y = linear.forward(&x);
        assert_eq!(y.shape, vec![2, 2]);
        assert!((y.get(&[0, 0]) - 10.0).abs() < 1e-5);
        assert!((y.get(&[0, 1]) - 20.0).abs() < 1e-5);
        assert!((y.get(&[1, 0]) - 40.0).abs() < 1e-5);
        assert!((y.get(&[1, 1]) - 50.0).abs() < 1e-5);
    }

    #[test]
    fn test_rand_shape() {
        let linear = Linear::rand(4, 3);
        assert_eq!(linear.weight.shape, vec![3, 4]);
        assert_eq!(linear.bias.shape, vec![3]);
    }

    #[test]
    #[should_panic]
    fn test_forward_shape_mismatch() {
        let w = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = Tensor::new(vec![0.1, 0.2], vec![2]);
        let linear = Linear::new(w, b);

        let x = Tensor::new(vec![1.0, 2.0], vec![1, 2]); // in_features mismatch
        linear.forward(&x);
    }

    // ==================== backward ====================

    #[test]
    fn test_backward_shapes() {
        let linear = Linear::rand(3, 2);
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let d_output = Tensor::new(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let (d_input, d_weight, d_bias) = linear.backward(&d_output, &x);
        assert_eq!(d_input.shape, vec![2, 3]);
        assert_eq!(d_weight.shape, vec![2, 3]);
        assert_eq!(d_bias.shape, vec![2]);
    }

    #[test]
    fn test_backward_numerical_gradient() {
        let linear = Linear::new(
            Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
            Tensor::new(vec![0.5, -0.5], vec![2]),
        );
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let d_output = Tensor::new(vec![1.0, 0.5, -0.5, 1.0], vec![2, 2]);
        let (d_input, d_weight, d_bias) = linear.backward(&d_output, &x);

        let eps = 1e-4;

        // Check d_input
        for i in 0..x.data.len() {
            let mut x_plus = x.clone();
            x_plus.data[i] += eps;
            let mut x_minus = x.clone();
            x_minus.data[i] -= eps;
            let y_plus = linear.forward(&x_plus);
            let y_minus = linear.forward(&x_minus);
            let mut numerical = 0.0;
            for j in 0..d_output.data.len() {
                numerical += (y_plus.data[j] - y_minus.data[j]) / (2.0 * eps) * d_output.data[j];
            }
            assert!(
                (d_input.data[i] - numerical).abs() < 1e-2,
                "d_input[{}]: analytical {} vs numerical {}", i, d_input.data[i], numerical
            );
        }

        // Check d_weight
        for i in 0..linear.weight.data.len() {
            let mut w_plus = linear.weight.clone();
            w_plus.data[i] += eps;
            let mut w_minus = linear.weight.clone();
            w_minus.data[i] -= eps;
            let l_plus = Linear::new(w_plus, linear.bias.clone());
            let l_minus = Linear::new(w_minus, linear.bias.clone());
            let y_plus = l_plus.forward(&x);
            let y_minus = l_minus.forward(&x);
            let mut numerical = 0.0;
            for j in 0..d_output.data.len() {
                numerical += (y_plus.data[j] - y_minus.data[j]) / (2.0 * eps) * d_output.data[j];
            }
            assert!(
                (d_weight.data[i] - numerical).abs() < 1e-2,
                "d_weight[{}]: {} vs {}", i, d_weight.data[i], numerical
            );
        }

        // Check d_bias
        for i in 0..linear.bias.data.len() {
            let mut b_plus = linear.bias.clone();
            b_plus.data[i] += eps;
            let mut b_minus = linear.bias.clone();
            b_minus.data[i] -= eps;
            let l_plus = Linear::new(linear.weight.clone(), b_plus);
            let l_minus = Linear::new(linear.weight.clone(), b_minus);
            let y_plus = l_plus.forward(&x);
            let y_minus = l_minus.forward(&x);
            let mut numerical = 0.0;
            for j in 0..d_output.data.len() {
                numerical += (y_plus.data[j] - y_minus.data[j]) / (2.0 * eps) * d_output.data[j];
            }
            assert!(
                (d_bias.data[i] - numerical).abs() < 1e-2,
                "d_bias[{}]: {} vs {}", i, d_bias.data[i], numerical
            );
        }
    }
}
