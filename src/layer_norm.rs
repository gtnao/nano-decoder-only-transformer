use crate::tensor::Tensor;

pub struct LayerNorm {
    pub gamma: Tensor, // [normalized_shape] scale
    pub beta: Tensor,  // [normalized_shape] shift
    pub eps: f32,
}

impl LayerNorm {
    pub fn new(normalized_shape: usize) -> Self {
        Self {
            gamma: Tensor::new(vec![1.0; normalized_shape], vec![normalized_shape]),
            beta: Tensor::new(vec![0.0; normalized_shape], vec![normalized_shape]),
            eps: 1e-5,
        }
    }

    /// output = gamma * (x - mean) / sqrt(var + eps) + beta
    /// Normalizes along the last axis.
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let last_dim = *x.shape.last().expect("empty shape");
        let num_groups = x.data.len() / last_dim;
        let mut data = vec![0.0_f32; x.data.len()];

        for g in 0..num_groups {
            let start = g * last_dim;
            let end = start + last_dim;
            let group = &x.data[start..end];

            let mean = group.iter().sum::<f32>() / last_dim as f32;
            let var = group.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / last_dim as f32;
            let std_inv = 1.0 / (var + self.eps).sqrt();

            for i in 0..last_dim {
                data[start + i] = self.gamma.data[i] * (group[i] - mean) * std_inv + self.beta.data[i];
            }
        }

        Tensor::new(data, x.shape.clone())
    }

    /// Backward pass for LayerNorm.
    /// d_output: same shape as forward output
    /// x: cached input from forward
    /// Returns: (d_input, d_gamma, d_beta)
    pub fn backward(&self, d_output: &Tensor, x: &Tensor) -> (Tensor, Tensor, Tensor) {
        let last_dim = *x.shape.last().expect("empty shape");
        let num_groups = x.data.len() / last_dim;
        let n = last_dim as f32;

        let mut d_input = vec![0.0_f32; x.data.len()];
        let mut d_gamma = vec![0.0_f32; last_dim];
        let mut d_beta = vec![0.0_f32; last_dim];

        for g in 0..num_groups {
            let start = g * last_dim;
            let group = &x.data[start..start + last_dim];
            let dy = &d_output.data[start..start + last_dim];

            let mean = group.iter().sum::<f32>() / n;
            let var = group.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n;
            let std_inv = 1.0 / (var + self.eps).sqrt();

            // normalized values: x_hat[i] = (x[i] - mean) * std_inv
            let x_hat: Vec<f32> = group.iter().map(|&v| (v - mean) * std_inv).collect();

            // d_gamma and d_beta accumulate over groups
            for i in 0..last_dim {
                d_gamma[i] += dy[i] * x_hat[i];
                d_beta[i] += dy[i];
            }

            // d_input: apply chain rule through normalization
            // dy_hat[i] = dy[i] * gamma[i]
            let dy_hat: Vec<f32> = (0..last_dim).map(|i| dy[i] * self.gamma.data[i]).collect();

            let mean_dy_hat = dy_hat.iter().sum::<f32>() / n;
            let mean_dy_hat_x_hat: f32 = dy_hat.iter().zip(x_hat.iter()).map(|(&d, &x)| d * x).sum::<f32>() / n;

            for i in 0..last_dim {
                d_input[start + i] = std_inv * (dy_hat[i] - mean_dy_hat - x_hat[i] * mean_dy_hat_x_hat);
            }
        }

        (
            Tensor::new(d_input, x.shape.clone()),
            Tensor::new(d_gamma, vec![last_dim]),
            Tensor::new(d_beta, vec![last_dim]),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_approx(a: &[f32], b: &[f32], eps: f32) {
        assert_eq!(a.len(), b.len());
        for (i, (&va, &vb)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (va - vb).abs() < eps,
                "index {}: {} vs {} (diff {})",
                i,
                va,
                vb,
                (va - vb).abs()
            );
        }
    }

    #[test]
    fn test_new() {
        let ln = LayerNorm::new(4);
        assert_eq!(ln.gamma.shape, vec![4]);
        assert_eq!(ln.gamma.data, vec![1.0; 4]);
        assert_eq!(ln.beta.shape, vec![4]);
        assert_eq!(ln.beta.data, vec![0.0; 4]);
    }

    #[test]
    fn test_forward_zero_mean_unit_var() {
        // With default gamma=1, beta=0, output should have ~zero mean and ~unit variance
        let ln = LayerNorm::new(4);
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let y = ln.forward(&x);
        assert_eq!(y.shape, vec![1, 4]);

        // mean should be ~0
        let mean: f32 = y.data.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5, "mean: {}", mean);

        // variance should be ~1
        let var: f32 = y.data.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / 4.0;
        assert!((var - 1.0).abs() < 0.01, "var: {}", var);
    }

    #[test]
    fn test_forward_known_values() {
        // x = [1, 2, 3, 4], mean=2.5, var=1.25
        // normalized = (x - 2.5) / sqrt(1.25 + 1e-5)
        //            = [-1.5, -0.5, 0.5, 1.5] / 1.11803...
        //            ≈ [-1.3416, -0.4472, 0.4472, 1.3416]
        let ln = LayerNorm::new(4);
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let y = ln.forward(&x);
        let std = (1.25_f32 + 1e-5).sqrt();
        let expected: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0]
            .iter()
            .map(|&v| (v - 2.5) / std)
            .collect();
        assert_approx(&y.data, &expected, 1e-4);
    }

    #[test]
    fn test_forward_batch() {
        // [2, 3]: two independent rows
        let ln = LayerNorm::new(3);
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0], vec![2, 3]);
        let y = ln.forward(&x);
        assert_eq!(y.shape, vec![2, 3]);

        // each row should have ~zero mean
        let mean0: f32 = y.data[0..3].iter().sum::<f32>() / 3.0;
        let mean1: f32 = y.data[3..6].iter().sum::<f32>() / 3.0;
        assert!(mean0.abs() < 1e-5);
        assert!(mean1.abs() < 1e-5);

        // both rows have same relative pattern => same normalized output
        assert_approx(&y.data[0..3], &y.data[3..6], 1e-4);
    }

    #[test]
    fn test_forward_with_gamma_beta() {
        let mut ln = LayerNorm::new(3);
        ln.gamma = Tensor::new(vec![2.0, 2.0, 2.0], vec![3]);
        ln.beta = Tensor::new(vec![1.0, 1.0, 1.0], vec![3]);

        // x = [1, 2, 3], mean=2, var=2/3
        // normalized = [-1, 0, 1] / sqrt(2/3 + eps)
        // output = 2 * normalized + 1
        let x = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let y = ln.forward(&x);

        // middle element: normalized=0, so output = 2*0 + 1 = 1
        assert!((y.data[1] - 1.0).abs() < 1e-4);
        // output should be symmetric around 1.0
        assert!((y.data[0] + y.data[2] - 2.0).abs() < 1e-4);
    }

    #[test]
    fn test_forward_3d() {
        let ln = LayerNorm::new(2);
        let x = Tensor::new(vec![1.0, 3.0, 2.0, 4.0, 5.0, 7.0, 6.0, 8.0], vec![2, 2, 2]);
        let y = ln.forward(&x);
        assert_eq!(y.shape, vec![2, 2, 2]);

        // each pair should normalize to [-1, 1] (with gamma=1, beta=0)
        for i in 0..4 {
            let sum: f32 = y.data[i * 2..(i + 1) * 2].iter().sum();
            assert!(sum.abs() < 1e-5);
        }
    }

    // ==================== backward ====================

    #[test]
    fn test_backward_shapes() {
        let ln = LayerNorm::new(3);
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let d_out = Tensor::new(vec![1.0; 6], vec![2, 3]);
        let (d_input, d_gamma, d_beta) = ln.backward(&d_out, &x);
        assert_eq!(d_input.shape, vec![2, 3]);
        assert_eq!(d_gamma.shape, vec![3]);
        assert_eq!(d_beta.shape, vec![3]);
    }

    #[test]
    fn test_backward_numerical_d_input() {
        let mut ln = LayerNorm::new(3);
        ln.gamma = Tensor::new(vec![1.0, 2.0, 0.5], vec![3]);
        ln.beta = Tensor::new(vec![0.1, -0.1, 0.2], vec![3]);
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let d_out = Tensor::new(vec![0.1, -0.2, 0.3, -0.1, 0.5, -0.4], vec![2, 3]);
        let (d_input, _, _) = ln.backward(&d_out, &x);

        let eps = 1e-4;
        for i in 0..6 {
            let mut x_plus = x.clone();
            x_plus.data[i] += eps;
            let mut x_minus = x.clone();
            x_minus.data[i] -= eps;
            let y_plus = ln.forward(&x_plus);
            let y_minus = ln.forward(&x_minus);
            let mut numerical = 0.0;
            for j in 0..6 {
                numerical += (y_plus.data[j] - y_minus.data[j]) / (2.0 * eps) * d_out.data[j];
            }
            assert!(
                (d_input.data[i] - numerical).abs() < 1e-2,
                "d_input[{}]: analytical {} vs numerical {}", i, d_input.data[i], numerical
            );
        }
    }

    #[test]
    fn test_backward_numerical_d_gamma() {
        let ln = LayerNorm::new(3);
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let d_out = Tensor::new(vec![0.1, -0.2, 0.3, -0.1, 0.5, -0.4], vec![2, 3]);
        let (_, d_gamma, _) = ln.backward(&d_out, &x);

        let eps = 1e-4;
        for i in 0..3 {
            let mut ln_plus = LayerNorm::new(3);
            ln_plus.gamma.data[i] += eps;
            let mut ln_minus = LayerNorm::new(3);
            ln_minus.gamma.data[i] -= eps;
            let y_plus = ln_plus.forward(&x);
            let y_minus = ln_minus.forward(&x);
            let mut numerical = 0.0;
            for j in 0..6 {
                numerical += (y_plus.data[j] - y_minus.data[j]) / (2.0 * eps) * d_out.data[j];
            }
            assert!(
                (d_gamma.data[i] - numerical).abs() < 1e-2,
                "d_gamma[{}]: analytical {} vs numerical {}", i, d_gamma.data[i], numerical
            );
        }
    }
}
