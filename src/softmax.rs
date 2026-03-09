use crate::tensor::Tensor;

/// Softmax along the last axis (with numerical stability).
/// softmax(x_i) = exp(x_i - max(x)) / Σexp(x_j - max(x))
pub fn softmax(x: &Tensor) -> Tensor {
    let last_dim = *x.shape.last().expect("empty shape");
    let num_groups = x.data.len() / last_dim;
    let mut data = vec![0.0_f32; x.data.len()];

    for g in 0..num_groups {
        let start = g * last_dim;
        let end = start + last_dim;
        let group = &x.data[start..end];

        // subtract max for numerical stability
        let max_val = group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = group.iter().map(|&v| (v - max_val).exp()).collect();
        let sum: f32 = exps.iter().sum();

        for (i, &e) in exps.iter().enumerate() {
            data[start + i] = e / sum;
        }
    }

    Tensor::new(data, x.shape.clone())
}

/// Softmax backward along the last axis.
/// d_output: same shape as softmax output
/// softmax_output: cached softmax output from forward
/// d_input[i] = s[i] * (d_output[i] - Σ(d_output[j] * s[j]))
pub fn softmax_backward(d_output: &Tensor, softmax_output: &Tensor) -> Tensor {
    let last_dim = *softmax_output.shape.last().expect("empty shape");
    let num_groups = softmax_output.data.len() / last_dim;
    let mut data = vec![0.0_f32; softmax_output.data.len()];

    for g in 0..num_groups {
        let start = g * last_dim;
        let s = &softmax_output.data[start..start + last_dim];
        let dy = &d_output.data[start..start + last_dim];

        // dot = Σ(dy[j] * s[j])
        let dot: f32 = s.iter().zip(dy.iter()).map(|(&si, &di)| si * di).sum();

        // d_input[i] = s[i] * (dy[i] - dot)
        for i in 0..last_dim {
            data[start + i] = s[i] * (dy[i] - dot);
        }
    }

    Tensor::new(data, softmax_output.shape.clone())
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
    fn test_softmax_1d() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let y = softmax(&x);
        assert_eq!(y.shape, vec![3]);
        // sum should be 1.0
        let sum: f32 = y.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        // values should be monotonically increasing
        assert!(y.data[0] < y.data[1]);
        assert!(y.data[1] < y.data[2]);
    }

    #[test]
    fn test_softmax_uniform() {
        // equal inputs => equal outputs => 1/n each
        let x = Tensor::new(vec![1.0, 1.0, 1.0, 1.0], vec![4]);
        let y = softmax(&x);
        assert_approx(&y.data, &[0.25, 0.25, 0.25, 0.25], 1e-5);
    }

    #[test]
    fn test_softmax_2d_row_wise() {
        // each row is independent
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0], vec![2, 3]);
        let y = softmax(&x);
        assert_eq!(y.shape, vec![2, 3]);
        // row 0 sum
        let sum0: f32 = y.data[0..3].iter().sum();
        assert!((sum0 - 1.0).abs() < 1e-5);
        // row 1 sum
        let sum1: f32 = y.data[3..6].iter().sum();
        assert!((sum1 - 1.0).abs() < 1e-5);
        // both rows should be identical
        assert_approx(&y.data[0..3], &y.data[3..6], 1e-5);
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // large values should not cause overflow
        let x = Tensor::new(vec![1000.0, 1001.0, 1002.0], vec![3]);
        let y = softmax(&x);
        let sum: f32 = y.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(y.data.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_softmax_3d() {
        // [2, 2, 3] => softmax along last axis (size 3)
        let x = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0,
            ],
            vec![2, 2, 3],
        );
        let y = softmax(&x);
        assert_eq!(y.shape, vec![2, 2, 3]);
        // each group of 3 sums to 1
        for i in 0..4 {
            let sum: f32 = y.data[i * 3..(i + 1) * 3].iter().sum();
            assert!((sum - 1.0).abs() < 1e-5);
        }
    }

    // ==================== backward ====================

    #[test]
    fn test_softmax_backward_shape() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let s = softmax(&x);
        let d_out = Tensor::new(vec![1.0, 0.0, 0.0], vec![3]);
        let d_in = softmax_backward(&d_out, &s);
        assert_eq!(d_in.shape, vec![3]);
    }

    #[test]
    fn test_softmax_backward_numerical() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let s = softmax(&x);
        let d_out = Tensor::new(vec![1.0, -0.5, 0.5], vec![3]);
        let d_in = softmax_backward(&d_out, &s);

        let eps = 1e-4;
        for i in 0..3 {
            let mut x_plus = x.clone();
            x_plus.data[i] += eps;
            let mut x_minus = x.clone();
            x_minus.data[i] -= eps;
            let s_plus = softmax(&x_plus);
            let s_minus = softmax(&x_minus);
            let mut numerical = 0.0;
            for j in 0..3 {
                numerical += (s_plus.data[j] - s_minus.data[j]) / (2.0 * eps) * d_out.data[j];
            }
            assert!(
                (d_in.data[i] - numerical).abs() < 1e-2,
                "index {}: analytical {} vs numerical {}", i, d_in.data[i], numerical
            );
        }
    }

    #[test]
    fn test_softmax_backward_2d_numerical() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let s = softmax(&x);
        let d_out = Tensor::new(vec![0.1, 0.2, 0.3, -0.1, 0.5, -0.4], vec![2, 3]);
        let d_in = softmax_backward(&d_out, &s);

        let eps = 1e-4;
        for i in 0..6 {
            let mut x_plus = x.clone();
            x_plus.data[i] += eps;
            let mut x_minus = x.clone();
            x_minus.data[i] -= eps;
            let s_plus = softmax(&x_plus);
            let s_minus = softmax(&x_minus);
            let mut numerical = 0.0;
            for j in 0..6 {
                numerical += (s_plus.data[j] - s_minus.data[j]) / (2.0 * eps) * d_out.data[j];
            }
            assert!(
                (d_in.data[i] - numerical).abs() < 1e-2,
                "index {}: analytical {} vs numerical {}", i, d_in.data[i], numerical
            );
        }
    }
}
