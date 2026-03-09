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
}
