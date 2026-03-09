use crate::tensor::Tensor;

/// ReLU: max(0, x)
pub fn relu(x: &Tensor) -> Tensor {
    let data = x.data.iter().map(|&v| v.max(0.0)).collect();
    Tensor::new(data, x.shape.clone())
}

/// GELU (tanh approximation):
/// 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
pub fn gelu(x: &Tensor) -> Tensor {
    let coeff = (2.0_f32 / std::f32::consts::PI).sqrt();
    let data = x
        .data
        .iter()
        .map(|&v| {
            let inner = coeff * (v + 0.044715 * v * v * v);
            0.5 * v * (1.0 + inner.tanh())
        })
        .collect();
    Tensor::new(data, x.shape.clone())
}

/// ReLU backward: d_input = d_output * (x > 0 ? 1 : 0)
pub fn relu_backward(d_output: &Tensor, x: &Tensor) -> Tensor {
    let data = d_output
        .data
        .iter()
        .zip(x.data.iter())
        .map(|(&d, &v)| if v > 0.0 { d } else { 0.0 })
        .collect();
    Tensor::new(data, x.shape.clone())
}

/// GELU backward (tanh approximation derivative)
pub fn gelu_backward(d_output: &Tensor, x: &Tensor) -> Tensor {
    let coeff = (2.0_f32 / std::f32::consts::PI).sqrt();
    let data = d_output
        .data
        .iter()
        .zip(x.data.iter())
        .map(|(&d, &v)| {
            let inner = coeff * (v + 0.044715 * v * v * v);
            let tanh_inner = inner.tanh();
            let sech2 = 1.0 - tanh_inner * tanh_inner;
            let d_inner = coeff * (1.0 + 3.0 * 0.044715 * v * v);
            // d(gelu)/dx = 0.5 * (1 + tanh) + 0.5 * x * sech^2 * d_inner
            let dgelu = 0.5 * (1.0 + tanh_inner) + 0.5 * v * sech2 * d_inner;
            d * dgelu
        })
        .collect();
    Tensor::new(data, x.shape.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== ReLU ====================

    #[test]
    fn test_relu_positive() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let y = relu(&x);
        assert_eq!(y.data, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_relu_negative() {
        let x = Tensor::new(vec![-1.0, -2.0, -3.0], vec![3]);
        let y = relu(&x);
        assert_eq!(y.data, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_relu_mixed() {
        let x = Tensor::new(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5]);
        let y = relu(&x);
        assert_eq!(y.data, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_relu_preserves_shape() {
        let x = Tensor::new(vec![1.0, -1.0, 2.0, -2.0], vec![2, 2]);
        let y = relu(&x);
        assert_eq!(y.shape, vec![2, 2]);
    }

    // ==================== GELU ====================

    #[test]
    fn test_gelu_zero() {
        let x = Tensor::new(vec![0.0], vec![1]);
        let y = gelu(&x);
        assert!(y.data[0].abs() < 1e-5);
    }

    #[test]
    fn test_gelu_positive_large() {
        // For large positive x, GELU(x) ≈ x
        let x = Tensor::new(vec![3.0], vec![1]);
        let y = gelu(&x);
        assert!((y.data[0] - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_gelu_negative_large() {
        // For large negative x, GELU(x) ≈ 0
        let x = Tensor::new(vec![-3.0], vec![1]);
        let y = gelu(&x);
        assert!(y.data[0].abs() < 0.01);
    }

    #[test]
    fn test_gelu_known_values() {
        // GELU(1.0) ≈ 0.8412
        // GELU(-1.0) ≈ -0.1588
        let x = Tensor::new(vec![1.0, -1.0], vec![2]);
        let y = gelu(&x);
        assert!((y.data[0] - 0.8412).abs() < 0.001);
        assert!((y.data[1] - (-0.1588)).abs() < 0.001);
    }

    #[test]
    fn test_gelu_preserves_shape() {
        let x = Tensor::new(vec![1.0, -1.0, 2.0, -2.0], vec![2, 2]);
        let y = gelu(&x);
        assert_eq!(y.shape, vec![2, 2]);
    }

    // ==================== backward ====================

    #[test]
    fn test_relu_backward_positive() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let d_out = Tensor::new(vec![1.0, 1.0, 1.0], vec![3]);
        let d_in = relu_backward(&d_out, &x);
        assert_eq!(d_in.data, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_relu_backward_negative() {
        let x = Tensor::new(vec![-1.0, -2.0, -3.0], vec![3]);
        let d_out = Tensor::new(vec![1.0, 1.0, 1.0], vec![3]);
        let d_in = relu_backward(&d_out, &x);
        assert_eq!(d_in.data, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_relu_backward_mixed() {
        let x = Tensor::new(vec![-1.0, 2.0, -3.0, 4.0], vec![4]);
        let d_out = Tensor::new(vec![0.5, 0.5, 0.5, 0.5], vec![4]);
        let d_in = relu_backward(&d_out, &x);
        assert_eq!(d_in.data, vec![0.0, 0.5, 0.0, 0.5]);
    }

    #[test]
    fn test_gelu_backward_numerical() {
        let x = Tensor::new(vec![-1.0, 0.0, 0.5, 1.0, 2.0], vec![5]);
        let d_out = Tensor::new(vec![1.0; 5], vec![5]);
        let d_in = gelu_backward(&d_out, &x);

        let eps = 1e-4;
        for i in 0..5 {
            let mut x_plus = x.clone();
            x_plus.data[i] += eps;
            let mut x_minus = x.clone();
            x_minus.data[i] -= eps;
            let numerical = (gelu(&x_plus).data[i] - gelu(&x_minus).data[i]) / (2.0 * eps);
            assert!(
                (d_in.data[i] - numerical).abs() < 1e-2,
                "index {}: analytical {} vs numerical {}", i, d_in.data[i], numerical
            );
        }
    }

    #[test]
    fn test_gelu_backward_scales_with_d_output() {
        let x = Tensor::new(vec![1.0, 2.0], vec![2]);
        let d_out1 = Tensor::new(vec![1.0, 1.0], vec![2]);
        let d_out2 = Tensor::new(vec![2.0, 2.0], vec![2]);
        let d_in1 = gelu_backward(&d_out1, &x);
        let d_in2 = gelu_backward(&d_out2, &x);
        for (a, b) in d_in1.data.iter().zip(d_in2.data.iter()) {
            assert!((b - 2.0 * a).abs() < 1e-5);
        }
    }
}
