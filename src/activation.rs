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
}
