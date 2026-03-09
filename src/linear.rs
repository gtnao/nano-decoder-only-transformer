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
}
