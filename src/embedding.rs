use crate::tensor::Tensor;

pub struct Embedding {
    pub weight: Tensor, // [vocab_size, embed_dim]
}

impl Embedding {
    /// Create with explicit weight (for testing)
    pub fn new(weight: Tensor) -> Self {
        assert_eq!(weight.shape.len(), 2, "weight must be 2D");
        Self { weight }
    }

    /// Create with random weights
    pub fn rand(vocab_size: usize, embed_dim: usize) -> Self {
        use rand::RngExt;
        let mut rng = rand::rng();
        let data: Vec<f32> = (0..vocab_size * embed_dim)
            .map(|_| rng.random_range(-0.02..0.02))
            .collect();
        Self {
            weight: Tensor::new(data, vec![vocab_size, embed_dim]),
        }
    }

    /// Look up embeddings for given token IDs.
    /// input: &[usize] (token IDs) -> output: Tensor [len, embed_dim]
    pub fn forward(&self, token_ids: &[usize]) -> Tensor {
        let vocab_size = self.weight.shape[0];
        let embed_dim = self.weight.shape[1];
        let mut data = Vec::with_capacity(token_ids.len() * embed_dim);
        for &id in token_ids {
            assert!(id < vocab_size, "token id {} out of range (vocab_size={})", id, vocab_size);
            let start = id * embed_dim;
            data.extend_from_slice(&self.weight.data[start..start + embed_dim]);
        }
        Tensor::new(data, vec![token_ids.len(), embed_dim])
    }

    /// Backward pass for Embedding.
    /// d_output: [seq_len, embed_dim]
    /// token_ids: cached from forward
    /// Returns: d_weight [vocab_size, embed_dim]
    pub fn backward(&self, d_output: &Tensor, token_ids: &[usize]) -> Tensor {
        let vocab_size = self.weight.shape[0];
        let embed_dim = self.weight.shape[1];
        assert_eq!(d_output.shape, vec![token_ids.len(), embed_dim]);

        let mut d_weight = vec![0.0_f32; vocab_size * embed_dim];
        for (t, &id) in token_ids.iter().enumerate() {
            let src = t * embed_dim;
            let dst = id * embed_dim;
            for i in 0..embed_dim {
                d_weight[dst + i] += d_output.data[src + i];
            }
        }
        Tensor::new(d_weight, vec![vocab_size, embed_dim])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let w = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
        let emb = Embedding::new(w);
        assert_eq!(emb.weight.shape, vec![3, 2]);
    }

    #[test]
    #[should_panic]
    fn test_new_not_2d() {
        let w = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        Embedding::new(w);
    }

    #[test]
    fn test_forward_single() {
        // vocab=3, embed_dim=2
        // [[1, 2],   <- id 0
        //  [3, 4],   <- id 1
        //  [5, 6]]   <- id 2
        let w = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
        let emb = Embedding::new(w);

        let y = emb.forward(&[1]);
        assert_eq!(y.shape, vec![1, 2]);
        assert_eq!(y.data, vec![3.0, 4.0]);
    }

    #[test]
    fn test_forward_multiple() {
        let w = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
        let emb = Embedding::new(w);

        let y = emb.forward(&[2, 0, 1]);
        assert_eq!(y.shape, vec![3, 2]);
        assert_eq!(y.data, vec![5.0, 6.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_forward_duplicate_ids() {
        let w = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let emb = Embedding::new(w);

        let y = emb.forward(&[0, 0, 1, 1]);
        assert_eq!(y.shape, vec![4, 2]);
        assert_eq!(y.data, vec![1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0]);
    }

    #[test]
    #[should_panic]
    fn test_forward_out_of_range() {
        let w = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let emb = Embedding::new(w);
        emb.forward(&[5]); // vocab_size=2, id=5 is out of range
    }

    #[test]
    fn test_rand_shape() {
        let emb = Embedding::rand(100, 64);
        assert_eq!(emb.weight.shape, vec![100, 64]);
    }

    // ==================== backward ====================

    #[test]
    fn test_backward_shape() {
        // vocab=3, embed_dim=2, seq_len=2
        let w = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
        let emb = Embedding::new(w);
        let d_output = Tensor::new(vec![0.1, 0.2, 0.3, 0.4], vec![2, 2]);
        let d_weight = emb.backward(&d_output, &[0, 2]);
        assert_eq!(d_weight.shape, vec![3, 2]);
    }

    #[test]
    fn test_backward_simple() {
        // vocab=3, embed_dim=2
        // token_ids = [1], d_output = [[0.5, -0.3]]
        // d_weight should have row 1 = [0.5, -0.3], others zero
        let w = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
        let emb = Embedding::new(w);
        let d_output = Tensor::new(vec![0.5, -0.3], vec![1, 2]);
        let d_weight = emb.backward(&d_output, &[1]);
        assert_eq!(d_weight.data, vec![0.0, 0.0, 0.5, -0.3, 0.0, 0.0]);
    }

    #[test]
    fn test_backward_multiple() {
        // vocab=3, embed_dim=2
        // token_ids = [2, 0], d_output = [[0.1, 0.2], [0.3, 0.4]]
        // d_weight: row 0 = [0.3, 0.4], row 1 = [0, 0], row 2 = [0.1, 0.2]
        let w = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
        let emb = Embedding::new(w);
        let d_output = Tensor::new(vec![0.1, 0.2, 0.3, 0.4], vec![2, 2]);
        let d_weight = emb.backward(&d_output, &[2, 0]);
        assert_eq!(d_weight.data, vec![0.3, 0.4, 0.0, 0.0, 0.1, 0.2]);
    }

    #[test]
    fn test_backward_duplicate_ids() {
        // vocab=2, embed_dim=2
        // token_ids = [0, 0, 1], d_output = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        // d_weight: row 0 = [0.1+0.3, 0.2+0.4] = [0.4, 0.6], row 1 = [0.5, 0.6]
        let w = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let emb = Embedding::new(w);
        let d_output = Tensor::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], vec![3, 2]);
        let d_weight = emb.backward(&d_output, &[0, 0, 1]);

        fn assert_approx(a: &[f32], b: &[f32], eps: f32) {
            assert_eq!(a.len(), b.len());
            for (i, (&va, &vb)) in a.iter().zip(b.iter()).enumerate() {
                assert!(
                    (va - vb).abs() < eps,
                    "index {}: {} vs {} (diff {})",
                    i, va, vb, (va - vb).abs()
                );
            }
        }
        assert_approx(&d_weight.data, &[0.4, 0.6, 0.5, 0.6], 1e-6);
    }

    #[test]
    fn test_backward_numerical() {
        // Verify with numerical gradients
        let w = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
        let emb = Embedding::new(w.clone());
        let token_ids: &[usize] = &[2, 0, 1];
        let d_output = Tensor::new(vec![0.1, -0.2, 0.3, 0.4, -0.5, 0.6], vec![3, 2]);

        let d_weight = emb.backward(&d_output, token_ids);

        let eps = 1e-4;
        for i in 0..6 {
            let mut w_plus = w.clone();
            w_plus.data[i] += eps;
            let emb_plus = Embedding::new(w_plus);
            let y_plus = emb_plus.forward(token_ids);

            let mut w_minus = w.clone();
            w_minus.data[i] -= eps;
            let emb_minus = Embedding::new(w_minus);
            let y_minus = emb_minus.forward(token_ids);

            let mut numerical = 0.0_f32;
            for j in 0..6 {
                numerical += (y_plus.data[j] - y_minus.data[j]) / (2.0 * eps) * d_output.data[j];
            }
            assert!(
                (d_weight.data[i] - numerical).abs() < 1e-2,
                "d_weight[{}]: analytical {} vs numerical {}", i, d_weight.data[i], numerical
            );
        }
    }
}
