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
}
