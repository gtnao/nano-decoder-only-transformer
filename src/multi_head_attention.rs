use crate::attention::{causal_mask, scaled_dot_product_attention};
use crate::linear::Linear;
use crate::tensor::Tensor;

pub struct MultiHeadAttention {
    pub wq: Linear, // [d_model, d_model]
    pub wk: Linear, // [d_model, d_model]
    pub wv: Linear, // [d_model, d_model]
    pub wo: Linear, // [d_model, d_model]
    pub n_heads: usize,
}

impl MultiHeadAttention {
    pub fn new(wq: Linear, wk: Linear, wv: Linear, wo: Linear, n_heads: usize) -> Self {
        let d_model = wq.weight.shape[0];
        assert_eq!(d_model % n_heads, 0, "d_model must be divisible by n_heads");
        Self { wq, wk, wv, wo, n_heads }
    }

    pub fn rand(d_model: usize, n_heads: usize) -> Self {
        assert_eq!(d_model % n_heads, 0, "d_model must be divisible by n_heads");
        Self {
            wq: Linear::rand(d_model, d_model),
            wk: Linear::rand(d_model, d_model),
            wv: Linear::rand(d_model, d_model),
            wo: Linear::rand(d_model, d_model),
            n_heads,
        }
    }

    /// Self-attention: Q, K, V are all projected from x.
    /// x: [seq_len, d_model]
    /// Returns: [seq_len, d_model]
    pub fn forward(&self, x: &Tensor, use_causal_mask: bool) -> Tensor {
        let seq_len = x.shape[0];
        let d_model = x.shape[1];
        let d_k = d_model / self.n_heads;

        // Project: [seq_len, d_model]
        let q_full = self.wq.forward(x);
        let k_full = self.wk.forward(x);
        let v_full = self.wv.forward(x);

        let mask = if use_causal_mask {
            Some(causal_mask(seq_len))
        } else {
            None
        };

        // Split heads, apply attention, collect results
        let mut head_outputs = vec![0.0_f32; seq_len * d_model];

        for h in 0..self.n_heads {
            let offset = h * d_k;

            // Extract head h: [seq_len, d_k]
            let mut q_h = vec![0.0_f32; seq_len * d_k];
            let mut k_h = vec![0.0_f32; seq_len * d_k];
            let mut v_h = vec![0.0_f32; seq_len * d_k];
            for s in 0..seq_len {
                for d in 0..d_k {
                    q_h[s * d_k + d] = q_full.data[s * d_model + offset + d];
                    k_h[s * d_k + d] = k_full.data[s * d_model + offset + d];
                    v_h[s * d_k + d] = v_full.data[s * d_model + offset + d];
                }
            }

            let q_head = Tensor::new(q_h, vec![seq_len, d_k]);
            let k_head = Tensor::new(k_h, vec![seq_len, d_k]);
            let v_head = Tensor::new(v_h, vec![seq_len, d_k]);

            let attn_out = scaled_dot_product_attention(
                &q_head,
                &k_head,
                &v_head,
                mask.as_ref(),
            );

            // Write back into concatenated output
            for s in 0..seq_len {
                for d in 0..d_k {
                    head_outputs[s * d_model + offset + d] = attn_out.data[s * d_k + d];
                }
            }
        }

        // Concatenated: [seq_len, d_model] -> project through Wo
        let concat = Tensor::new(head_outputs, vec![seq_len, d_model]);
        self.wo.forward(&concat)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rand_shapes() {
        let mha = MultiHeadAttention::rand(8, 2);
        assert_eq!(mha.wq.weight.shape, vec![8, 8]);
        assert_eq!(mha.wk.weight.shape, vec![8, 8]);
        assert_eq!(mha.wv.weight.shape, vec![8, 8]);
        assert_eq!(mha.wo.weight.shape, vec![8, 8]);
        assert_eq!(mha.n_heads, 2);
    }

    #[test]
    fn test_forward_shape() {
        let mha = MultiHeadAttention::rand(8, 2);
        let x = Tensor::new(vec![0.1; 3 * 8], vec![3, 8]);
        let out = mha.forward(&x, false);
        assert_eq!(out.shape, vec![3, 8]);
    }

    #[test]
    fn test_forward_causal_shape() {
        let mha = MultiHeadAttention::rand(8, 2);
        let x = Tensor::new(vec![0.1; 3 * 8], vec![3, 8]);
        let out = mha.forward(&x, true);
        assert_eq!(out.shape, vec![3, 8]);
    }

    #[test]
    fn test_forward_single_head_equals_attention() {
        // With 1 head and identity projections (W=I, b=0),
        // MHA should reduce to plain scaled dot-product attention + Wo projection.
        let d = 4;
        let eye_data: Vec<f32> = (0..d)
            .flat_map(|r| (0..d).map(move |c| if r == c { 1.0 } else { 0.0 }))
            .collect();
        let zeros = vec![0.0; d];

        let wq = Linear::new(
            Tensor::new(eye_data.clone(), vec![d, d]),
            Tensor::new(zeros.clone(), vec![d]),
        );
        let wk = Linear::new(
            Tensor::new(eye_data.clone(), vec![d, d]),
            Tensor::new(zeros.clone(), vec![d]),
        );
        let wv = Linear::new(
            Tensor::new(eye_data.clone(), vec![d, d]),
            Tensor::new(zeros.clone(), vec![d]),
        );
        let wo = Linear::new(
            Tensor::new(eye_data.clone(), vec![d, d]),
            Tensor::new(zeros.clone(), vec![d]),
        );

        let mha = MultiHeadAttention::new(wq, wk, wv, wo, 1);
        let x = Tensor::new(vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);

        let mha_out = mha.forward(&x, false);

        // Compare with direct attention
        let direct_out = scaled_dot_product_attention(&x, &x, &x, None);

        for (a, b) in mha_out.data.iter().zip(direct_out.data.iter()) {
            assert!((a - b).abs() < 1e-4, "{} vs {}", a, b);
        }
    }

    #[test]
    fn test_forward_deterministic() {
        let mha = MultiHeadAttention::rand(8, 2);
        let x = Tensor::new(vec![0.5; 3 * 8], vec![3, 8]);
        let out1 = mha.forward(&x, true);
        let out2 = mha.forward(&x, true);
        assert_eq!(out1.data, out2.data);
    }

    #[test]
    #[should_panic]
    fn test_d_model_not_divisible_by_n_heads() {
        MultiHeadAttention::rand(7, 2);
    }
}
