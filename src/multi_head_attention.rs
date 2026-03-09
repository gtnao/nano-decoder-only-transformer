use crate::attention::{causal_mask, scaled_dot_product_attention, scaled_dot_product_attention_backward};
use crate::linear::Linear;
use crate::tensor::Tensor;
use rayon::prelude::*;

pub struct MHAGradients {
    pub d_x: Tensor,
    pub d_wq_weight: Tensor,
    pub d_wq_bias: Tensor,
    pub d_wk_weight: Tensor,
    pub d_wk_bias: Tensor,
    pub d_wv_weight: Tensor,
    pub d_wv_bias: Tensor,
    pub d_wo_weight: Tensor,
    pub d_wo_bias: Tensor,
}

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

        // Split heads, apply attention in parallel, collect results
        let head_results: Vec<(usize, Tensor)> = (0..self.n_heads)
            .into_par_iter()
            .map(|h| {
                let offset = h * d_k;
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
                let attn_out = scaled_dot_product_attention(&q_head, &k_head, &v_head, mask.as_ref());
                (offset, attn_out)
            })
            .collect();

        let mut head_outputs = vec![0.0_f32; seq_len * d_model];
        for (offset, attn_out) in &head_results {
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

    /// Backward pass for Multi-Head Attention.
    /// d_output: [seq_len, d_model]
    /// x: cached input from forward
    /// Returns MHAGradients with d_x and all parameter gradients.
    pub fn backward(&self, d_output: &Tensor, x: &Tensor, use_causal_mask: bool) -> MHAGradients {
        let seq_len = x.shape[0];
        let d_model = x.shape[1];
        let d_k = d_model / self.n_heads;

        // Recompute forward intermediates
        let q_full = self.wq.forward(x);
        let k_full = self.wk.forward(x);
        let v_full = self.wv.forward(x);

        let mask = if use_causal_mask {
            Some(causal_mask(seq_len))
        } else {
            None
        };

        // Recompute per-head attention outputs in parallel
        let head_results: Vec<(usize, Tensor)> = (0..self.n_heads)
            .into_par_iter()
            .map(|h| {
                let offset = h * d_k;
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
                let attn_out = scaled_dot_product_attention(&q_head, &k_head, &v_head, mask.as_ref());
                (offset, attn_out)
            })
            .collect();

        let mut head_outputs = vec![0.0_f32; seq_len * d_model];
        for (offset, attn_out) in &head_results {
            for s in 0..seq_len {
                for d in 0..d_k {
                    head_outputs[s * d_model + offset + d] = attn_out.data[s * d_k + d];
                }
            }
        }
        let concat = Tensor::new(head_outputs, vec![seq_len, d_model]);

        // Backward through Wo
        let (d_concat, d_wo_weight, d_wo_bias) = self.wo.backward(d_output, &concat);

        // Backward through heads in parallel
        let head_grads: Vec<(usize, Tensor, Tensor, Tensor)> = (0..self.n_heads)
            .into_par_iter()
            .map(|h| {
                let offset = h * d_k;
                let mut d_out_h = vec![0.0_f32; seq_len * d_k];
                let mut q_h = vec![0.0_f32; seq_len * d_k];
                let mut k_h = vec![0.0_f32; seq_len * d_k];
                let mut v_h = vec![0.0_f32; seq_len * d_k];
                for s in 0..seq_len {
                    for d in 0..d_k {
                        d_out_h[s * d_k + d] = d_concat.data[s * d_model + offset + d];
                        q_h[s * d_k + d] = q_full.data[s * d_model + offset + d];
                        k_h[s * d_k + d] = k_full.data[s * d_model + offset + d];
                        v_h[s * d_k + d] = v_full.data[s * d_model + offset + d];
                    }
                }
                let d_out_head = Tensor::new(d_out_h, vec![seq_len, d_k]);
                let q_head = Tensor::new(q_h, vec![seq_len, d_k]);
                let k_head = Tensor::new(k_h, vec![seq_len, d_k]);
                let v_head = Tensor::new(v_h, vec![seq_len, d_k]);
                let (d_qh, d_kh, d_vh) = scaled_dot_product_attention_backward(
                    &d_out_head, &q_head, &k_head, &v_head, mask.as_ref(),
                );
                (offset, d_qh, d_kh, d_vh)
            })
            .collect();

        let mut d_q_full = vec![0.0_f32; seq_len * d_model];
        let mut d_k_full = vec![0.0_f32; seq_len * d_model];
        let mut d_v_full = vec![0.0_f32; seq_len * d_model];
        for (offset, d_qh, d_kh, d_vh) in &head_grads {
            for s in 0..seq_len {
                for d in 0..d_k {
                    d_q_full[s * d_model + offset + d] = d_qh.data[s * d_k + d];
                    d_k_full[s * d_model + offset + d] = d_kh.data[s * d_k + d];
                    d_v_full[s * d_model + offset + d] = d_vh.data[s * d_k + d];
                }
            }
        }

        let d_q_full = Tensor::new(d_q_full, vec![seq_len, d_model]);
        let d_k_full = Tensor::new(d_k_full, vec![seq_len, d_model]);
        let d_v_full = Tensor::new(d_v_full, vec![seq_len, d_model]);

        // Backward through Wq, Wk, Wv projections
        let (d_x_from_q, d_wq_weight, d_wq_bias) = self.wq.backward(&d_q_full, x);
        let (d_x_from_k, d_wk_weight, d_wk_bias) = self.wk.backward(&d_k_full, x);
        let (d_x_from_v, d_wv_weight, d_wv_bias) = self.wv.backward(&d_v_full, x);

        // d_x = sum of contributions from Q, K, V projections
        let d_x = d_x_from_q.add(&d_x_from_k).add(&d_x_from_v);

        MHAGradients {
            d_x,
            d_wq_weight,
            d_wq_bias,
            d_wk_weight,
            d_wk_bias,
            d_wv_weight,
            d_wv_bias,
            d_wo_weight,
            d_wo_bias,
        }
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

    // ==================== backward ====================

    #[test]
    fn test_backward_shapes() {
        let mha = MultiHeadAttention::rand(4, 2);
        let x = Tensor::new(vec![0.1; 3 * 4], vec![3, 4]);
        let d_out = Tensor::new(vec![0.1; 3 * 4], vec![3, 4]);
        let grads = mha.backward(&d_out, &x, true);
        assert_eq!(grads.d_x.shape, vec![3, 4]);
        assert_eq!(grads.d_wq_weight.shape, vec![4, 4]);
        assert_eq!(grads.d_wq_bias.shape, vec![4]);
        assert_eq!(grads.d_wo_weight.shape, vec![4, 4]);
        assert_eq!(grads.d_wo_bias.shape, vec![4]);
    }

    #[test]
    fn test_backward_numerical_d_x() {
        let mha = MultiHeadAttention::rand(4, 2);
        let x = Tensor::new(
            vec![0.1, -0.2, 0.3, 0.4, 0.5, 0.1, -0.3, 0.2],
            vec![2, 4],
        );
        let d_out = Tensor::new(
            vec![0.5, -0.3, 0.1, 0.2, -0.1, 0.4, -0.2, 0.6],
            vec![2, 4],
        );
        let grads = mha.backward(&d_out, &x, true);

        let eps = 1e-4;
        let out_size = 8; // 2x4
        for i in 0..8 {
            let mut x_plus = x.clone();
            x_plus.data[i] += eps;
            let mut x_minus = x.clone();
            x_minus.data[i] -= eps;
            let y_plus = mha.forward(&x_plus, true);
            let y_minus = mha.forward(&x_minus, true);
            let mut numerical = 0.0;
            for j in 0..out_size {
                numerical += (y_plus.data[j] - y_minus.data[j]) / (2.0 * eps) * d_out.data[j];
            }
            assert!(
                (grads.d_x.data[i] - numerical).abs() < 1e-2,
                "d_x[{}]: analytical {} vs numerical {}",
                i,
                grads.d_x.data[i],
                numerical
            );
        }
    }

    #[test]
    fn test_backward_numerical_d_wo_weight() {
        let mha = MultiHeadAttention::rand(4, 2);
        let x = Tensor::new(
            vec![0.1, -0.2, 0.3, 0.4, 0.5, 0.1, -0.3, 0.2],
            vec![2, 4],
        );
        let d_out = Tensor::new(
            vec![0.5, -0.3, 0.1, 0.2, -0.1, 0.4, -0.2, 0.6],
            vec![2, 4],
        );
        let grads = mha.backward(&d_out, &x, false);

        let eps = 1e-4;
        let out_size = 8;
        // Check a few elements of d_wo_weight
        for i in 0..4 {
            let mut mha_plus = MultiHeadAttention::rand(4, 2);
            // Copy all params from mha
            mha_plus.wq = Linear::new(mha.wq.weight.clone(), mha.wq.bias.clone());
            mha_plus.wk = Linear::new(mha.wk.weight.clone(), mha.wk.bias.clone());
            mha_plus.wv = Linear::new(mha.wv.weight.clone(), mha.wv.bias.clone());
            mha_plus.wo = Linear::new(mha.wo.weight.clone(), mha.wo.bias.clone());
            mha_plus.wo.weight.data[i] += eps;

            let mut mha_minus = MultiHeadAttention::rand(4, 2);
            mha_minus.wq = Linear::new(mha.wq.weight.clone(), mha.wq.bias.clone());
            mha_minus.wk = Linear::new(mha.wk.weight.clone(), mha.wk.bias.clone());
            mha_minus.wv = Linear::new(mha.wv.weight.clone(), mha.wv.bias.clone());
            mha_minus.wo = Linear::new(mha.wo.weight.clone(), mha.wo.bias.clone());
            mha_minus.wo.weight.data[i] -= eps;

            let y_plus = mha_plus.forward(&x, false);
            let y_minus = mha_minus.forward(&x, false);
            let mut numerical = 0.0;
            for j in 0..out_size {
                numerical += (y_plus.data[j] - y_minus.data[j]) / (2.0 * eps) * d_out.data[j];
            }
            assert!(
                (grads.d_wo_weight.data[i] - numerical).abs() < 1e-2,
                "d_wo_weight[{}]: analytical {} vs numerical {}",
                i,
                grads.d_wo_weight.data[i],
                numerical
            );
        }
    }
}
