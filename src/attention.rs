use crate::tensor::Tensor;

/// Create a causal mask (upper triangular = -inf, diagonal and below = 0).
/// Returns Tensor [seq_len, seq_len]
pub fn causal_mask(seq_len: usize) -> Tensor {
    let mut data = vec![0.0_f32; seq_len * seq_len];
    for row in 0..seq_len {
        for col in (row + 1)..seq_len {
            data[row * seq_len + col] = f32::NEG_INFINITY;
        }
    }
    Tensor::new(data, vec![seq_len, seq_len])
}

/// Scaled Dot-Product Attention (forward)
/// attn = softmax((Q @ K^T / sqrt(d_k)) + mask) @ V
///
/// Q: [seq_len, d_k]
/// K: [seq_len, d_k]
/// V: [seq_len, d_v]
/// mask: Option<&Tensor> [seq_len, seq_len] (0 or -inf)
/// Returns: [seq_len, d_v]
pub fn scaled_dot_product_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
) -> Tensor {
    let d_k = q.shape[1] as f32;
    let scale = 1.0 / d_k.sqrt();

    // scores = Q @ K^T => [seq_len, seq_len]
    let kt = k.transpose();
    let mut scores = q.matmul(&kt);

    // scale
    for val in scores.data.iter_mut() {
        *val *= scale;
    }

    // apply mask
    if let Some(m) = mask {
        scores = scores.add(m);
    }

    // softmax along last axis
    let weights = crate::softmax::softmax(&scores);

    // weights @ V => [seq_len, d_v]
    weights.matmul(v)
}

/// Backward pass for Scaled Dot-Product Attention.
/// d_output: [seq_len, d_v]
/// q, k, v, mask: cached from forward
/// Returns: (d_q, d_k, d_v) with shapes matching Q, K, V
pub fn scaled_dot_product_attention_backward(
    d_output: &Tensor,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
) -> (Tensor, Tensor, Tensor) {
    let d_k = q.shape[1] as f32;
    let scale = 1.0 / d_k.sqrt();

    // Recompute forward intermediates
    let kt = k.transpose();
    let mut scores = q.matmul(&kt);
    for val in scores.data.iter_mut() {
        *val *= scale;
    }
    if let Some(m) = mask {
        scores = scores.add(m);
    }
    let weights = crate::softmax::softmax(&scores);

    // Backward through: output = weights @ V
    // d_weights = d_output @ V^T
    let vt = v.transpose();
    let d_weights = d_output.matmul(&vt);
    // d_v = weights^T @ d_output
    let weights_t = weights.transpose();
    let d_v = weights_t.matmul(d_output);

    // Backward through softmax
    let d_scores_scaled = crate::softmax::softmax_backward(&d_weights, &weights);

    // Backward through scaling: d_scores = d_scores_scaled * scale
    let mut d_scores = d_scores_scaled;
    for val in d_scores.data.iter_mut() {
        *val *= scale;
    }

    // Backward through: scores = Q @ K^T
    // d_q = d_scores @ K
    let d_q = d_scores.matmul(k);
    // d_k = d_scores^T @ Q
    let d_scores_t = d_scores.transpose();
    let d_k_out = d_scores_t.matmul(q);

    (d_q, d_k_out, d_v)
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

    // ==================== causal_mask ====================

    #[test]
    fn test_causal_mask_shape() {
        let m = causal_mask(4);
        assert_eq!(m.shape, vec![4, 4]);
    }

    #[test]
    fn test_causal_mask_values() {
        // [[0,    -inf, -inf],
        //  [0,    0,    -inf],
        //  [0,    0,    0   ]]
        let m = causal_mask(3);
        assert_eq!(m.get(&[0, 0]), 0.0);
        assert!(m.get(&[0, 1]) == f32::NEG_INFINITY);
        assert!(m.get(&[0, 2]) == f32::NEG_INFINITY);
        assert_eq!(m.get(&[1, 0]), 0.0);
        assert_eq!(m.get(&[1, 1]), 0.0);
        assert!(m.get(&[1, 2]) == f32::NEG_INFINITY);
        assert_eq!(m.get(&[2, 0]), 0.0);
        assert_eq!(m.get(&[2, 1]), 0.0);
        assert_eq!(m.get(&[2, 2]), 0.0);
    }

    // ==================== attention without mask ====================

    #[test]
    fn test_attention_no_mask_shape() {
        let q = Tensor::new(vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0], vec![3, 2]);
        let k = Tensor::new(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], vec![3, 2]);
        let v = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape, vec![3, 2]);
    }

    #[test]
    fn test_attention_identity_like() {
        // Q=K => Q@K^T has equal diagonal => uniform attention => output is mean of V rows
        let q = Tensor::new(vec![1.0, 0.0, 1.0, 0.0], vec![2, 2]);
        let k = Tensor::new(vec![1.0, 0.0, 1.0, 0.0], vec![2, 2]);
        let v = Tensor::new(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        // Q@K^T = [[1,1],[1,1]], after scale and softmax => [[0.5,0.5],[0.5,0.5]]
        // output = 0.5*[10,20] + 0.5*[30,40] = [20, 30]
        assert_approx(&out.data, &[20.0, 30.0, 20.0, 30.0], 0.1);
    }

    // ==================== attention with causal mask ====================

    #[test]
    fn test_attention_causal_first_token() {
        // With causal mask, first token can only attend to itself
        let q = Tensor::new(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], vec![3, 2]);
        let k = q.clone();
        let v = Tensor::new(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], vec![3, 2]);
        let mask = causal_mask(3);
        let out = scaled_dot_product_attention(&q, &k, &v, Some(&mask));
        // First token output should be exactly V[0] = [10, 20]
        assert_approx(&out.data[0..2], &[10.0, 20.0], 1e-4);
    }

    #[test]
    fn test_attention_causal_second_token() {
        // With causal mask, second token attends to [0, 1] only
        let q = Tensor::new(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let k = Tensor::new(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let v = Tensor::new(vec![0.0, 10.0, 20.0, 30.0], vec![2, 2]);
        let mask = causal_mask(2);
        let out = scaled_dot_product_attention(&q, &k, &v, Some(&mask));
        // Token 0: attends only to itself => [0, 10]
        assert_approx(&out.data[0..2], &[0.0, 10.0], 1e-4);
        // Token 1: attends to [0,1] equally (same Q,K) => mean([0,10],[20,30]) = [10, 20]
        assert_approx(&out.data[2..4], &[10.0, 20.0], 0.1);
    }

    #[test]
    fn test_attention_scaling() {
        // Verify that scaling by 1/sqrt(d_k) is applied
        // With d_k=4 and large values, without scaling softmax would be nearly one-hot
        // With scaling, the distribution should be smoother
        let q = Tensor::new(vec![5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0], vec![2, 4]);
        let k = Tensor::new(vec![5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0], vec![2, 4]);
        let v = Tensor::new(vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        // With scaling by 1/sqrt(4)=0.5, the attention should not be too sharp
        // Output should be somewhere between v[0] and v[1]
        assert!(out.data[0] > 0.0 && out.data[0] < 1.0);
    }

    // ==================== backward ====================

    #[test]
    fn test_attention_backward_shapes() {
        // seq_len=2, d_k=3, d_v=2
        let q = Tensor::new(vec![1.0, 0.0, 0.5, 0.0, 1.0, -0.5], vec![2, 3]);
        let k = Tensor::new(vec![0.5, 1.0, 0.0, -0.5, 0.0, 1.0], vec![2, 3]);
        let v = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let d_out = Tensor::new(vec![0.1, 0.2, 0.3, 0.4], vec![2, 2]);
        let (d_q, d_k, d_v) = scaled_dot_product_attention_backward(&d_out, &q, &k, &v, None);
        assert_eq!(d_q.shape, vec![2, 3]);
        assert_eq!(d_k.shape, vec![2, 3]);
        assert_eq!(d_v.shape, vec![2, 2]);
    }

    #[test]
    fn test_attention_backward_numerical_no_mask() {
        let q = Tensor::new(vec![0.1, 0.2, 0.3, 0.4, -0.1, 0.5], vec![2, 3]);
        let k = Tensor::new(vec![0.3, -0.1, 0.2, 0.1, 0.4, -0.2], vec![2, 3]);
        let v = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let d_out = Tensor::new(vec![0.5, -0.3, 0.1, 0.7], vec![2, 2]);

        let (d_q, d_k, d_v) = scaled_dot_product_attention_backward(&d_out, &q, &k, &v, None);

        let eps = 1e-4;
        let out_size = 4; // 2x2

        // Check d_q
        for i in 0..6 {
            let mut q_plus = q.clone();
            q_plus.data[i] += eps;
            let mut q_minus = q.clone();
            q_minus.data[i] -= eps;
            let y_plus = scaled_dot_product_attention(&q_plus, &k, &v, None);
            let y_minus = scaled_dot_product_attention(&q_minus, &k, &v, None);
            let mut numerical = 0.0;
            for j in 0..out_size {
                numerical += (y_plus.data[j] - y_minus.data[j]) / (2.0 * eps) * d_out.data[j];
            }
            assert!(
                (d_q.data[i] - numerical).abs() < 1e-2,
                "d_q[{}]: analytical {} vs numerical {}", i, d_q.data[i], numerical
            );
        }

        // Check d_k
        for i in 0..6 {
            let mut k_plus = k.clone();
            k_plus.data[i] += eps;
            let mut k_minus = k.clone();
            k_minus.data[i] -= eps;
            let y_plus = scaled_dot_product_attention(&q, &k_plus, &v, None);
            let y_minus = scaled_dot_product_attention(&q, &k_minus, &v, None);
            let mut numerical = 0.0;
            for j in 0..out_size {
                numerical += (y_plus.data[j] - y_minus.data[j]) / (2.0 * eps) * d_out.data[j];
            }
            assert!(
                (d_k.data[i] - numerical).abs() < 1e-2,
                "d_k[{}]: analytical {} vs numerical {}", i, d_k.data[i], numerical
            );
        }

        // Check d_v
        for i in 0..4 {
            let mut v_plus = v.clone();
            v_plus.data[i] += eps;
            let mut v_minus = v.clone();
            v_minus.data[i] -= eps;
            let y_plus = scaled_dot_product_attention(&q, &k, &v_plus, None);
            let y_minus = scaled_dot_product_attention(&q, &k, &v_minus, None);
            let mut numerical = 0.0;
            for j in 0..out_size {
                numerical += (y_plus.data[j] - y_minus.data[j]) / (2.0 * eps) * d_out.data[j];
            }
            assert!(
                (d_v.data[i] - numerical).abs() < 1e-2,
                "d_v[{}]: analytical {} vs numerical {}", i, d_v.data[i], numerical
            );
        }
    }

    #[test]
    fn test_attention_backward_numerical_with_mask() {
        let q = Tensor::new(vec![0.1, 0.2, 0.3, 0.4, -0.1, 0.5, 0.2, -0.3, 0.1], vec![3, 3]);
        let k = Tensor::new(vec![0.3, -0.1, 0.2, 0.1, 0.4, -0.2, -0.3, 0.1, 0.5], vec![3, 3]);
        let v = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
        let mask = causal_mask(3);
        let d_out = Tensor::new(vec![0.5, -0.3, 0.1, 0.7, -0.2, 0.4], vec![3, 2]);

        let (d_q, d_k, d_v) = scaled_dot_product_attention_backward(
            &d_out, &q, &k, &v, Some(&mask),
        );

        let eps = 1e-4;
        let out_size = 6; // 3x2

        // Check d_q
        for i in 0..9 {
            let mut q_plus = q.clone();
            q_plus.data[i] += eps;
            let mut q_minus = q.clone();
            q_minus.data[i] -= eps;
            let y_plus = scaled_dot_product_attention(&q_plus, &k, &v, Some(&mask));
            let y_minus = scaled_dot_product_attention(&q_minus, &k, &v, Some(&mask));
            let mut numerical = 0.0;
            for j in 0..out_size {
                numerical += (y_plus.data[j] - y_minus.data[j]) / (2.0 * eps) * d_out.data[j];
            }
            assert!(
                (d_q.data[i] - numerical).abs() < 1e-2,
                "d_q[{}]: analytical {} vs numerical {}", i, d_q.data[i], numerical
            );
        }

        // Check d_k
        for i in 0..9 {
            let mut k_plus = k.clone();
            k_plus.data[i] += eps;
            let mut k_minus = k.clone();
            k_minus.data[i] -= eps;
            let y_plus = scaled_dot_product_attention(&q, &k_plus, &v, Some(&mask));
            let y_minus = scaled_dot_product_attention(&q, &k_minus, &v, Some(&mask));
            let mut numerical = 0.0;
            for j in 0..out_size {
                numerical += (y_plus.data[j] - y_minus.data[j]) / (2.0 * eps) * d_out.data[j];
            }
            assert!(
                (d_k.data[i] - numerical).abs() < 1e-2,
                "d_k[{}]: analytical {} vs numerical {}", i, d_k.data[i], numerical
            );
        }

        // Check d_v
        for i in 0..6 {
            let mut v_plus = v.clone();
            v_plus.data[i] += eps;
            let mut v_minus = v.clone();
            v_minus.data[i] -= eps;
            let y_plus = scaled_dot_product_attention(&q, &k, &v_plus, Some(&mask));
            let y_minus = scaled_dot_product_attention(&q, &k, &v_minus, Some(&mask));
            let mut numerical = 0.0;
            for j in 0..out_size {
                numerical += (y_plus.data[j] - y_minus.data[j]) / (2.0 * eps) * d_out.data[j];
            }
            assert!(
                (d_v.data[i] - numerical).abs() < 1e-2,
                "d_v[{}]: analytical {} vs numerical {}", i, d_v.data[i], numerical
            );
        }
    }
}
