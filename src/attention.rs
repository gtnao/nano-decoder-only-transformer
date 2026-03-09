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

/// Scaled Dot-Product Attention
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
}
