use crate::tensor::Tensor;

/// Apply dropout. Each element zeroed with probability `rate`,
/// remaining scaled by 1/(1-rate) to preserve expected value.
/// Returns (output, mask) where mask[i] = true means element was kept.
pub fn dropout(x: &Tensor, rate: f32) -> (Tensor, Vec<bool>) {
    use rand::RngExt;
    let mut rng = rand::rng();
    let scale = 1.0 / (1.0 - rate);
    let mut data = Vec::with_capacity(x.data.len());
    let mut mask = Vec::with_capacity(x.data.len());

    for &v in &x.data {
        let keep = rng.random_range(0.0_f32..1.0) >= rate;
        mask.push(keep);
        data.push(if keep { v * scale } else { 0.0 });
    }

    (Tensor::new(data, x.shape.clone()), mask)
}

/// Dropout backward: apply same mask and scale as forward.
pub fn dropout_backward(d_output: &Tensor, mask: &[bool], rate: f32) -> Tensor {
    let scale = 1.0 / (1.0 - rate);
    let data: Vec<f32> = d_output
        .data
        .iter()
        .zip(mask.iter())
        .map(|(&d, &keep)| if keep { d * scale } else { 0.0 })
        .collect();
    Tensor::new(data, d_output.shape.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dropout_zero_rate() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let (y, mask) = dropout(&x, 0.0);
        // rate=0 => nothing dropped, no scaling
        assert_eq!(y.data, vec![1.0, 2.0, 3.0, 4.0]);
        assert!(mask.iter().all(|&k| k));
    }

    #[test]
    fn test_dropout_preserves_expected_value() {
        // Over many elements, mean should be approximately preserved
        let n = 10000;
        let x = Tensor::new(vec![1.0; n], vec![n]);
        let (y, _) = dropout(&x, 0.3);
        let mean = y.data.iter().sum::<f32>() / n as f32;
        // Expected value = 1.0 (since kept elements are scaled by 1/(1-0.3))
        assert!(
            (mean - 1.0).abs() < 0.1,
            "mean should be ~1.0, got {}",
            mean
        );
    }

    #[test]
    fn test_dropout_zeros_some_elements() {
        let n = 1000;
        let x = Tensor::new(vec![1.0; n], vec![n]);
        let (y, mask) = dropout(&x, 0.5);
        let num_zeros = y.data.iter().filter(|&&v| v == 0.0).count();
        let num_kept = mask.iter().filter(|&&k| k).count();
        // With rate=0.5, roughly half should be zero
        assert!(num_zeros > 300 && num_zeros < 700,
            "expected ~500 zeros, got {}", num_zeros);
        assert_eq!(num_kept, n - num_zeros);
    }

    #[test]
    fn test_dropout_backward_matches_mask() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let (_, mask) = dropout(&x, 0.5);
        let d_out = Tensor::new(vec![1.0, 1.0, 1.0, 1.0], vec![4]);
        let d_in = dropout_backward(&d_out, &mask, 0.5);
        // Backward should zero the same elements and scale the kept ones
        let scale = 1.0 / (1.0 - 0.5);
        for (i, &keep) in mask.iter().enumerate() {
            if keep {
                assert!((d_in.data[i] - scale).abs() < 1e-6);
            } else {
                assert_eq!(d_in.data[i], 0.0);
            }
        }
    }

    #[test]
    fn test_dropout_shape_preserved() {
        let x = Tensor::new(vec![1.0; 6], vec![2, 3]);
        let (y, mask) = dropout(&x, 0.3);
        assert_eq!(y.shape, vec![2, 3]);
        assert_eq!(mask.len(), 6);
    }
}
