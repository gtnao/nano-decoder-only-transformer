use crate::softmax::softmax;
use crate::tensor::Tensor;

/// Cross Entropy Loss for language modeling.
/// logits: [seq_len, vocab_size] (raw model output)
/// targets: &[usize] (target token IDs, length = seq_len)
/// Returns scalar loss value (averaged over positions).
pub fn cross_entropy_loss(logits: &Tensor, targets: &[usize]) -> f32 {
    assert_eq!(logits.shape.len(), 2);
    let seq_len = logits.shape[0];
    let vocab_size = logits.shape[1];
    assert_eq!(targets.len(), seq_len);

    let mut total_loss = 0.0_f32;

    for pos in 0..seq_len {
        let start = pos * vocab_size;
        let row = &logits.data[start..start + vocab_size];

        // log-softmax with numerical stability:
        // log_softmax(i) = x_i - max - log(Σexp(x_j - max))
        let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let log_sum_exp = row.iter().map(|&v| (v - max_val).exp()).sum::<f32>().ln();
        let log_prob = row[targets[pos]] - max_val - log_sum_exp;

        total_loss += -log_prob;
    }

    total_loss / seq_len as f32
}

/// Backward of cross entropy loss with softmax.
/// Returns d_logits: [seq_len, vocab_size]
/// d_logits = (softmax(logits) - one_hot(targets)) / seq_len
pub fn cross_entropy_loss_backward(logits: &Tensor, targets: &[usize]) -> Tensor {
    let seq_len = logits.shape[0];
    let probs = softmax(logits);
    let mut d_logits = probs.data.clone();
    let vocab_size = logits.shape[1];

    for pos in 0..seq_len {
        d_logits[pos * vocab_size + targets[pos]] -= 1.0;
    }

    // Average over positions
    let scale = 1.0 / seq_len as f32;
    for v in d_logits.iter_mut() {
        *v *= scale;
    }

    Tensor::new(d_logits, logits.shape.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perfect_prediction() {
        // logits strongly favor the correct token => loss should be near 0
        let logits = Tensor::new(
            vec![
                10.0, -10.0, -10.0, // position 0: strongly predicts token 0
                -10.0, 10.0, -10.0, // position 1: strongly predicts token 1
            ],
            vec![2, 3],
        );
        let targets = vec![0, 1];
        let loss = cross_entropy_loss(&logits, &targets);
        assert!(loss < 0.001, "loss: {}", loss);
    }

    #[test]
    fn test_worst_prediction() {
        // logits strongly favor the wrong token => loss should be high
        let logits = Tensor::new(
            vec![
                -10.0, 10.0, -10.0, // position 0: predicts token 1, but target is 0
            ],
            vec![1, 3],
        );
        let targets = vec![0];
        let loss = cross_entropy_loss(&logits, &targets);
        assert!(loss > 10.0, "loss: {}", loss);
    }

    #[test]
    fn test_uniform_prediction() {
        // uniform logits => loss = ln(vocab_size)
        let vocab_size = 4;
        let logits = Tensor::new(vec![0.0; vocab_size * 2], vec![2, vocab_size]);
        let targets = vec![0, 1];
        let loss = cross_entropy_loss(&logits, &targets);
        let expected = (vocab_size as f32).ln();
        assert!(
            (loss - expected).abs() < 0.001,
            "loss: {}, expected: {}",
            loss,
            expected
        );
    }

    #[test]
    fn test_numerical_stability() {
        // Large logits should not cause overflow
        let logits = Tensor::new(
            vec![1000.0, 1001.0, 1002.0],
            vec![1, 3],
        );
        let targets = vec![2];
        let loss = cross_entropy_loss(&logits, &targets);
        assert!(loss.is_finite(), "loss: {}", loss);
    }

    #[test]
    fn test_single_position() {
        // Manual calculation:
        // logits = [1.0, 2.0, 3.0], target = 0
        // log_softmax(0) = 1.0 - log(exp(1)+exp(2)+exp(3))
        //                = 1.0 - log(2.718 + 7.389 + 20.086)
        //                = 1.0 - log(30.193)
        //                = 1.0 - 3.408 = -2.408
        // loss = -log_softmax(0) = 2.408
        let logits = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let targets = vec![0];
        let loss = cross_entropy_loss(&logits, &targets);
        assert!(
            (loss - 2.408).abs() < 0.01,
            "loss: {}, expected: ~2.408",
            loss
        );
    }

    #[test]
    fn test_averaged_over_positions() {
        // Loss should be average, not sum
        let logits = Tensor::new(vec![0.0; 3 * 2], vec![2, 3]); // 2 positions
        let targets = vec![0, 0];
        let loss_2pos = cross_entropy_loss(&logits, &targets);

        let logits1 = Tensor::new(vec![0.0; 3], vec![1, 3]); // 1 position
        let targets1 = vec![0];
        let loss_1pos = cross_entropy_loss(&logits1, &targets1);

        // Both should be ln(3) since all uniform
        assert!((loss_2pos - loss_1pos).abs() < 0.001);
    }

    // ==================== backward ====================

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

    #[test]
    fn test_backward_shape() {
        let logits = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let targets = vec![0, 1];
        let d_logits = cross_entropy_loss_backward(&logits, &targets);
        assert_eq!(d_logits.shape, vec![2, 3]);
    }

    #[test]
    fn test_backward_sums_to_zero_per_row() {
        // d_logits = softmax - one_hot => each row sums to (1 - 1)/N = 0
        let logits = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let targets = vec![0, 2];
        let d = cross_entropy_loss_backward(&logits, &targets);
        let sum0: f32 = d.data[0..3].iter().sum();
        let sum1: f32 = d.data[3..6].iter().sum();
        assert!(sum0.abs() < 1e-5, "sum0: {}", sum0);
        assert!(sum1.abs() < 1e-5, "sum1: {}", sum1);
    }

    #[test]
    fn test_backward_target_position_negative() {
        // Gradient at target index should be negative (softmax(i) - 1) / N < 0
        let logits = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let targets = vec![0];
        let d = cross_entropy_loss_backward(&logits, &targets);
        assert!(d.data[0] < 0.0, "d[target]: {}", d.data[0]);
    }

    #[test]
    fn test_backward_numerical_gradient() {
        // Verify backward against numerical gradient (finite differences)
        let logits = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let targets = vec![1];
        let d = cross_entropy_loss_backward(&logits, &targets);

        let eps = 1e-4;
        for i in 0..3 {
            let mut logits_plus = logits.clone();
            logits_plus.data[i] += eps;
            let mut logits_minus = logits.clone();
            logits_minus.data[i] -= eps;
            let numerical = (cross_entropy_loss(&logits_plus, &targets)
                - cross_entropy_loss(&logits_minus, &targets))
                / (2.0 * eps);
            assert!(
                (d.data[i] - numerical).abs() < 1e-3,
                "index {}: analytical {} vs numerical {}",
                i, d.data[i], numerical
            );
        }
    }
}
