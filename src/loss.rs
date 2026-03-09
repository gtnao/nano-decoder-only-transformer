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
}
