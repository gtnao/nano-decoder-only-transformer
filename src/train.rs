use crate::loss::{cross_entropy_loss, cross_entropy_loss_backward};
use crate::optimizer::{Adam, LRSchedule};
use crate::tokenizer::TokenizerTrait;
use crate::transformer::{Transformer, TransformerGradients};

/// Prepare training data: split token sequence into (input, target) pairs.
/// For language modeling: input = tokens[0..n-1], target = tokens[1..n]
/// Returns Vec<(input_ids, target_ids)> of fixed-length windows.
pub fn prepare_data(token_ids: &[usize], seq_len: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
    let mut pairs = Vec::new();
    // Sliding window with stride = seq_len (non-overlapping)
    let mut start = 0;
    while start + seq_len + 1 <= token_ids.len() {
        let input = token_ids[start..start + seq_len].to_vec();
        let target = token_ids[start + 1..start + seq_len + 1].to_vec();
        pairs.push((input, target));
        start += seq_len;
    }
    pairs
}

/// Run one training step on a single sample. Returns loss value.
pub fn train_step(
    model: &mut Transformer,
    adam: &mut Adam,
    input: &[usize],
    target: &[usize],
    max_grad_norm: f32,
) -> f32 {
    train_step_batch(model, adam, &[(input.to_vec(), target.to_vec())], max_grad_norm)
}

/// Run one training step on a batch. Returns average loss.
pub fn train_step_batch(
    model: &mut Transformer,
    adam: &mut Adam,
    batch: &[(Vec<usize>, Vec<usize>)],
    max_grad_norm: f32,
) -> f32 {
    let batch_size = batch.len();
    let mut accumulated = TransformerGradients::zeros_like(model);
    let mut total_loss = 0.0;

    for (input, target) in batch {
        // Forward (with dropout enabled)
        let logits = model.forward_with_training(input, true);
        let loss = cross_entropy_loss(&logits, target);
        total_loss += loss;

        // Backward
        let d_logits = cross_entropy_loss_backward(&logits, target);
        let grads = model.backward(&d_logits, input);
        accumulated.accumulate(&grads);
    }

    // Average gradients over batch
    accumulated.scale(1.0 / batch_size as f32);

    // Gradient clipping
    accumulated.clip_norm(max_grad_norm);

    // Update
    model.apply_gradients(&accumulated, adam);

    total_loss / batch_size as f32
}

/// Train the model for a given number of epochs.
/// Returns loss history (one per training step).
pub fn train(
    model: &mut Transformer,
    tokenizer: &dyn TokenizerTrait,
    text: &str,
    seq_len: usize,
    epochs: usize,
    lr: f32,
) -> Vec<f32> {
    train_with_batch(model, tokenizer, text, seq_len, epochs, lr, 1)
}

/// Train with explicit batch size.
pub fn train_with_batch(
    model: &mut Transformer,
    tokenizer: &dyn TokenizerTrait,
    text: &str,
    seq_len: usize,
    epochs: usize,
    lr: f32,
    batch_size: usize,
) -> Vec<f32> {
    let token_ids = tokenizer.encode(text);
    let data = prepare_data(&token_ids, seq_len);

    if data.is_empty() {
        return Vec::new();
    }

    let max_grad_norm = 1.0;
    let batches_per_epoch = (data.len() + batch_size - 1) / batch_size;
    let total_steps = batches_per_epoch * epochs;
    let warmup_steps = total_steps / 10; // 10% warmup
    let schedule = LRSchedule::new(lr, warmup_steps, total_steps);
    let mut adam = Adam::new(lr, &model.param_sizes());
    let mut losses = Vec::new();
    let mut step = 0;

    let train_start = std::time::Instant::now();
    eprintln!(
        "Training: {} epochs x {} batches/epoch (batch_size={}, {} samples) = {} total steps",
        epochs, batches_per_epoch, batch_size, data.len(), total_steps
    );

    for epoch in 0..epochs {
        let mut epoch_loss = 0.0;
        let mut epoch_batches = 0;
        for batch in data.chunks(batch_size) {
            adam.set_lr(schedule.get_lr(step));
            let batch_vec: Vec<_> = batch.to_vec();
            let loss = train_step_batch(model, &mut adam, &batch_vec, max_grad_norm);
            epoch_loss += loss;
            epoch_batches += 1;
            losses.push(loss);
            step += 1;

            let elapsed = train_start.elapsed().as_secs_f64();
            let per_step = elapsed / step as f64;
            let eta = per_step * (total_steps - step) as f64;
            eprintln!(
                "  step {}/{} (epoch {}/{}) loss={:.4} lr={:.6} [{:.1}s elapsed, ETA {:.0}s]",
                step, total_steps, epoch + 1, epochs,
                loss, schedule.get_lr(step.saturating_sub(1)),
                elapsed, eta
            );
        }
        let avg_loss = epoch_loss / epoch_batches as f32;
        eprintln!("Epoch {}/{}: avg_loss={:.4}", epoch + 1, epochs, avg_loss);
    }

    losses
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::Tokenizer;
    use crate::transformer::{Transformer, TransformerGradients};

    #[test]
    fn test_prepare_data_basic() {
        let tokens = vec![0, 1, 2, 3, 4, 5, 6];
        let pairs = prepare_data(&tokens, 3);
        // Window 1: input=[0,1,2], target=[1,2,3]
        // Window 2: input=[3,4,5], target=[4,5,6]
        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs[0].0, vec![0, 1, 2]);
        assert_eq!(pairs[0].1, vec![1, 2, 3]);
        assert_eq!(pairs[1].0, vec![3, 4, 5]);
        assert_eq!(pairs[1].1, vec![4, 5, 6]);
    }

    #[test]
    fn test_prepare_data_too_short() {
        let tokens = vec![0, 1];
        let pairs = prepare_data(&tokens, 3);
        assert_eq!(pairs.len(), 0);
    }

    #[test]
    fn test_prepare_data_exact() {
        let tokens = vec![0, 1, 2, 3];
        let pairs = prepare_data(&tokens, 3);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].0, vec![0, 1, 2]);
        assert_eq!(pairs[0].1, vec![1, 2, 3]);
    }

    #[test]
    fn test_train_step_returns_finite_loss() {
        let tokenizer = Tokenizer::from_corpus("abcd");
        let model_vocab = tokenizer.vocab_size();
        let mut model = Transformer::rand(model_vocab, 8, 2, 16, 1);
        let mut adam = Adam::new(0.001, &model.param_sizes());
        let input = vec![2, 3, 4]; // some token ids
        let target = vec![3, 4, 5];
        let loss = train_step(&mut model, &mut adam, &input, &target, 1.0);
        assert!(loss.is_finite(), "loss should be finite, got {}", loss);
    }

    #[test]
    fn test_train_loss_decreases() {
        // Train on a tiny corpus and verify loss decreases
        let corpus = "abcabc";
        let tokenizer = Tokenizer::from_corpus(corpus);
        let mut model = Transformer::rand(tokenizer.vocab_size(), 16, 2, 32, 1);

        let losses = train(&mut model, &tokenizer, corpus, 3, 50, 0.001);

        assert!(!losses.is_empty(), "should have training steps");

        // Compare average of first 5 losses vs last 5 losses
        let n = losses.len();
        let first_avg: f32 = losses[..5.min(n)].iter().sum::<f32>() / 5.0_f32.min(n as f32);
        let last_avg: f32 = losses[n.saturating_sub(5)..].iter().sum::<f32>()
            / 5.0_f32.min(n as f32);
        assert!(
            last_avg < first_avg,
            "loss should decrease: first_avg={}, last_avg={}",
            first_avg,
            last_avg
        );
    }

    #[test]
    fn test_overfit_tiny_sequence() {
        // Model should be able to memorize a tiny sequence
        let corpus = "aba";
        let tokenizer = Tokenizer::from_corpus(corpus);
        let mut model = Transformer::rand(tokenizer.vocab_size(), 16, 2, 32, 1);

        let losses = train(&mut model, &tokenizer, corpus, 2, 200, 0.001);

        let final_loss = *losses.last().unwrap();
        assert!(
            final_loss < 1.0,
            "should overfit to low loss, got {}",
            final_loss
        );
    }

    #[test]
    fn test_zeros_like_shapes() {
        let model = Transformer::rand(6, 8, 2, 16, 2);
        let grads = TransformerGradients::zeros_like(&model);
        assert_eq!(grads.d_embedding_weight.shape, vec![6, 8]);
        assert_eq!(grads.block_grads.len(), 2);
        assert_eq!(grads.d_lm_head_weight.shape, vec![6, 8]);
        assert_eq!(grads.d_lm_head_bias.shape, vec![6]);
        assert_eq!(grads.block_grads[0].mha_grads.d_wq_weight.shape, vec![8, 8]);
        assert_eq!(grads.block_grads[0].ffn_grads.d_l1_weight.shape, vec![16, 8]);
    }

    #[test]
    fn test_train_step_batch_returns_finite_loss() {
        let tokenizer = Tokenizer::from_corpus("abcdef");
        let mut model = Transformer::rand(tokenizer.vocab_size(), 8, 2, 16, 1);
        let mut adam = Adam::new(0.001, &model.param_sizes());
        let batch = vec![
            (vec![2, 3, 4], vec![3, 4, 5]),
            (vec![3, 4, 5], vec![4, 5, 6]),
        ];
        let loss = train_step_batch(&mut model, &mut adam, &batch, 1.0);
        assert!(loss.is_finite(), "batch loss should be finite, got {}", loss);
    }

    #[test]
    fn test_train_with_batch_loss_decreases() {
        let corpus = "abcabcabc";
        let tokenizer = Tokenizer::from_corpus(corpus);
        let mut model = Transformer::rand(tokenizer.vocab_size(), 16, 2, 32, 1);

        let losses = train_with_batch(&mut model, &tokenizer, corpus, 3, 50, 0.001, 2);

        assert!(!losses.is_empty(), "should have training steps");
        let n = losses.len();
        let first_avg: f32 = losses[..3.min(n)].iter().sum::<f32>() / 3.0_f32.min(n as f32);
        let last_avg: f32 = losses[n.saturating_sub(3)..].iter().sum::<f32>()
            / 3.0_f32.min(n as f32);
        assert!(
            last_avg < first_avg,
            "loss should decrease: first_avg={}, last_avg={}",
            first_avg,
            last_avg
        );
    }
}
