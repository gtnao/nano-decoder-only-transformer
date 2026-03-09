use crate::softmax::softmax;
use crate::tensor::Tensor;
use crate::tokenizer::Tokenizer;
use crate::transformer::Transformer;

/// Pick the token with the highest logit (greedy decoding).
pub fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap()
        .0
}

/// Sample from probability distribution using temperature.
/// Higher temperature => more random, lower => more deterministic.
pub fn sample_with_temperature(logits: &[f32], temperature: f32) -> usize {
    use rand::RngExt;

    // Scale logits by temperature
    let scaled: Vec<f32> = logits.iter().map(|&v| v / temperature).collect();
    let probs = softmax(&Tensor::new(scaled, vec![logits.len()]));

    // Sample from cumulative distribution
    let mut rng = rand::rng();
    let r: f32 = rng.random_range(0.0..1.0);
    let mut cumulative = 0.0;
    for (i, &p) in probs.data.iter().enumerate() {
        cumulative += p;
        if r < cumulative {
            return i;
        }
    }
    probs.data.len() - 1
}

/// Auto-regressive text generation.
pub fn generate(
    model: &Transformer,
    tokenizer: &Tokenizer,
    prompt: &str,
    max_new_tokens: usize,
    temperature: f32,
) -> String {
    let mut token_ids = tokenizer.encode(prompt);

    for _ in 0..max_new_tokens {
        let logits = model.forward(&token_ids);

        // Take logits of the last position
        let vocab_size = logits.shape[1];
        let last_pos = logits.shape[0] - 1;
        let start = last_pos * vocab_size;
        let last_logits = &logits.data[start..start + vocab_size];

        let next_id = if temperature <= 0.01 {
            argmax(last_logits)
        } else {
            sample_with_temperature(last_logits, temperature)
        };

        token_ids.push(next_id);
    }

    tokenizer.decode(&token_ids)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== argmax ====================

    #[test]
    fn test_argmax_basic() {
        assert_eq!(argmax(&[1.0, 3.0, 2.0]), 1);
    }

    #[test]
    fn test_argmax_first() {
        assert_eq!(argmax(&[5.0, 1.0, 2.0]), 0);
    }

    #[test]
    fn test_argmax_last() {
        assert_eq!(argmax(&[1.0, 2.0, 9.0]), 2);
    }

    #[test]
    fn test_argmax_negative() {
        assert_eq!(argmax(&[-1.0, -3.0, -2.0]), 0);
    }

    // ==================== sample_with_temperature ====================

    #[test]
    fn test_sample_low_temperature() {
        // Very low temperature => almost deterministic => should pick max
        let logits = vec![1.0, 10.0, 2.0];
        // Run multiple times; with temp=0.01 it should always pick index 1
        for _ in 0..10 {
            assert_eq!(sample_with_temperature(&logits, 0.01), 1);
        }
    }

    #[test]
    fn test_sample_returns_valid_index() {
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        for _ in 0..20 {
            let idx = sample_with_temperature(&logits, 1.0);
            assert!(idx < 4);
        }
    }

    // ==================== generate ====================

    #[test]
    fn test_generate_returns_string() {
        let corpus = "abcde";
        let tokenizer = Tokenizer::from_corpus(corpus);
        let model = Transformer::rand(tokenizer.vocab_size(), 8, 2, 16, 1);
        let result = generate(&model, &tokenizer, "ab", 5, 1.0);
        // Result should start with the prompt
        assert!(result.starts_with("ab"));
        // And have generated some additional characters
        assert!(result.len() > 2);
    }

    #[test]
    fn test_generate_max_tokens() {
        let corpus = "abcde";
        let tokenizer = Tokenizer::from_corpus(corpus);
        let model = Transformer::rand(tokenizer.vocab_size(), 8, 2, 16, 1);
        let result = generate(&model, &tokenizer, "a", 3, 1.0);
        // prompt (1 char) + 3 generated => at most 4 chars
        assert!(result.chars().count() <= 4);
    }

    #[test]
    fn test_generate_greedy_deterministic() {
        let corpus = "abcde";
        let tokenizer = Tokenizer::from_corpus(corpus);
        let model = Transformer::rand(tokenizer.vocab_size(), 8, 2, 16, 1);
        // temperature=0.01 => nearly greedy => deterministic
        let r1 = generate(&model, &tokenizer, "ab", 5, 0.01);
        let r2 = generate(&model, &tokenizer, "ab", 5, 0.01);
        assert_eq!(r1, r2);
    }
}
