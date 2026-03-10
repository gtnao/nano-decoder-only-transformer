use std::collections::HashMap;
use crate::tokenizer::TokenizerTrait;

/// BPE (Byte Pair Encoding) tokenizer.
/// Learns subword units by iteratively merging the most frequent adjacent pairs.
pub struct BpeTokenizer {
    /// Ordered merge rules: (token_a, token_b) -> merged_token
    merges: Vec<(String, String)>,
    /// Token string -> ID
    token_to_id: HashMap<String, usize>,
    /// ID -> Token string
    id_to_token: Vec<String>,
}

impl BpeTokenizer {
    /// Train BPE on a corpus to build a vocabulary of the given size.
    /// vocab_size includes special tokens (<PAD>=0, <UNK>=1).
    pub fn train(corpus: &str, vocab_size: usize) -> Self {
        // Initial vocabulary: special tokens + unique characters
        let mut id_to_token: Vec<String> = vec!["<PAD>".to_string(), "<UNK>".to_string()];
        let mut token_to_id: HashMap<String, usize> = HashMap::new();
        token_to_id.insert("<PAD>".to_string(), 0);
        token_to_id.insert("<UNK>".to_string(), 1);

        // Add all unique characters
        let mut chars: Vec<char> = corpus.chars().collect::<std::collections::BTreeSet<_>>().into_iter().collect();
        chars.sort();
        for c in &chars {
            let s = c.to_string();
            if !token_to_id.contains_key(&s) {
                let id = id_to_token.len();
                token_to_id.insert(s.clone(), id);
                id_to_token.push(s);
            }
        }

        // Represent corpus as a sequence of token strings
        let mut sequence: Vec<String> = corpus.chars().map(|c| c.to_string()).collect();

        let mut merges: Vec<(String, String)> = Vec::new();
        let num_merges = vocab_size.saturating_sub(id_to_token.len());

        for _ in 0..num_merges {
            // Count adjacent pairs
            let pair_counts = count_pairs(&sequence);
            if pair_counts.is_empty() {
                break;
            }

            // Find most frequent pair
            let best_pair = pair_counts
                .iter()
                .max_by_key(|&(_, count)| count)
                .unwrap()
                .0
                .clone();

            // Create merged token
            let merged = format!("{}{}", best_pair.0, best_pair.1);

            // Add to vocabulary
            let id = id_to_token.len();
            token_to_id.insert(merged.clone(), id);
            id_to_token.push(merged);

            // Record merge rule
            merges.push(best_pair.clone());

            // Apply merge to sequence
            sequence = apply_merge(&sequence, &best_pair.0, &best_pair.1);
        }

        eprintln!(
            "BPE: {} merges, vocab_size={}",
            merges.len(),
            id_to_token.len()
        );

        BpeTokenizer {
            merges,
            token_to_id,
            id_to_token,
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }

    /// Encode text into token IDs by applying learned merge rules.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        if text.is_empty() {
            return Vec::new();
        }

        // Start with character-level tokens
        let mut tokens: Vec<String> = text.chars().map(|c| c.to_string()).collect();

        // Apply merge rules in order
        for (a, b) in &self.merges {
            tokens = apply_merge(&tokens, a, b);
        }

        // Map to IDs
        let unk_id = 1;
        tokens
            .iter()
            .map(|t| *self.token_to_id.get(t).unwrap_or(&unk_id))
            .collect()
    }

    /// Decode token IDs back into text.
    pub fn decode(&self, ids: &[usize]) -> String {
        ids.iter()
            .map(|&id| {
                if id < self.id_to_token.len() {
                    self.id_to_token[id].as_str()
                } else {
                    "<UNK>"
                }
            })
            .collect()
    }
}

impl BpeTokenizer {
    /// Escape special characters for text-based save format.
    fn escape(s: &str) -> String {
        s.replace('\\', "\\\\")
            .replace('\n', "\\n")
            .replace('\t', "\\t")
    }

    /// Unescape special characters when loading.
    fn unescape(s: &str) -> String {
        let mut result = String::with_capacity(s.len());
        let mut chars = s.chars();
        while let Some(c) = chars.next() {
            if c == '\\' {
                match chars.next() {
                    Some('n') => result.push('\n'),
                    Some('t') => result.push('\t'),
                    Some('\\') => result.push('\\'),
                    Some(other) => {
                        result.push('\\');
                        result.push(other);
                    }
                    None => result.push('\\'),
                }
            } else {
                result.push(c);
            }
        }
        result
    }

    /// Save tokenizer to a text file.
    /// Format: one line per merge rule "token_a\ttoken_b", then "---", then vocabulary.
    /// Special characters (\n, \t, \\) are escaped.
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        use std::io::Write;
        let mut file = std::fs::File::create(path)?;

        // Write merge rules (escaped)
        for (a, b) in &self.merges {
            writeln!(file, "{}\t{}", Self::escape(a), Self::escape(b))?;
        }
        writeln!(file, "---")?;

        // Write vocabulary (id -> token, escaped)
        for token in &self.id_to_token {
            writeln!(file, "{}", Self::escape(token))?;
        }
        Ok(())
    }

    /// Load tokenizer from a text file.
    pub fn load(path: &str) -> std::io::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let mut lines = content.lines();

        // Read merge rules until "---"
        let mut merges = Vec::new();
        for line in lines.by_ref() {
            if line == "---" {
                break;
            }
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() == 2 {
                merges.push((Self::unescape(parts[0]), Self::unescape(parts[1])));
            }
        }

        // Read vocabulary (unescape each token)
        let mut id_to_token = Vec::new();
        let mut token_to_id = HashMap::new();
        for line in lines {
            let token = Self::unescape(line);
            let id = id_to_token.len();
            token_to_id.insert(token.clone(), id);
            id_to_token.push(token);
        }

        Ok(BpeTokenizer {
            merges,
            token_to_id,
            id_to_token,
        })
    }
}

impl TokenizerTrait for BpeTokenizer {
    fn vocab_size(&self) -> usize { self.vocab_size() }
    fn encode(&self, text: &str) -> Vec<usize> { self.encode(text) }
    fn decode(&self, ids: &[usize]) -> String { self.decode(ids) }
}

/// Count all adjacent pairs in a token sequence.
fn count_pairs(tokens: &[String]) -> HashMap<(String, String), usize> {
    let mut counts = HashMap::new();
    for pair in tokens.windows(2) {
        let key = (pair[0].clone(), pair[1].clone());
        *counts.entry(key).or_insert(0) += 1;
    }
    counts
}

/// Merge all occurrences of (a, b) into a single token "ab".
fn apply_merge(tokens: &[String], a: &str, b: &str) -> Vec<String> {
    let mut result = Vec::with_capacity(tokens.len());
    let mut i = 0;
    while i < tokens.len() {
        if i + 1 < tokens.len() && tokens[i] == a && tokens[i + 1] == b {
            result.push(format!("{}{}", a, b));
            i += 2;
        } else {
            result.push(tokens[i].clone());
            i += 1;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_train_basic() {
        let tok = BpeTokenizer::train("aaabab", 10);
        assert!(tok.vocab_size() >= 4); // PAD, UNK, a, b + merges
    }

    #[test]
    fn test_train_merges_most_frequent() {
        // "ab" appears 3 times, should be first merge
        let tok = BpeTokenizer::train("ababab", 6);
        assert_eq!(tok.merges[0], ("a".to_string(), "b".to_string()));
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let corpus = "こんにちは世界";
        let tok = BpeTokenizer::train(corpus, 20);
        let ids = tok.encode(corpus);
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, corpus);
    }

    #[test]
    fn test_encode_unknown_chars() {
        let tok = BpeTokenizer::train("abc", 10);
        let ids = tok.encode("axz");
        // 'x' and 'z' are not in vocab => UNK=1
        assert_eq!(ids[1], 1);
        assert_eq!(ids[2], 1);
    }

    #[test]
    fn test_encode_empty() {
        let tok = BpeTokenizer::train("abc", 10);
        let ids = tok.encode("");
        assert!(ids.is_empty());
    }

    #[test]
    fn test_merges_reduce_tokens() {
        let corpus = "aaaa";
        let tok = BpeTokenizer::train(corpus, 5);
        // With merges, encoding should produce fewer tokens than chars
        let ids = tok.encode("aaaa");
        assert!(ids.len() < 4, "expected fewer tokens, got {}", ids.len());
    }

    #[test]
    fn test_vocab_size_limit() {
        let tok = BpeTokenizer::train("abcabc", 6);
        assert!(tok.vocab_size() <= 6);
    }

    #[test]
    fn test_japanese_roundtrip() {
        let corpus = "走れメロス走れメロス走れメロス";
        let tok = BpeTokenizer::train(corpus, 20);
        let ids = tok.encode("走れメロス");
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, "走れメロス");
    }

    #[test]
    fn test_japanese_merges() {
        let corpus = "ましたましたました";
        let tok = BpeTokenizer::train(corpus, 10);
        // "ました" should be merged
        let ids = tok.encode("ました");
        // Should be fewer than 3 tokens (ま、し、た)
        assert!(ids.len() < 3, "expected merged tokens, got {} tokens", ids.len());
    }

    #[test]
    fn test_save_load_roundtrip() {
        let corpus = "走れメロス走れメロス走れメロス";
        let tok = BpeTokenizer::train(corpus, 15);

        let path = "/tmp/test_bpe_tokenizer.txt";
        tok.save(path).unwrap();

        let loaded = BpeTokenizer::load(path).unwrap();

        // Verify same vocab size
        assert_eq!(tok.vocab_size(), loaded.vocab_size());

        // Verify same merges
        assert_eq!(tok.merges.len(), loaded.merges.len());
        for (a, b) in tok.merges.iter().zip(loaded.merges.iter()) {
            assert_eq!(a, b);
        }

        // Verify encode produces same results
        let text = "走れメロス";
        let ids_orig = tok.encode(text);
        let ids_loaded = loaded.encode(text);
        assert_eq!(ids_orig, ids_loaded);

        // Verify decode roundtrip
        let decoded = loaded.decode(&ids_loaded);
        assert_eq!(decoded, text);

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_save_load_with_newlines() {
        // Corpus with newlines to ensure they are handled correctly
        let corpus = "あ\nい\nあ\nい\nあ\nい\n";
        let tok = BpeTokenizer::train(corpus, 10);

        // Verify newline is in vocabulary
        assert!(tok.token_to_id.contains_key("\n"), "newline should be in vocab");

        let path = "/tmp/test_bpe_newline.txt";
        tok.save(path).unwrap();
        let loaded = BpeTokenizer::load(path).unwrap();

        // Vocab size must match exactly
        assert_eq!(tok.vocab_size(), loaded.vocab_size(),
            "vocab_size mismatch: original {} vs loaded {}", tok.vocab_size(), loaded.vocab_size());

        // Encode/decode roundtrip with newlines
        let text = "あ\nい";
        let ids_orig = tok.encode(text);
        let ids_loaded = loaded.encode(text);
        assert_eq!(ids_orig, ids_loaded);
        assert_eq!(loaded.decode(&ids_loaded), text);

        std::fs::remove_file(path).ok();
    }
}
