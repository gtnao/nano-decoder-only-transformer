/// Simple character-level tokenizer.
/// Each unique character gets an ID. Special tokens: <PAD>=0, <UNK>=1.
pub struct Tokenizer {
    pub char_to_id: std::collections::HashMap<char, usize>,
    pub id_to_char: Vec<char>,
}

impl Tokenizer {
    /// Build vocabulary from a corpus string.
    pub fn from_corpus(corpus: &str) -> Self {
        // Collect unique chars, sorted for deterministic ordering
        let mut chars: Vec<char> = corpus.chars().collect::<std::collections::BTreeSet<_>>().into_iter().collect();
        // id_to_char: [PAD_placeholder, UNK_placeholder, ...corpus chars...]
        let mut id_to_char = vec!['\0', '\u{FFFD}'];
        id_to_char.append(&mut chars);

        let char_to_id: std::collections::HashMap<char, usize> = id_to_char
            .iter()
            .enumerate()
            .map(|(i, &c)| (c, i))
            .collect();

        Self { char_to_id, id_to_char }
    }

    pub fn vocab_size(&self) -> usize {
        self.id_to_char.len()
    }

    /// Encode a string into token IDs.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let unk_id = 1;
        text.chars()
            .map(|c| *self.char_to_id.get(&c).unwrap_or(&unk_id))
            .collect()
    }

    /// Decode token IDs back into a string.
    pub fn decode(&self, ids: &[usize]) -> String {
        ids.iter()
            .map(|&id| {
                if id < self.id_to_char.len() {
                    self.id_to_char[id]
                } else {
                    self.id_to_char[1] // UNK
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_corpus() {
        let tok = Tokenizer::from_corpus("abc");
        // <PAD>, <UNK>, a, b, c => 5
        assert_eq!(tok.vocab_size(), 5);
    }

    #[test]
    fn test_from_corpus_dedup() {
        let tok = Tokenizer::from_corpus("aabbcc");
        assert_eq!(tok.vocab_size(), 5); // still 5: <PAD>, <UNK>, a, b, c
    }

    #[test]
    fn test_encode_basic() {
        let tok = Tokenizer::from_corpus("abc");
        let ids = tok.encode("abc");
        assert_eq!(ids.len(), 3);
        // each char should map to a unique ID >= 2
        assert!(ids.iter().all(|&id| id >= 2));
        assert_ne!(ids[0], ids[1]);
        assert_ne!(ids[1], ids[2]);
    }

    #[test]
    fn test_encode_unknown_char() {
        let tok = Tokenizer::from_corpus("abc");
        let ids = tok.encode("axz");
        // 'x' and 'z' are not in vocab => UNK=1
        assert_eq!(ids[1], 1);
        assert_eq!(ids[2], 1);
    }

    #[test]
    fn test_decode_roundtrip() {
        let tok = Tokenizer::from_corpus("hello world");
        let text = "hello";
        let ids = tok.encode(text);
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_decode_special_tokens() {
        let tok = Tokenizer::from_corpus("abc");
        // PAD=0 decodes to some placeholder, UNK=1 decodes to placeholder
        let decoded = tok.decode(&[0, 1]);
        assert_eq!(decoded.chars().count(), 2); // 2 chars
    }

    #[test]
    fn test_encode_empty() {
        let tok = Tokenizer::from_corpus("abc");
        let ids = tok.encode("");
        assert!(ids.is_empty());
    }

    #[test]
    fn test_corpus_with_spaces_and_newlines() {
        let tok = Tokenizer::from_corpus("a b\nc");
        let ids = tok.encode("a b\nc");
        assert_eq!(ids.len(), 5);
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, "a b\nc");
    }

    #[test]
    fn test_sorted_vocab() {
        // Vocabulary order should be deterministic (sorted)
        let tok1 = Tokenizer::from_corpus("cba");
        let tok2 = Tokenizer::from_corpus("abc");
        assert_eq!(tok1.encode("abc"), tok2.encode("abc"));
    }
}
