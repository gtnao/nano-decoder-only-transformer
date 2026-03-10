pub mod activation;
pub mod attention;
pub mod bpe_tokenizer;
pub mod dropout;
pub mod embedding;
pub mod feed_forward;
pub mod generate;
pub mod layer_norm;
pub mod loss;
pub mod linear;
pub mod multi_head_attention;
pub mod optimizer;
pub mod positional_encoding;
pub mod softmax;
pub mod tensor;
pub mod tokenizer;
pub mod train;
pub mod transformer;
pub mod transformer_block;

const MODEL_PATH: &str = "model.bin";
const TOKENIZER_PATH: &str = "tokenizer.txt";
const DEFAULT_CONFIG_PATH: &str = "train_config.txt";

/// Simple key=value config parser
fn load_config(path: &str) -> std::collections::HashMap<String, String> {
    let mut map = std::collections::HashMap::new();
    if let Ok(content) = std::fs::read_to_string(path) {
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if let Some((key, val)) = line.split_once('=') {
                map.insert(key.trim().to_string(), val.trim().to_string());
            }
        }
    }
    map
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("help");

    match mode {
        "train" => {
            let config_path = args.get(2).map(|s| s.as_str()).unwrap_or(DEFAULT_CONFIG_PATH);
            cmd_train(config_path);
        }
        "generate" => cmd_generate(&args[2..]),
        "retokenize" => {
            let config_path = args.get(2).map(|s| s.as_str()).unwrap_or(DEFAULT_CONFIG_PATH);
            cmd_retokenize(config_path);
        }
        _ => {
            eprintln!("Usage:");
            eprintln!("  cargo run --release -- train [config_file]");
            eprintln!("  cargo run --release -- generate \"prompt text\" [temperature]");
            eprintln!("  cargo run --release -- retokenize [config_file]");
        }
    }
}

/// Load corpus from data_dir with optional per-file truncation.
fn load_corpus(data_dir: &str, max_chars: usize) -> (String, usize) {
    let mut files: Vec<_> = std::fs::read_dir(data_dir)
        .unwrap_or_else(|e| panic!("Failed to read {}: {}", data_dir, e))
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            if path.extension().map(|e| e == "txt").unwrap_or(false) {
                Some(path)
            } else {
                None
            }
        })
        .collect();
    files.sort();
    let n_files = files.len();

    let corpus: String = files
        .iter()
        .map(|f| {
            let text = std::fs::read_to_string(f).unwrap_or_else(|_| panic!("Failed to read {:?}", f));
            if max_chars > 0 && text.chars().count() > max_chars {
                text.chars().take(max_chars).collect::<String>()
            } else {
                text
            }
        })
        .collect::<Vec<_>>()
        .join("\n");
    let corpus = corpus.trim().to_string();
    (corpus, n_files)
}

fn cmd_train(config_path: &str) {
    use bpe_tokenizer::BpeTokenizer;
    use generate::generate;
    use train::train_with_batch;
    use transformer::Transformer;

    let cfg = load_config(config_path);
    let get = |key: &str, default: &str| -> String {
        cfg.get(key).cloned().unwrap_or_else(|| default.to_string())
    };

    let d_model: usize = get("d_model", "128").parse().unwrap();
    let n_heads: usize = get("n_heads", "4").parse().unwrap();
    let d_ff: usize = get("d_ff", "512").parse().unwrap();
    let n_layers: usize = get("n_layers", "4").parse().unwrap();
    let seq_len: usize = get("seq_len", "128").parse().unwrap();
    let epochs: usize = get("epochs", "30").parse().unwrap();
    let lr: f32 = get("lr", "0.001").parse().unwrap();
    let dropout: f32 = get("dropout", "0.1").parse().unwrap();
    let batch_size: usize = get("batch_size", "8").parse().unwrap();
    let bpe_vocab_size: usize = get("bpe_vocab_size", "1000").parse().unwrap();
    let data_dir = get("data_dir", "data");
    let max_chars: usize = get("max_chars_per_file", "0").parse().unwrap();

    println!("Config: {}", config_path);

    let (corpus, n_files) = load_corpus(&data_dir, max_chars);

    // Train BPE tokenizer
    let tokenizer = BpeTokenizer::train(&corpus, bpe_vocab_size);

    println!("Corpus: {} chars ({} files)", corpus.chars().count(), n_files);
    println!("Vocab size: {} (BPE)", tokenizer.vocab_size());
    println!("Model: d_model={}, n_heads={}, d_ff={}, n_layers={}, dropout={}", d_model, n_heads, d_ff, n_layers, dropout);
    println!("Training: seq_len={}, epochs={}, lr={}, batch_size={}", seq_len, epochs, lr, batch_size);

    let mut model = Transformer::rand_with_dropout(tokenizer.vocab_size(), d_model, n_heads, d_ff, n_layers, dropout);

    // Train
    println!("\n--- Training ---");
    let start = std::time::Instant::now();
    let losses = train_with_batch(&mut model, &tokenizer, &corpus, seq_len, epochs, lr, batch_size);
    let elapsed = start.elapsed();
    let n = losses.len();
    println!("Steps: {} ({:.1}s)", n, elapsed.as_secs_f64());
    for i in [0, n/4, n/2, 3*n/4, n-1] {
        println!("  step {:>4}: loss={:.4}", i, losses[i]);
    }

    // Save model and tokenizer
    model.save(MODEL_PATH).expect("Failed to save model");
    tokenizer.save(TOKENIZER_PATH).expect("Failed to save tokenizer");
    println!("\nSaved model to {} and tokenizer to {}", MODEL_PATH, TOKENIZER_PATH);

    // Generate samples
    println!("\n--- After training ---");
    for prompt in &["ある日の", "メロスは", "下人は", "蜘蛛の糸", "二人の若い紳士"] {
        let result = generate(&model, &tokenizer, prompt, 100, 0.8);
        println!("prompt: \"{}\"", prompt);
        println!("  → {}", result);
        println!();
    }
}

fn cmd_retokenize(config_path: &str) {
    use bpe_tokenizer::BpeTokenizer;
    use transformer::Transformer;

    let cfg = load_config(config_path);
    let get = |key: &str, default: &str| -> String {
        cfg.get(key).cloned().unwrap_or_else(|| default.to_string())
    };

    let bpe_vocab_size: usize = get("bpe_vocab_size", "1000").parse().unwrap();
    let data_dir = get("data_dir", "data");
    let max_chars: usize = get("max_chars_per_file", "0").parse().unwrap();

    // Load model to verify vocab_size compatibility
    let model = Transformer::load(MODEL_PATH)
        .unwrap_or_else(|e| panic!("Failed to load {}: {}. Run 'train' first.", MODEL_PATH, e));
    let model_vocab = model.token_embedding.weight.shape[0];

    let (corpus, n_files) = load_corpus(&data_dir, max_chars);

    // Retrain tokenizer from corpus
    let tokenizer = BpeTokenizer::train(&corpus, bpe_vocab_size);

    println!("Corpus: {} chars ({} files)", corpus.chars().count(), n_files);
    println!("Model vocab_size: {}", model_vocab);
    println!("Tokenizer vocab_size: {}", tokenizer.vocab_size());

    if tokenizer.vocab_size() != model_vocab {
        eprintln!(
            "WARNING: vocab_size mismatch! Model expects {} but tokenizer has {}.",
            model_vocab, tokenizer.vocab_size()
        );
        eprintln!("The model may not work correctly. Consider retraining.");
    } else {
        println!("vocab_size matches.");
    }

    tokenizer.save(TOKENIZER_PATH).expect("Failed to save tokenizer");
    println!("Saved tokenizer to {}", TOKENIZER_PATH);
}

fn cmd_generate(args: &[String]) {
    use bpe_tokenizer::BpeTokenizer;
    use generate::generate;
    use transformer::Transformer;

    let prompt = args.first().map(|s| s.as_str()).unwrap_or("ある日");
    let temperature: f32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(0.8);

    // Load model and tokenizer
    let model = Transformer::load(MODEL_PATH)
        .unwrap_or_else(|e| panic!("Failed to load {}: {}. Run 'train' first.", MODEL_PATH, e));
    let tokenizer = BpeTokenizer::load(TOKENIZER_PATH)
        .unwrap_or_else(|e| panic!("Failed to load {}: {}. Run 'train' first.", TOKENIZER_PATH, e));

    println!("Loaded model from {} (vocab_size={})", MODEL_PATH, tokenizer.vocab_size());

    let result = generate(&model, &tokenizer, prompt, 200, temperature);
    println!("{}", result);
}
