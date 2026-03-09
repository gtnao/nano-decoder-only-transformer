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

fn main() {
    use bpe_tokenizer::BpeTokenizer;
    use generate::generate;
    use train::train_with_batch;
    use transformer::Transformer;

    // Load multiple works from Aozora Bunko
    let files = [
        "data/kumo_no_ito.txt",
        "data/rashomon.txt",
        "data/hana.txt",
        "data/hashire_merosu.txt",
        "data/chuumon.txt",
    ];
    let corpus: String = files
        .iter()
        .map(|f| std::fs::read_to_string(f).unwrap_or_else(|_| panic!("Failed to read {}", f)))
        .collect::<Vec<_>>()
        .join("\n");
    let corpus = corpus.trim();

    let bpe_vocab_size = 1000;
    let tokenizer = BpeTokenizer::train(corpus, bpe_vocab_size);

    println!("Corpus: {} chars ({} files)", corpus.chars().count(), files.len());
    println!("Vocab size: {} (BPE)", tokenizer.vocab_size());

    let d_model = 128;
    let n_heads = 4;
    let d_ff = 512;
    let n_layers = 4;
    let seq_len = 128;
    let epochs = 30;
    let lr = 0.001;
    let dropout = 0.1;

    let batch_size = 8;

    println!("Model: d_model={}, n_heads={}, d_ff={}, n_layers={}, dropout={}", d_model, n_heads, d_ff, n_layers, dropout);
    println!("Training: seq_len={}, epochs={}, lr={}, batch_size={}", seq_len, epochs, lr, batch_size);

    let mut model = Transformer::rand_with_dropout(tokenizer.vocab_size(), d_model, n_heads, d_ff, n_layers, dropout);

    // Generate before training
    let prompt = "ある日";
    println!("\n--- Before training ---");
    let result = generate(&model, &tokenizer, prompt, 50, 0.01);
    println!("\"{}\"", result);

    // Train
    println!("\n--- Training ---");
    let start = std::time::Instant::now();
    let losses = train_with_batch(&mut model, &tokenizer, corpus, seq_len, epochs, lr, batch_size);
    let elapsed = start.elapsed();
    let n = losses.len();
    println!("Steps: {} ({:.1}s)", n, elapsed.as_secs_f64());
    // Print loss at intervals
    for i in [0, n/4, n/2, 3*n/4, n-1] {
        println!("  step {:>4}: loss={:.4}", i, losses[i]);
    }

    // Generate after training
    println!("\n--- After training ---");
    for prompt in &["ある日の", "メロスは", "下人は", "蜘蛛の糸", "二人の若い紳士"] {
        let result = generate(&model, &tokenizer, prompt, 100, 0.8);
        println!("prompt: \"{}\"", prompt);
        println!("  → {}", result);
        println!();
    }
}
