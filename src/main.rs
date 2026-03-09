pub mod activation;
pub mod attention;
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
    use generate::generate;
    use tokenizer::Tokenizer;
    use train::train;
    use transformer::Transformer;

    let corpus = std::fs::read_to_string("data/kumo_no_ito.txt")
        .expect("Failed to read data/kumo_no_ito.txt");
    let corpus = corpus.trim();
    let tokenizer = Tokenizer::from_corpus(corpus);

    println!("Corpus: {} chars", corpus.chars().count());
    println!("Vocab size: {}", tokenizer.vocab_size());

    let d_model = 64;
    let n_heads = 4;
    let d_ff = 128;
    let n_layers = 2;
    let seq_len = 64;
    let epochs = 50;
    let lr = 0.001;
    let dropout = 0.1;

    println!("Model: d_model={}, n_heads={}, d_ff={}, n_layers={}, dropout={}", d_model, n_heads, d_ff, n_layers, dropout);
    println!("Training: seq_len={}, epochs={}, lr={}", seq_len, epochs, lr);

    let mut model = Transformer::rand_with_dropout(tokenizer.vocab_size(), d_model, n_heads, d_ff, n_layers, dropout);

    // Generate before training
    let prompt = "極楽";
    println!("\n--- Before training ---");
    let result = generate(&model, &tokenizer, prompt, 50, 0.01);
    println!("\"{}\"", result);

    // Train
    println!("\n--- Training ---");
    let losses = train(&mut model, &tokenizer, corpus, seq_len, epochs, lr);
    let n = losses.len();
    println!("Steps: {}", n);
    // Print loss at intervals
    for i in [0, n/4, n/2, 3*n/4, n-1] {
        println!("  step {:>4}: loss={:.4}", i, losses[i]);
    }

    // Generate after training
    println!("\n--- After training ---");
    for prompt in &["極楽", "地獄", "蜘蛛の糸"] {
        let result = generate(&model, &tokenizer, prompt, 80, 0.8);
        println!("\"{}\"", result);
        println!();
    }
}
