pub mod activation;
pub mod attention;
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

    let corpus = "hello world hello world hello world";
    let tokenizer = Tokenizer::from_corpus(corpus);

    println!("Vocab size: {}", tokenizer.vocab_size());

    let mut model = Transformer::rand(tokenizer.vocab_size(), 32, 4, 64, 2);

    // Generate before training
    let prompt = "hello";
    println!("\n--- Before training ---");
    let result = generate(&model, &tokenizer, prompt, 20, 0.01);
    println!("Prompt: \"{}\" -> \"{}\"", prompt, result);

    // Train
    println!("\n--- Training ---");
    let losses = train(&mut model, &tokenizer, corpus, 8, 100, 0.001);
    let first = losses[0];
    let last = *losses.last().unwrap();
    println!(
        "Steps: {}, Loss: {:.4} -> {:.4}",
        losses.len(),
        first,
        last
    );

    // Generate after training
    println!("\n--- After training ---");
    let result = generate(&model, &tokenizer, prompt, 20, 0.01);
    println!("Prompt: \"{}\" -> \"{}\"", prompt, result);

    println!("\n--- Sampling (temperature=0.8) ---");
    for i in 0..3 {
        let result = generate(&model, &tokenizer, prompt, 20, 0.8);
        println!("Output {}: \"{}\"", i + 1, result);
    }
}
