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
pub mod transformer;
pub mod transformer_block;

fn main() {
    use generate::generate;
    use tokenizer::Tokenizer;
    use transformer::Transformer;

    let corpus = "hello world this is a test of the transformer model";
    let tokenizer = Tokenizer::from_corpus(corpus);

    println!("Vocab size: {}", tokenizer.vocab_size());
    println!("Vocab: {:?}", &tokenizer.id_to_char[2..]); // skip PAD, UNK

    // Tiny random model (untrained)
    let model = Transformer::rand(tokenizer.vocab_size(), 32, 4, 64, 2);

    let prompt = "hello";
    println!("\nPrompt: \"{}\"", prompt);

    println!("\n--- Greedy (temperature=0.01) ---");
    let result = generate(&model, &tokenizer, prompt, 20, 0.01);
    println!("Output: \"{}\"", result);

    println!("\n--- Sampling (temperature=1.0) ---");
    for i in 0..3 {
        let result = generate(&model, &tokenizer, prompt, 20, 1.0);
        println!("Output {}: \"{}\"", i + 1, result);
    }
}
