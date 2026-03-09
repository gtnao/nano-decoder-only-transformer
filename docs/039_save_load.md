# 039: モデルの保存・読み込みとCLIモード分離

## 背景

毎回学習するのは非効率。学習済みモデルを保存し、生成時に読み込む仕組みが必要。

## 実装

### モデルの保存 (transformer.rs)

バイナリ形式でパラメータを保存:

```rust
pub fn save(&self, path: &str) -> std::io::Result<()> {
    // Header: [vocab_size, d_model, n_heads, d_ff, n_layers] as u32
    // Body: all f32 parameters in deterministic order
}

pub fn load(path: &str) -> std::io::Result<Self> {
    // Read header, create model, load parameters
}
```

パラメータの直列化順序は `param_sizes()` / `apply_gradients()` と同一:
1. Embedding weight
2. 各ブロック: MHA (Wq, Wk, Wv, Wo weight/bias) → FFN (L1, L2 weight/bias) → LayerNorm (gamma, beta) x2
3. Final LayerNorm (gamma, beta)
4. LM head (weight, bias)

### BPE トークナイザの保存 (bpe_tokenizer.rs)

テキスト形式:

```
マージルール1_a\tマージルール1_b
マージルール2_a\tマージルール2_b
...
---
<PAD>
<UNK>
あ
い
...
```

### CLI モード分離 (main.rs)

```bash
# 学習 → model.bin, tokenizer.txt を保存
cargo run --release -- train

# 生成 → 保存済みモデルを読み込んで生成
cargo run --release -- generate "プロンプト" [temperature]
```

## テスト

```
183 tests passed (+1 new)
- test_save_load_roundtrip (transformer): モデルの保存→読み込み→forward結果の一致を確認
- test_save_load_roundtrip (bpe_tokenizer): トークナイザの保存→読み込み→encode結果の一致を確認
```
