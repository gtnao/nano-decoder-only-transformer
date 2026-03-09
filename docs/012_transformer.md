# 012: Transformer全体 — 全部品を接続して完成

## 全体像

Decoder-Only Transformer の全体を接続する。トークンID列を入力として受け取り、各位置で「次にどのトークンが来るか」の確率分布（logits）を出力する。

```
token_ids: [0, 5, 3, 7]

         ┌─────────────────┐
         │ Token Embedding  │  token_id → ベクトル
         └────────┬────────┘
                  │ [seq_len, d_model]
                  +  ← Positional Encoding を加算
                  │
         ┌────────┴────────┐
         │ Transformer     │
         │ Block × N       │  Attention + FFN を N回繰り返す
         └────────┬────────┘
                  │ [seq_len, d_model]
         ┌────────┴────────┐
         │ Final LayerNorm │
         └────────┬────────┘
                  │ [seq_len, d_model]
         ┌────────┴────────┐
         │ LM Head (Linear)│  d_model → vocab_size に射影
         └────────┬────────┘
                  │
                  ▼
         logits: [seq_len, vocab_size]
```

各位置の出力 `logits[i]` は、位置 `i` の次に来るトークンの非正規化確率。Softmax を通すと確率分布になる。

## 各部品の接続

### 1. Token Embedding + Positional Encoding

```rust
let tok_emb = self.token_embedding.forward(token_ids);  // [seq_len, d_model]
let pos_enc = positional_encoding(seq_len, d_model);     // [seq_len, d_model]
let mut x = tok_emb.add(&pos_enc);                       // [seq_len, d_model]
```

トークンの意味情報（Embedding）と位置情報（Positional Encoding）を加算して合成する。加算であって結合（concatenation）ではない点に注意。同じ `d_model` 次元の空間で混ぜ合わせる。

### 2. N × Transformer Block

```rust
for block in &self.blocks {
    x = block.forward(&x);   // [seq_len, d_model] → [seq_len, d_model]
}
```

入出力の shape が同じなので、ループで順番に通すだけ。各 Block は前の Block の出力を入力として受け取る。層が深くなるほど、より抽象的な表現が獲得される。

### 3. Final LayerNorm + LM Head

```rust
let x = self.ln_final.forward(&x);   // [seq_len, d_model]
self.lm_head.forward(&x)             // [seq_len, vocab_size]
```

Pre-LN 構成では最後の Block の出力はまだ正規化されていないため、Final LayerNorm が必要。その後 LM Head（Linear 層）で `d_model` 次元から `vocab_size` 次元に射影し、各トークンのスコア（logits）を得る。

## パラメータ数の概算

`vocab_size=V, d_model=D, n_heads=H, d_ff=F, n_layers=L` のとき：

| 部品 | パラメータ数 |
|---|---|
| Token Embedding | V × D |
| 各 Block の MHA | 4 × D² (Wq, Wk, Wv, Wo) |
| 各 Block の FFN | 2 × D × F |
| 各 Block の LN × 2 | 4 × D |
| Final LN | 2 × D |
| LM Head | V × D |

GPT-2 small の場合（V=50257, D=768, H=12, F=3072, L=12）：約1.24億パラメータ。

## 実装の全コード

```rust
pub fn forward(&self, token_ids: &[usize]) -> Tensor {
    let seq_len = token_ids.len();
    let d_model = self.token_embedding.weight.shape[1];

    let tok_emb = self.token_embedding.forward(token_ids);
    let pos_enc = positional_encoding(seq_len, d_model);
    let mut x = tok_emb.add(&pos_enc);

    for block in &self.blocks {
        x = block.forward(&x);
    }

    let x = self.ln_final.forward(&x);
    self.lm_head.forward(&x)
}
```

これまでの12ステップで実装した部品が全て使われている。

## テスト

```
running 6 tests
test transformer::tests::test_forward_deterministic ... ok
test transformer::tests::test_forward_different_inputs_differ ... ok
test transformer::tests::test_forward_logits_are_finite ... ok
test transformer::tests::test_forward_shape ... ok
test transformer::tests::test_forward_single_token ... ok
test transformer::tests::test_rand_structure ... ok
```

- **shape**: 出力が `[seq_len, vocab_size]` であること
- **logits_are_finite**: NaN や Infinity が出ていない（数値安定性）
- **rand_structure**: 各部品の shape が正しいこと

## 次回

トークナイザ（簡易版）を実装する。文字列をトークンID列に変換し、モデルに入力できるようにする。
