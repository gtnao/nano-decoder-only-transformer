# 026: Transformer 全体の backward

## 計算グラフ

```
token_ids → Embedding → (+pos_enc) → Block1 → Block2 → ... → LayerNorm → Linear → logits
```

## Backward の流れ

### Step 1: LM Head の逆伝播

```
(d_after_ln, d_lm_weight, d_lm_bias) = lm_head.backward(d_logits, x_after_ln)
```

### Step 2: Final LayerNorm の逆伝播

```
(d_x, d_ln_gamma, d_ln_beta) = ln_final.backward(d_after_ln, x_before_ln)
```

### Step 3: Blocks の逆伝播（逆順）

```
for block in blocks.rev():
    grads = block.backward(d_x, block_input)
    d_x = grads.d_x
```

各ブロックの backward が内部で MHA, FFN, LayerNorm の勾配をすべて計算して返す。

### Step 4: Embedding の逆伝播

Positional encoding は固定（学習パラメータではない）ので、勾配はそのまま embedding に流れる。

```
d_embedding_weight = embedding.backward(d_x, token_ids)
```

## Forward の再計算

Backward では forward の中間値（各ブロックの入力）が必要。本実装では forward をもう一度実行して `block_inputs` を保存している。

```rust
let mut block_inputs = Vec::new();
let mut x = x0.clone();
for block in &self.blocks {
    block_inputs.push(x.clone());
    x = block.forward(&x);
}
```

本格的な実装ではメモリ効率のためにチェックポイント再計算（gradient checkpointing）を使うこともあるが、今回はシンプルに全中間値を保持。

## 勾配の構造体

```rust
pub struct TransformerGradients {
    pub d_embedding_weight: Tensor,
    pub block_grads: Vec<TransformerBlockGradients>,
    pub d_ln_final_gamma: Tensor,
    pub d_ln_final_beta: Tensor,
    pub d_lm_head_weight: Tensor,
    pub d_lm_head_bias: Tensor,
}
```

`block_grads` は各ブロックの全パラメータ勾配（MHA×4 Linear + LN×2）を含む。これにより、全パラメータをまとめて更新できる。

## テスト

```
running 9 tests
test transformer::tests::test_backward_numerical_d_lm_head_weight ... ok
test transformer::tests::test_backward_gradients_nonzero ... ok
test transformer::tests::test_backward_shapes ... ok
(+ 6 forward tests)
```

`d_lm_head_weight` を数値微分で検証。全勾配が非ゼロであることも確認。

## 次回

パラメータ更新（SGD / Adam）を実装する。
