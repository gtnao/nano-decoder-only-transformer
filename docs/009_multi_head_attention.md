# 009: Multi-Head Attention — 複数の視点で同時に注目する

## なぜ複数のヘッドが必要か

単一の Attention では、1つの「注目パターン」しか表現できない。しかし言語理解には複数の関係性を同時に捉える必要がある。

例えば「The cat that I saw yesterday sat on the mat」という文では：
- **構文的な関係**: 「cat」→「sat」（主語と動詞）
- **修飾関係**: 「cat」→「saw」（関係代名詞節）
- **近接関係**: 「on」→「mat」（前置詞と目的語）

Multi-Head Attention は、入力を複数の「ヘッド」に分割し、各ヘッドが異なる注目パターンを学習する。

## 計算の流れ

```
入力 x: [seq_len, d_model]

1. 射影:    Q = x @ Wq,  K = x @ Wk,  V = x @ Wv   (各 [seq_len, d_model])
2. ヘッド分割: d_k = d_model / n_heads
              Q, K, V を n_heads 個の [seq_len, d_k] に分割
3. 各ヘッドで Attention:
              head_i = Attention(Q_i, K_i, V_i)      (各 [seq_len, d_k])
4. 結合:     concat = [head_0 | head_1 | ... ]       [seq_len, d_model]
5. 出力射影: output = concat @ Wo                     [seq_len, d_model]
```

### なぜ射影するのか

直接 `x` を分割するのではなく、`Wq, Wk, Wv` で射影してから分割する。これにより各ヘッドが独自の「Q, K, V の作り方」を学習できる。

ヘッド0は構文関係に有用な射影を、ヘッド1は意味的な類似度に有用な射影を、それぞれ学ぶ。

### なぜ出力射影 Wo が必要か

各ヘッドの出力を単純に結合しただけでは、ヘッド間の情報が混ざらない。`Wo` による射影で全ヘッドの結果を統合する。

## ヘッドの分割と結合

`d_model = 8, n_heads = 2` のとき `d_k = 4`。

```
Q_full = [seq_len, 8]

Q_head0 = Q_full[:, 0:4]   → [seq_len, 4]
Q_head1 = Q_full[:, 4:8]   → [seq_len, 4]
```

各ヘッドの Attention 出力も `[seq_len, 4]` なので、結合すると `[seq_len, 8]` に戻る。

## 実装

```rust
pub fn forward(&self, x: &Tensor, use_causal_mask: bool) -> Tensor {
    let seq_len = x.shape[0];
    let d_model = x.shape[1];
    let d_k = d_model / self.n_heads;

    let q_full = self.wq.forward(x);
    let k_full = self.wk.forward(x);
    let v_full = self.wv.forward(x);

    let mask = if use_causal_mask {
        Some(causal_mask(seq_len))
    } else {
        None
    };

    let mut head_outputs = vec![0.0_f32; seq_len * d_model];

    for h in 0..self.n_heads {
        let offset = h * d_k;

        // Extract head h
        let mut q_h = vec![0.0_f32; seq_len * d_k];
        for s in 0..seq_len {
            for d in 0..d_k {
                q_h[s * d_k + d] = q_full.data[s * d_model + offset + d];
            }
        }
        // ... same for k_h, v_h

        let attn_out = scaled_dot_product_attention(&q_head, &k_head, &v_head, mask.as_ref());

        // Write back
        for s in 0..seq_len {
            for d in 0..d_k {
                head_outputs[s * d_model + offset + d] = attn_out.data[s * d_k + d];
            }
        }
    }

    let concat = Tensor::new(head_outputs, vec![seq_len, d_model]);
    self.wo.forward(&concat)
}
```

ヘッドの分割・結合は、row-major order のデータ上でオフセット計算により行う。各行（トークン）の `d_model` 次元を `d_k` ずつ区切って各ヘッドに割り当てる。

### 計算量

Single-Head で `d_model` 次元の Attention を1回行うのと、Multi-Head で `d_k` 次元の Attention を `n_heads` 回行うのは、総計算量がほぼ同じ。ヘッドを増やしてもコストが増えないのがこの設計の利点。

## テスト

```
running 6 tests
test multi_head_attention::tests::test_d_model_not_divisible_by_n_heads - should panic ... ok
test multi_head_attention::tests::test_forward_causal_shape ... ok
test multi_head_attention::tests::test_forward_deterministic ... ok
test multi_head_attention::tests::test_forward_shape ... ok
test multi_head_attention::tests::test_forward_single_head_equals_attention ... ok
test multi_head_attention::tests::test_rand_shapes ... ok
```

- **single_head_equals_attention**: 1ヘッド + 単位行列の射影 = 素の Attention と一致することを検証。MHA が正しく Attention を包含していることの証明
- **d_model_not_divisible**: `d_model` が `n_heads` で割り切れないときに panic
- **deterministic**: 同じ入力に対して同じ出力

## 次回

Feed-Forward Network を実装する。Attention で「どこに注目するか」を決めた後、FFN で各トークンの表現を変換する。
