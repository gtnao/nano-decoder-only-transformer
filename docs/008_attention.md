# 008: Scaled Dot-Product Attention — Transformer の心臓部

## Attention とは

Attention は「入力系列の中で、今どの部分に注目すべきか」を動的に決定する仕組みである。

例えば「The cat sat on the mat because it was tired」という文で「it」の意味を理解するには「cat」に注目する必要がある。Attention はこの「どこに注目するか」をデータから自動的に学習する。

## Q, K, V の直感

Attention は3つの入力を取る：

- **Query (Q)**: 「何を探しているか」— 問い合わせ
- **Key (K)**: 「何を持っているか」— 各トークンのラベル
- **Value (V)**: 「実際の中身」— 各トークンが提供する情報

辞書に例えると、Q で検索ワードを投げ、K と照合して関連度を測り、関連度に応じて V を重み付き平均で取り出す。

## 計算の流れ

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

4つのステップに分解できる：

### 1. スコア計算: `QK^T`

Q と K の内積で、全トークンペア間の類似度を計算する。

```
Q: [seq_len, d_k]
K: [seq_len, d_k]
QK^T: [seq_len, seq_len]
```

結果は seq_len × seq_len の行列で、`[i, j]` は「トークン i がトークン j にどれだけ注目するか」のスコア。

### 2. スケーリング: `/ √d_k`

内積は次元数 `d_k` が大きいと値が大きくなる。大きすぎるとSoftmax の出力がほぼ one-hot になり、勾配が消失する。

`√d_k` で割ることで、`d_k` に依存しないスケールに正規化する。

### 3. Softmax

スコアを確率分布に変換する（各行の合計が1になる）。

### 4. 重み付き平均: `× V`

Softmax の出力（注目度の確率分布）を使って V の重み付き平均を取る。

```
softmax 出力: [seq_len, seq_len]
V:            [seq_len, d_v]
結果:         [seq_len, d_v]
```

## Causal Mask

Decoder-Only Transformer では、各トークンは自分より**前**のトークンにしか注目できない。未来の情報が漏れると、次トークン予測の意味がなくなるため。

Causal Mask は上三角部分を `-∞` にした行列。

```
[[  0, -∞, -∞],
 [  0,   0, -∞],
 [  0,   0,   0]]
```

これをスコアに加算すると、未来のトークンのスコアが `-∞` になり、Softmax を通すと確率が 0 になる。

```
exp(-∞) = 0
```

例えばトークン0（1行目）は：
- 自分自身(col=0)は見える: スコア + 0 = そのまま
- トークン1,2(col=1,2)は見えない: スコア + (-∞) = -∞ → Softmax で 0

## 実装

```rust
pub fn scaled_dot_product_attention(
    q: &Tensor, k: &Tensor, v: &Tensor,
    mask: Option<&Tensor>,
) -> Tensor {
    let d_k = q.shape[1] as f32;
    let scale = 1.0 / d_k.sqrt();

    // scores = Q @ K^T
    let kt = k.transpose();
    let mut scores = q.matmul(&kt);

    // scale
    for val in scores.data.iter_mut() {
        *val *= scale;
    }

    // apply mask
    if let Some(m) = mask {
        scores = scores.add(m);
    }

    // softmax -> weighted sum
    let weights = softmax(&scores);
    weights.matmul(v)
}
```

これまでに実装した `transpose`、`matmul`、`add`、`softmax` を組み合わせるだけで Attention が完成する。

## テスト

```
running 7 tests
test attention::tests::test_attention_causal_first_token ... ok
test attention::tests::test_attention_causal_second_token ... ok
test attention::tests::test_attention_identity_like ... ok
test attention::tests::test_attention_no_mask_shape ... ok
test attention::tests::test_attention_scaling ... ok
test attention::tests::test_causal_mask_shape ... ok
test attention::tests::test_causal_mask_values ... ok
```

- **causal_mask_values**: 下三角が 0、上三角が `-∞` であることを全要素検証
- **identity_like**: Q=K のとき均等な注目 → V の平均が出力
- **causal_first_token**: マスク付きで1番目のトークンは自分自身の V のみ出力
- **causal_second_token**: 2番目のトークンは [0,1] にのみ注目
- **scaling**: `1/√d_k` スケーリングにより分布が鋭くなりすぎない

## 次回

Multi-Head Attention を実装する。1つの Attention を複数の「ヘッド」に分割して並列実行し、異なる観点からの注目パターンを同時に捉える。
