# 022: Attention の backward

## 計算グラフ

Scaled Dot-Product Attention の forward は以下の5ステップ：

```
1. scores   = Q @ K^T
2. scaled   = scores / sqrt(d_k)
3. masked   = scaled + mask
4. weights  = softmax(masked)
5. output   = weights @ V
```

Backward はこれを逆順にたどる。

## 各ステップの逆伝播

### Step 5: output = weights @ V

行列積の逆伝播（Linear backward と同じパターン）：

```
d_weights = d_output @ V^T
d_V       = weights^T @ d_output
```

### Step 4: weights = softmax(masked)

Step 19 で実装した `softmax_backward` をそのまま使用：

```
d_scaled = softmax_backward(d_weights, weights)
```

### Step 3: masked = scaled + mask

mask は定数なので勾配はそのまま通過：

```
d_scaled = d_masked  (同じ)
```

### Step 2: scaled = scores * scale

スカラー倍の逆伝播：

```
d_scores = d_scaled * scale
```

### Step 1: scores = Q @ K^T

行列積の逆伝播：

```
d_Q = d_scores @ K
d_K = d_scores^T @ Q
```

`scores = Q @ K^T` の微分では、K^T に対する勾配は `Q^T @ d_scores` だが、K 自体の勾配が欲しいので転置して `d_K = d_scores^T @ Q`。

## 実装

```rust
pub fn scaled_dot_product_attention_backward(
    d_output: &Tensor, q: &Tensor, k: &Tensor, v: &Tensor,
    mask: Option<&Tensor>,
) -> (Tensor, Tensor, Tensor) {
    // Recompute forward intermediates
    let weights = softmax(Q @ K^T * scale + mask);

    // Step 5 backward
    let d_weights = d_output @ V^T;
    let d_v = weights^T @ d_output;

    // Step 4 backward
    let d_scores_scaled = softmax_backward(d_weights, weights);

    // Step 2 backward
    let d_scores = d_scores_scaled * scale;

    // Step 1 backward
    let d_q = d_scores @ K;
    let d_k = d_scores^T @ Q;

    (d_q, d_k, d_v)
}
```

Forward の中間結果（`weights`）を再計算している。本格的な実装ではキャッシュするが、シンプルさを優先。

## テスト

```
running 10 tests
test attention::tests::test_attention_backward_numerical_no_mask ... ok
test attention::tests::test_attention_backward_numerical_with_mask ... ok
test attention::tests::test_attention_backward_shapes ... ok
(+ 7 forward tests)
```

mask なし・ありの両方で、Q, K, V それぞれに対する勾配を数値微分で検証。

## 次回

Multi-Head Attention の backward を実装する。
