# 024: Feed-Forward Network の backward

## FFN の復習

```
hidden    = Linear1(x)       [seq_len, d_ff]
activated = GELU(hidden)     [seq_len, d_ff]
output    = Linear2(activated) [seq_len, d_model]
```

## Backward の流れ

3つのステップを逆順にたどるだけ。各ステップの backward は既に実装済み。

### Step 3: Linear2 の逆伝播

```
(d_activated, d_l2_weight, d_l2_bias) = linear2.backward(d_output, activated)
```

### Step 2: GELU の逆伝播

```
d_hidden = gelu_backward(d_activated, hidden)
```

GELU backward は Step 18 で実装済み。forward の入力 `hidden` が必要。

### Step 1: Linear1 の逆伝播

```
(d_x, d_l1_weight, d_l1_bias) = linear1.backward(d_hidden, x)
```

## 実装

```rust
pub fn backward(&self, d_output: &Tensor, x: &Tensor) -> FFNGradients {
    // Recompute forward intermediates
    let hidden = self.linear1.forward(x);
    let activated = gelu(&hidden);

    let (d_activated, d_l2_weight, d_l2_bias) = self.linear2.backward(d_output, &activated);
    let d_hidden = gelu_backward(&d_activated, &hidden);
    let (d_x, d_l1_weight, d_l1_bias) = self.linear1.backward(&d_hidden, x);

    FFNGradients { d_x, d_l1_weight, d_l1_bias, d_l2_weight, d_l2_bias }
}
```

既存の backward 関数を組み合わせるだけなので、実装はシンプル。forward の中間値（`hidden`, `activated`）は再計算している。

## テスト

```
running 8 tests
test feed_forward::tests::test_backward_numerical_d_l1_weight ... ok
test feed_forward::tests::test_backward_numerical_d_x ... ok
test feed_forward::tests::test_backward_shapes ... ok
(+ 5 forward tests)
```

`d_x` と `d_l1_weight` の両方を数値微分で検証。

## 次回

Transformer Block の backward を実装する。
