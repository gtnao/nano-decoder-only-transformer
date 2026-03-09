# 018: 活性化関数の backward

## ReLU の backward

ReLU は `max(0, x)` なので、微分は単純なステップ関数：

```
d(ReLU)/dx = 1  (x > 0)
           = 0  (x ≤ 0)
```

勾配は「正の入力はそのまま通す、負の入力は遮断する」というゲート。forward と全く同じ判定。

```rust
pub fn relu_backward(d_output: &Tensor, x: &Tensor) -> Tensor {
    d_output.iter().zip(x.iter())
        .map(|(&d, &v)| if v > 0.0 { d } else { 0.0 })
        .collect()
}
```

## GELU の backward

GELU は `0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715x³)))` で、微分は積の法則を適用：

```
f(x) = 0.5 * x
g(x) = 1 + tanh(inner)
inner = √(2/π) * (x + 0.044715 * x³)

d(GELU)/dx = 0.5 * (1 + tanh(inner)) + 0.5 * x * sech²(inner) * d(inner)/dx
```

ここで `d(inner)/dx = √(2/π) * (1 + 3 * 0.044715 * x²)` で、`sech²(z) = 1 - tanh²(z)`。

ReLU に比べて複雑だが、forward で計算した `tanh(inner)` の値を再利用できる。

## 数値微分による検証

GELU の微分は式が複雑なので、数値微分 `(GELU(x+ε) - GELU(x-ε)) / 2ε` との一致を検証して実装の正しさを保証している。

## テスト

```
running 14 tests
test activation::tests::test_gelu_backward_numerical ... ok
test activation::tests::test_gelu_backward_scales_with_d_output ... ok
test activation::tests::test_relu_backward_mixed ... ok
test activation::tests::test_relu_backward_negative ... ok
test activation::tests::test_relu_backward_positive ... ok
(+ 9 forward tests)
```

## 次回

Softmax の backward を実装する。
