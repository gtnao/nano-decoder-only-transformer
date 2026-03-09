# 017: Linear の backward — 行列微分の基本

## Linear 層の復習

```
y = x @ W^T + b
```

- `x`: [batch, in_features]
- `W`: [out_features, in_features]
- `b`: [out_features]
- `y`: [batch, out_features]

## 勾配の導出

上流から `d_output = ∂L/∂y` が `[batch, out_features]` で渡されてくる。求めるのは3つ：

### d_input (∂L/∂x)

`y = x @ W^T` の `x` に関する微分。

```
d_input = d_output @ W     [batch, in_features]
```

直感: `W^T` を掛けて `in → out` に変換したので、戻すには `W`（転置の転置 = 元）を掛ける。

### d_weight (∂L/∂W)

```
d_weight = d_output^T @ x   [out_features, in_features]
```

`d_output` の各出力ニューロンの勾配と、`x` の各入力を外積的に掛け合わせたもの。

### d_bias (∂L/∂b)

バイアスは各行に同じ値を足しているので、バッチ方向に合計する。

```
d_bias = Σ_batch d_output   [out_features]
```

## 実装

```rust
pub fn backward(&self, d_output: &Tensor, input: &Tensor) -> (Tensor, Tensor, Tensor) {
    let d_input = d_output.matmul(&self.weight);
    let d_weight = d_output.transpose().matmul(input);

    let batch = d_output.shape[0];
    let out_features = d_output.shape[1];
    let mut d_bias_data = vec![0.0_f32; out_features];
    for b in 0..batch {
        for o in 0..out_features {
            d_bias_data[o] += d_output.data[b * out_features + o];
        }
    }
    let d_bias = Tensor::new(d_bias_data, vec![out_features]);

    (d_input, d_weight, d_bias)
}
```

全て行列積と合計で表現できる。forward と同様に `matmul` と `transpose` を組み合わせるだけ。

## 数値微分による検証

各パラメータを `±ε` ずらして forward を2回実行し、差分から数値的に勾配を近似する。解析的な勾配と一致することを確認。

## テスト

```
running 9 tests
test linear::tests::test_backward_numerical_gradient ... ok
test linear::tests::test_backward_shapes ... ok
(+ 7 forward tests)
```

## 次回

活性化関数（ReLU, GELU）の backward を実装する。
