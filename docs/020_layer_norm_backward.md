# 020: Layer Normalization の backward

## LayerNorm の復習

```
x_hat = (x - mean) / sqrt(var + eps)
output = gamma * x_hat + beta
```

backward では `d_input`, `d_gamma`, `d_beta` の3つを求める。

## d_gamma と d_beta

これらは単純。forward の最後が `gamma * x_hat + beta` なので：

```
d_gamma[i] = Σ_groups d_output[i] * x_hat[i]
d_beta[i]  = Σ_groups d_output[i]
```

各グループ（行）の対応する要素を合計するだけ。

## d_input

これが LayerNorm backward の最も複雑な部分。正規化は `mean` と `var` を経由しており、全要素が互いに影響し合う。

導出のステップ：

1. `dy_hat[i] = d_output[i] * gamma[i]` — gamma の寄与を外す
2. 正規化の逆伝播：

```
d_input[i] = (1/sqrt(var+eps)) * (dy_hat[i] - mean(dy_hat) - x_hat[i] * mean(dy_hat * x_hat))
```

### なぜ3項になるか

`x_hat = (x - mean) / std` の微分では：
- `dy_hat[i]`: 直接の寄与
- `- mean(dy_hat)`: mean を引くことの寄与（全要素に影響）
- `- x_hat[i] * mean(dy_hat * x_hat)`: var の計算の寄与（全要素に影響）

## 実装

```rust
pub fn backward(&self, d_output: &Tensor, x: &Tensor) -> (Tensor, Tensor, Tensor) {
    // ... per group:
    let x_hat: Vec<f32> = group.iter().map(|&v| (v - mean) * std_inv).collect();

    // d_gamma, d_beta
    for i in 0..last_dim {
        d_gamma[i] += dy[i] * x_hat[i];
        d_beta[i] += dy[i];
    }

    // d_input
    let dy_hat: Vec<f32> = (0..last_dim).map(|i| dy[i] * self.gamma.data[i]).collect();
    let mean_dy_hat = dy_hat.iter().sum::<f32>() / n;
    let mean_dy_hat_x_hat: f32 = dy_hat.iter().zip(x_hat.iter())
        .map(|(&d, &x)| d * x).sum::<f32>() / n;

    for i in 0..last_dim {
        d_input[start + i] = std_inv * (dy_hat[i] - mean_dy_hat - x_hat[i] * mean_dy_hat_x_hat);
    }
}
```

forward で計算した `mean`, `var`, `std_inv`, `x_hat` を再計算している。本格的な実装ではこれらを forward 時にキャッシュするが、今回はシンプルさを優先。

## テスト

```
running 9 tests
test layer_norm::tests::test_backward_numerical_d_gamma ... ok
test layer_norm::tests::test_backward_numerical_d_input ... ok
test layer_norm::tests::test_backward_shapes ... ok
(+ 6 forward tests)
```

`d_input` と `d_gamma` の両方を数値微分で検証。gamma を非デフォルト値 `[1, 2, 0.5]` にしたケースでも正しく動作することを確認。

## 次回

Embedding の backward を実装する。
