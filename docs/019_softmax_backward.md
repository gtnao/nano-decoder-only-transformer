# 019: Softmax の backward

## Softmax の微分

Softmax 出力を `s_i = softmax(x)_i` とする。ヤコビ行列の要素は：

```
∂s_i/∂x_j = s_i * (δ_ij - s_j)
```

`δ_ij` はクロネッカーのデルタ（i=j なら 1、それ以外 0）。

- `i = j` のとき: `s_i * (1 - s_i)` — 自分自身への影響
- `i ≠ j` のとき: `-s_i * s_j` — 他の要素への影響（負の相関）

Softmax は「全体で1」という制約があるため、1つの要素を上げると他が下がる。この性質が微分にも反映されている。

## 効率的な backward の導出

ヤコビ行列を明示的に構築せず、ベクトル-ヤコビ積として直接計算する。

上流の勾配 `dy` に対して：

```
d_input[i] = Σ_j (dy[j] * ∂s_j/∂x_i)
           = Σ_j (dy[j] * s_j * (δ_ji - s_i))
           = s_i * dy[i] - s_i * Σ_j(dy[j] * s_j)
           = s_i * (dy[i] - dot(dy, s))
```

ここで `dot(dy, s) = Σ(dy[j] * s[j])` は1つのスカラー。

最終的に各要素の計算は：

```
d_input[i] = s[i] * (dy[i] - dot)
```

## 実装

```rust
pub fn softmax_backward(d_output: &Tensor, softmax_output: &Tensor) -> Tensor {
    let last_dim = *softmax_output.shape.last().expect("empty shape");
    let num_groups = softmax_output.data.len() / last_dim;
    let mut data = vec![0.0_f32; softmax_output.data.len()];

    for g in 0..num_groups {
        let start = g * last_dim;
        let s = &softmax_output.data[start..start + last_dim];
        let dy = &d_output.data[start..start + last_dim];

        let dot: f32 = s.iter().zip(dy.iter()).map(|(&si, &di)| si * di).sum();

        for i in 0..last_dim {
            data[start + i] = s[i] * (dy[i] - dot);
        }
    }

    Tensor::new(data, softmax_output.shape.clone())
}
```

forward と同様に「最後の軸に沿ったグループ」ごとに独立に処理する。

注意点として、この関数は forward の出力 `softmax_output`（確率）を引数に取る。入力 `x` は不要。backward に必要な情報は forward の出力だけで足りる。

## Cross Entropy Loss との関係

実はステップ16で実装した `cross_entropy_loss_backward` は、Softmax の backward と Cross Entropy の backward を合成した結果 `softmax - one_hot` を直接計算している。合成した方が数値的に安定で計算も効率的。

しかし Softmax 単体の backward もAttention の中で使われる（`softmax(QK^T/√d_k)` の勾配を逆伝播するため）ので、独立した実装が必要。

## テスト

```
running 8 tests
test softmax::tests::test_softmax_backward_2d_numerical ... ok
test softmax::tests::test_softmax_backward_numerical ... ok
test softmax::tests::test_softmax_backward_shape ... ok
(+ 5 forward tests)
```

1D と 2D の両方で数値微分と照合して検証している。

## 次回

Layer Normalization の backward を実装する。
