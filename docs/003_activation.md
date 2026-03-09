# 003: 活性化関数 — ニューラルネットに非線形性を与える

## なぜ活性化関数が必要か

Linear 層は `y = xW + b` という線形変換である。線形変換を何層重ねても結果は線形のままであり、表現力が増えない。

```
y = x @ W1 @ W2 = x @ (W1 @ W2) = x @ W'
```

2層の Linear を重ねても、1つの行列 `W'` で置き換えられてしまう。
活性化関数を間に挟むことでこの性質を壊し、複雑なパターンを学習できるようにする。

## ReLU (Rectified Linear Unit)

```
ReLU(x) = max(0, x)
```

負の値を 0 にし、正の値はそのまま通す。最もシンプルな活性化関数。

```
入力:  [-2, -1, 0, 1, 2]
出力:  [ 0,  0, 0, 1, 2]
```

実装は1行で済む：

```rust
let data = x.data.iter().map(|&v| v.max(0.0)).collect();
```

## GELU (Gaussian Error Linear Unit)

GPT や BERT など、現代の Transformer で標準的に使われる活性化関数。

正確な定義は：

```
GELU(x) = x * Φ(x)
```

`Φ(x)` は標準正規分布の累積分布関数。直感的には「入力が大きいほどそのまま通し、小さいほど抑制する」という確率的なゲートである。

計算効率のため、tanh による近似式が広く使われる：

```
GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
```

### ReLU との違い

| 特性 | ReLU | GELU |
| --- | --- | --- |
| x < 0 の出力 | 常に 0（完全遮断） | 小さな負の値を許容 |
| x = 0 での滑らかさ | 不連続な折れ線 | 滑らかな曲線 |
| 性能 | 計算が速い | やや遅いが精度が良い傾向 |

ReLU は負の入力を完全に0にしてしまうが、GELU は滑らかに減衰させる。
例えば `GELU(-1.0) ≈ -0.159` であり、完全には潰れない。

## 実装

```rust
pub fn gelu(x: &Tensor) -> Tensor {
    let coeff = (2.0_f32 / std::f32::consts::PI).sqrt();
    let data = x
        .data
        .iter()
        .map(|&v| {
            let inner = coeff * (v + 0.044715 * v * v * v);
            0.5 * v * (1.0 + inner.tanh())
        })
        .collect();
    Tensor::new(data, x.shape.clone())
}
```

各要素に独立して適用する要素ごとの演算なので、`iter().map()` でシンプルに書ける。
shape は入力と同じまま変わらない。

## テスト

```
running 9 tests
test activation::tests::test_gelu_known_values ... ok
test activation::tests::test_gelu_negative_large ... ok
test activation::tests::test_gelu_preserves_shape ... ok
test activation::tests::test_gelu_positive_large ... ok
test activation::tests::test_gelu_zero ... ok
test activation::tests::test_relu_mixed ... ok
test activation::tests::test_relu_negative ... ok
test activation::tests::test_relu_positive ... ok
test activation::tests::test_relu_preserves_shape ... ok
```

GELU のテストでは以下の性質を検証している：

- `GELU(0) = 0`
- 大きな正の値ではほぼ恒等関数（`GELU(3.0) ≈ 3.0`）
- 大きな負の値ではほぼ 0（`GELU(-3.0) ≈ 0`）
- 既知の値との照合（`GELU(1.0) ≈ 0.8412`, `GELU(-1.0) ≈ -0.1588`）

## 次回

Softmax を実装する。Attention 機構で「どのトークンにどれだけ注目するか」の確率分布を作るために使われる。
