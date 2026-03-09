# 002: Linear層 — ニューラルネットワークの最も基本的な部品

## Linear層とは

Linear層（全結合層）は、入力ベクトルに重み行列を掛けてバイアスを足す演算である。

```
y = x @ W^T + b
```

- `x`: 入力 `[batch, in_features]`
- `W`: 重み `[out_features, in_features]`
- `b`: バイアス `[out_features]`
- `y`: 出力 `[batch, out_features]`

Transformer の中では至るところで使われる。Attention の Q/K/V の射影、FFN の2層、最終出力の射影など、ほぼ全ての変換が Linear 層で行われる。

## なぜ `W^T` なのか

重み `W` の shape は `[out_features, in_features]` で格納する。
これは PyTorch の `nn.Linear` と同じ慣習で、各行が1つの出力ニューロンの重みに対応する。

しかし行列積 `x @ W` を計算するには `x: [batch, in_features]` と `W: [in_features, ???]` のように内側の次元が一致する必要がある。
`W` は `[out_features, in_features]` なのでそのままでは掛けられない。転置して `W^T: [in_features, out_features]` にすることで：

```
x @ W^T = [batch, in_features] @ [in_features, out_features] = [batch, out_features]
```

と次元が合う。

## バイアスの加算

`x @ W^T` の結果は `[batch, out_features]` だが、`b` は `[out_features]` の1次元ベクトルである。
バッチ内の各サンプルに同じバイアスを足す必要があるので、`b` を `batch` 回繰り返して `[batch, out_features]` に展開してから加算する。

```rust
let bias_broadcast = Tensor::new(
    self.bias.data.repeat(batch),
    vec![batch, out_features],
);
out.add(&bias_broadcast)
```

本格的な実装ではブロードキャスト機構を Tensor に組み込むが、今回はシンプルに `repeat` で展開している。

## Xavier初期化

重みの初期値はモデルの学習に大きく影響する。値が大きすぎると勾配が爆発し、小さすぎると勾配が消失する。

Xavier初期化は、入力と出力の次元数に基づいて適切な範囲で一様乱数を生成する。

```rust
let limit = (6.0 / (in_features + out_features) as f64).sqrt() as f32;
// weight ~ Uniform(-limit, +limit)
```

これにより、層を通過しても値のスケールがおおむね保たれる。バイアスはゼロで初期化する。

## 実装

```rust
pub fn forward(&self, input: &Tensor) -> Tensor {
    let wt = self.weight.transpose();
    let out = input.matmul(&wt);
    let batch = out.shape[0];
    let out_features = out.shape[1];
    let bias_broadcast = Tensor::new(
        self.bias.data.repeat(batch),
        vec![batch, out_features],
    );
    out.add(&bias_broadcast)
}
```

Phase 1 で実装した `transpose`、`matmul`、`add` をそのまま組み合わせているだけである。
基盤をしっかり作っておくと、上位の部品はシンプルに書ける。

## テスト

```
running 7 tests
test linear::tests::test_forward ... ok
test linear::tests::test_forward_batch ... ok
test linear::tests::test_forward_shape_mismatch - should panic ... ok
test linear::tests::test_new ... ok
test linear::tests::test_new_bias_shape_mismatch - should panic ... ok
test linear::tests::test_new_weight_not_2d - should panic ... ok
test linear::tests::test_rand_shape ... ok
```

`new` で明示的に重みを渡せるようにすることで、`forward` の計算結果を手計算と照合できる。

## 次回

活性化関数（ReLU, GELU）を実装する。Linear 層だけでは入力の線形変換しかできないが、活性化関数を挟むことで非線形な表現が可能になる。
