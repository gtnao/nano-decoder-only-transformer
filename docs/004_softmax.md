# 004: Softmax — 任意の数値列を確率分布に変換する

## Softmax とは

Softmax は、任意の実数のベクトルを「合計が1になる正の値のベクトル」に変換する関数である。つまり確率分布を作る。

```
softmax(x_i) = exp(x_i) / Σ exp(x_j)
```

例：

```
入力:  [1.0, 2.0, 3.0]
exp:   [2.72, 7.39, 20.09]   (合計: 30.19)
出力:  [0.09, 0.24, 0.67]    (合計: 1.0)
```

大きい値ほど大きな確率が割り当てられ、差が指数関数的に増幅される。

## Transformer での役割

Attention 機構の中核で使われる。

```
Attention(Q, K, V) = softmax(QK^T / √d) V
```

`QK^T` で各トークン間の類似度（スコア）を計算した後、Softmax で確率分布に変換する。「どのトークンにどれだけ注目するか」を決める部分である。

## 数値安定性の問題

ナイーブに `exp(x)` を計算すると、`x` が大きいときオーバーフローする。

```
exp(1000.0) = Infinity
```

これを防ぐため、全要素から最大値を引いてから `exp` を計算する。

```
softmax(x_i) = exp(x_i - max(x)) / Σ exp(x_j - max(x))
```

数学的に結果は同一だが、`exp` の引数が常に0以下になるためオーバーフローが起きない。

### 同値性の証明

```
exp(x_i - c) / Σexp(x_j - c)
= exp(x_i) * exp(-c) / (Σexp(x_j) * exp(-c))
= exp(x_i) / Σexp(x_j)
```

`exp(-c)` が分子・分母で打ち消し合うため、任意の定数 `c` を引いても結果は変わらない。`c = max(x)` とするのが最も安全。

## 「最後の軸に沿って」適用する

2D テンソル `[batch, features]` の場合、各行が独立した確率分布になる。3D `[batch, seq, features]` でも同様に最後の軸（features）に沿って適用する。

実装では、最後の軸のサイズ `last_dim` でデータをグループ分けし、各グループに独立して Softmax を適用する。

```rust
let last_dim = *x.shape.last().expect("empty shape");
let num_groups = x.data.len() / last_dim;

for g in 0..num_groups {
    let start = g * last_dim;
    let group = &x.data[start..start + last_dim];

    let max_val = group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = group.iter().map(|&v| (v - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();

    for (i, &e) in exps.iter().enumerate() {
        data[start + i] = e / sum;
    }
}
```

row-major order で格納されているため、最後の軸の要素はメモリ上で連続している。`last_dim` 個ずつ区切ればちょうど各グループに対応する。

## テスト

```
running 5 tests
test softmax::tests::test_softmax_1d ... ok
test softmax::tests::test_softmax_2d_row_wise ... ok
test softmax::tests::test_softmax_3d ... ok
test softmax::tests::test_softmax_numerical_stability ... ok
test softmax::tests::test_softmax_uniform ... ok
```

- **uniform**: 同じ値を入れたら均等な確率 (`1/n`) になる
- **row_wise**: 2D で各行が独立に合計1になる
- **numerical_stability**: `[1000, 1001, 1002]` でもオーバーフローせず正常動作
- **3d**: 3階テンソルでも最後の軸に沿って正しく適用される

## 次回

Layer Normalization を実装する。Transformer の各サブレイヤーの前に適用され、学習を安定させる正規化手法である。
