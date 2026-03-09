# 005: Layer Normalization — 学習を安定させる正規化

## なぜ正規化が必要か

ニューラルネットワークの学習では、各層の出力の分布が学習中に変動する（内部共変量シフト）。値のスケールが層を通過するたびに大きくなったり小さくなったりすると、勾配が爆発・消失し学習が不安定になる。

正規化はこの問題を緩和する。各層の出力を「平均0・分散1」に揃えてから次の層に渡す。

## Layer Norm の計算

最後の軸に沿って、平均と分散を求めて正規化する。

```
output = γ * (x - μ) / √(σ² + ε) + β
```

- `μ`: 最後の軸方向の平均
- `σ²`: 最後の軸方向の分散
- `γ` (gamma): 学習可能なスケールパラメータ（初期値 1）
- `β` (beta): 学習可能なシフトパラメータ（初期値 0）
- `ε` (eps): ゼロ除算防止の微小値（`1e-5`）

### 具体例

```
x = [1.0, 2.0, 3.0, 4.0]
μ = 2.5
σ² = 1.25
√(σ² + ε) ≈ 1.11803

normalized = [-1.5, -0.5, 0.5, 1.5] / 1.11803
           ≈ [-1.3416, -0.4472, 0.4472, 1.3416]
```

γ=1, β=0 のとき、出力は平均≈0、分散≈1 になる。

## γ と β の役割

正規化で平均0・分散1に固定してしまうと、モデルの表現力が制限される。γ と β を学習可能にすることで、モデルが「どの程度正規化を適用するか」を自ら決められる。

極端な場合、γ=σ, β=μ と学習すれば正規化を無効化できる。つまり正規化が不要な場面ではモデルが自動的にそれを学ぶ。

## Batch Norm との違い

| | Batch Norm | Layer Norm |
|---|---|---|
| 正規化の軸 | バッチ方向（同じニューロンを全サンプルで） | 特徴量方向（1サンプル内の全ニューロンで） |
| バッチサイズ依存 | あり（小バッチで不安定） | なし |
| 推論時の扱い | 移動平均が必要 | 学習時と同じ |

Transformer では Layer Norm が標準。系列長が可変で、バッチ方向の統計量に依存しないことが望ましいため。

## 実装

Softmax と同じく「最後の軸に沿った操作」パターン。row-major order のおかげで、最後の軸の要素がメモリ上で連続している。

```rust
pub fn forward(&self, x: &Tensor) -> Tensor {
    let last_dim = *x.shape.last().expect("empty shape");
    let num_groups = x.data.len() / last_dim;
    let mut data = vec![0.0_f32; x.data.len()];

    for g in 0..num_groups {
        let start = g * last_dim;
        let group = &x.data[start..start + last_dim];

        let mean = group.iter().sum::<f32>() / last_dim as f32;
        let var = group.iter()
            .map(|&v| (v - mean) * (v - mean))
            .sum::<f32>() / last_dim as f32;
        let std_inv = 1.0 / (var + self.eps).sqrt();

        for i in 0..last_dim {
            data[start + i] = self.gamma.data[i] * (group[i] - mean) * std_inv
                            + self.beta.data[i];
        }
    }

    Tensor::new(data, x.shape.clone())
}
```

ポイント：
- `std_inv`（標準偏差の逆数）を先に計算し、ループ内では乗算だけで済ませる
- `eps` を分散に足してから平方根を取ることで、分散がゼロのときのゼロ除算を防ぐ

## テスト

```
running 6 tests
test layer_norm::tests::test_forward_3d ... ok
test layer_norm::tests::test_forward_batch ... ok
test layer_norm::tests::test_forward_known_values ... ok
test layer_norm::tests::test_forward_with_gamma_beta ... ok
test layer_norm::tests::test_forward_zero_mean_unit_var ... ok
test layer_norm::tests::test_new ... ok
```

- **zero_mean_unit_var**: デフォルト(γ=1, β=0)で出力の平均≈0、分散≈1 を確認
- **known_values**: 手計算した値と一致するか検証
- **batch**: 各行が独立に正規化されること
- **with_gamma_beta**: γ, β を変更した場合の動作
- **3d**: 3階テンソルでも最後の軸に沿って正しく適用

## 次回

Embedding を実装する。トークンID（整数）を固定長のベクトルに変換する、Transformer への入力の入口となる層である。
