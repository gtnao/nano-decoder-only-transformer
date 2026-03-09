# 027: パラメータ更新 — SGD と Adam

## SGD (Stochastic Gradient Descent)

最もシンプルな最適化アルゴリズム：

```
param = param - lr * grad
```

`lr`（学習率）が唯一のハイパーパラメータ。

### 問題点

- すべてのパラメータに同じ学習率が適用される
- 勾配のノイズに敏感
- 「谷」状の損失面で振動しやすい

## Adam (Adaptive Moment Estimation)

SGD の問題を解決する、現在最も広く使われている最適化アルゴリズム。

### 更新則

```
m = β1 * m + (1 - β1) * grad          // 1次モーメント（勾配の移動平均）
v = β2 * v + (1 - β2) * grad²         // 2次モーメント（勾配の二乗の移動平均）
m_hat = m / (1 - β1^t)                // バイアス補正
v_hat = v / (1 - β2^t)                // バイアス補正
param = param - lr * m_hat / (√v_hat + ε)
```

### ハイパーパラメータ

| パラメータ | デフォルト | 意味 |
|---|---|---|
| lr | - | 学習率 |
| β1 | 0.9 | 1次モーメントの減衰率 |
| β2 | 0.999 | 2次モーメントの減衰率 |
| ε | 1e-8 | ゼロ除算防止 |

### バイアス補正

`m` と `v` はゼロ初期化されるため、学習初期は真の値より小さく偏る。`(1 - β^t)` で除算することで、初期のバイアスを補正する。`t` が大きくなると補正は 1 に近づき、影響が消える。

### 適応的学習率

`√v_hat` で除算することで、勾配の分散が大きいパラメータは小さく更新し、小さいパラメータは大きく更新する。これにより、パラメータごとに適切な学習率が自動的に調整される。

## Transformer への統合

`param_sizes()` で全パラメータグループのサイズを返し、`apply_gradients()` で Adam の更新を全パラメータに適用する。

```rust
let mut adam = Adam::new(lr, &model.param_sizes());
// 学習ループ内:
let grads = model.backward(&d_logits, &token_ids);
model.apply_gradients(&grads, &mut adam);
```

1ブロックあたり17パラメータグループ（LN×2 + MHA×8 + FFN×4）に Embedding, Final LN, LM Head を加えた合計が Adam の状態として管理される。

## テスト

```
running 7 tests
test optimizer::tests::test_adam_basic ... ok
test optimizer::tests::test_adam_bias_correction ... ok
test optimizer::tests::test_adam_convergence ... ok
test optimizer::tests::test_adam_multiple_params ... ok
test optimizer::tests::test_sgd_basic ... ok
test optimizer::tests::test_sgd_convergence ... ok
test optimizer::tests::test_sgd_zero_gradient ... ok
```

SGD, Adam ともに簡単な2次関数の最小化で収束を確認。

## 次回

学習ループ（データ準備、ミニバッチ、エポック）を実装する。
