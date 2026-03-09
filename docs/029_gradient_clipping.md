# 029: 勾配クリッピング

## なぜ必要か

Transformer は深いネットワーク＋残差接続＋Attention の組み合わせにより、勾配が突然非常に大きくなることがある（勾配爆発）。大きな勾配でパラメータが一気に動くと、以降の学習が不安定になり、loss が NaN に発散することもある。

勾配クリッピングは「勾配が大きすぎたらスケールダウンする」安全装置。

## アルゴリズム

**Global Norm Clipping**（PyTorch の `torch.nn.utils.clip_grad_norm_` と同じ方式）：

1. 全パラメータの勾配をまとめた L2 ノルムを計算：

```
norm = sqrt(Σ g_i²)   (全パラメータ、全要素について)
```

2. ノルムが閾値（`max_norm`）を超えたら、全勾配をスケーリング：

```
if norm > max_norm:
    g_i = g_i * (max_norm / norm)
```

### 重要な性質

- **方向を保つ**: 全勾配を同じ比率でスケーリングするので、勾配の方向は変わらない。大きさだけ制限する。
- **グローバルノルム**: 各パラメータ個別ではなく、全パラメータをまとめた1つのノルムを使う。これにより、パラメータ間の相対的な勾配の大きさが保たれる。
- **閾値以下なら何もしない**: 勾配が小さいときは一切変更しない。

## 実装

```rust
pub fn clip_grad_norm(grads: &mut [&mut [f32]], max_norm: f32) -> f32 {
    let mut total_sq = 0.0_f32;
    for g in grads.iter() {
        for &v in g.iter() {
            total_sq += v * v;
        }
    }
    let norm = total_sq.sqrt();

    if norm > max_norm {
        let scale = max_norm / norm;
        for g in grads.iter_mut() {
            for v in g.iter_mut() {
                *v *= scale;
            }
        }
    }
    norm
}
```

`TransformerGradients` にも `clip_norm` メソッドを追加し、全パラメータ勾配を一括でクリッピングできるようにした。

## 学習ループへの統合

```rust
let mut grads = model.backward(&d_logits, input);
grads.clip_norm(1.0);  // max_norm=1.0
model.apply_gradients(&grads, adam);
```

backward と update の間に1行挿入するだけ。`max_norm=1.0` が一般的なデフォルト値。

## テスト

```
running 11 tests
test optimizer::tests::test_clip_grad_norm_clips ... ok
test optimizer::tests::test_clip_grad_norm_multiple_slices ... ok
test optimizer::tests::test_clip_grad_norm_no_clip ... ok
test optimizer::tests::test_clip_grad_norm_preserves_direction ... ok
(+ 7 SGD/Adam tests)
```

ノルム超過時のスケーリング、未超過時の不変性、方向保存、複数スライス対応を検証。

## 次回

学習率スケジューリング（linear warmup + cosine decay）を実装する。
