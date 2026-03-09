# 037: バッチ処理

## 背景

これまでは1ステップで1シーケンスのみ処理していた。バッチ処理により：

- **勾配の安定化**: 複数サンプルの勾配を平均することで、ノイズが減少
- **ステップ数の削減**: batch_size=8 なら更新回数が 1/8 に
- **学習の効率化**: より安定した勾配で、学習率を大きくできる可能性がある

## アプローチ: 勾配累積

3Dテンソルへの拡張（全レイヤーの改修が必要）ではなく、**バッチ内の各サンプルを順次forward/backwardし、勾配を累積してからパラメータ更新**する方式を採用。

```
accumulated_grads = zeros
for each sample in batch:
    logits = forward(sample)
    loss = cross_entropy(logits, target)
    grads = backward(loss)
    accumulated_grads += grads

accumulated_grads /= batch_size
clip_norm(accumulated_grads)
apply_gradients(accumulated_grads)
```

### なぜこの方式か

- forward/backward のインターフェースを変更不要
- 既存の全レイヤー（Attention, FFN, LayerNorm 等）は一切変更不要
- 数学的に正しい（勾配は線形なので、加算→平均 = バッチ全体の平均勾配）
- 既存テストが全て通り続ける

## 実装

### TransformerGradients の拡張 (transformer.rs)

```rust
impl TransformerGradients {
    // Zero-initialized gradients matching model structure
    pub fn zeros_like(model: &Transformer) -> Self { ... }

    // In-place gradient accumulation
    pub fn accumulate(&mut self, other: &TransformerGradients) {
        self.d_embedding_weight.add_inplace(&other.d_embedding_weight);
        // ... all parameter gradients ...
    }

    // Scale for batch averaging
    pub fn scale(&mut self, s: f32) {
        self.d_embedding_weight.scale_inplace(s);
        // ... all parameter gradients ...
    }
}
```

### train_step_batch (train.rs)

```rust
pub fn train_step_batch(
    model: &mut Transformer,
    adam: &mut Adam,
    batch: &[(Vec<usize>, Vec<usize>)],
    max_grad_norm: f32,
) -> f32 {
    let mut accumulated = TransformerGradients::zeros_like(model);
    let mut total_loss = 0.0;

    for (input, target) in batch {
        let logits = model.forward_with_training(input, true);
        let loss = cross_entropy_loss(&logits, target);
        total_loss += loss;
        let d_logits = cross_entropy_loss_backward(&logits, target);
        let grads = model.backward(&d_logits, input);
        accumulated.accumulate(&grads);
    }

    accumulated.scale(1.0 / batch.len() as f32);
    accumulated.clip_norm(max_grad_norm);
    model.apply_gradients(&accumulated, adam);

    total_loss / batch.len() as f32
}
```

### Dropout との整合性

バッチ内の各サンプルで forward → backward を完結させるため、dropout マスクは各サンプルごとに正しくキャッシュ・参照される。

### 後方互換

既存の `train_step` は `train_step_batch` を batch_size=1 で呼ぶラッパーとして維持。既存の `train` 関数も同様に `train_with_batch` を batch_size=1 で呼ぶ。

## 効果

| | batch_size=1 | batch_size=8 |
|---|---|---|
| ステップ数/epoch | 247 | 31 |
| 総ステップ数 (30 epochs) | 7410 | 930 |
| パラメータ更新回数 | 7410 | 930 |

計算量自体は同じ（各サンプルのforward/backwardは同じ）だが、パラメータ更新のオーバーヘッドが減少し、勾配が安定化する。

## テスト

```
172 tests passed (+3 new)
- test_zeros_like_shapes
- test_train_step_batch_returns_finite_loss
- test_train_with_batch_loss_decreases
```
