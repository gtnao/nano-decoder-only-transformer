# 031: Dropout

## Dropout とは

学習時にランダムにニューロンの出力を 0 にする正則化手法。

```
forward (training):
  mask[i] = random() >= rate ? true : false
  output[i] = mask[i] ? input[i] / (1 - rate) : 0

forward (inference):
  output = input  (何もしない)
```

### スケーリング

kept 要素を `1/(1-rate)` 倍することで、学習時と推論時の出力の期待値を一致させる（inverted dropout）。推論時にスケーリングが不要になるため、実用的。

### なぜ効くか

- 各学習ステップで異なるサブネットワークが訓練される
- 特定のニューロンへの過度な依存（co-adaptation）を防ぐ
- アンサンブル学習的な効果

## Transformer での適用箇所

```
x1 = x + dropout(MHA(LayerNorm1(x)))     ← dropout 1
x2 = x1 + dropout(FFN(LayerNorm2(x1)))   ← dropout 2
```

サブレイヤーの出力に dropout を適用してから残差接続に加算する。

## 実装上の課題: マスクのキャッシュ

Backward では forward と同じマスクを使う必要がある。本プロジェクトでは backward 内で forward を再計算しているため、マスクの扱いが問題になる。

解決策: `RefCell` で forward 時のマスクをキャッシュし、backward で参照する。

```rust
pub struct TransformerBlock {
    // ...
    dropout_masks: RefCell<Option<(Vec<bool>, Vec<bool>)>>,
}
```

### Forward

```rust
let (attn_out, mask1) = if training && dropout_rate > 0.0 {
    dropout(&attn_out, dropout_rate)
} else {
    (attn_out, vec![true; ...])
};
// Cache masks for backward
*self.dropout_masks.borrow_mut() = Some((mask1, mask2));
```

### Backward

```rust
let masks = self.dropout_masks.borrow().clone();
// Apply cached mask to gradient
if let Some((ref mask1, _)) = masks {
    d_attn_out = dropout_backward(&d_attn_out, mask1, dropout_rate);
}
```

## training フラグ

`forward(x, training)` パラメータで学習/推論を切り替える：
- `training=true`: dropout 適用、マスクキャッシュ
- `training=false`: dropout スキップ（推論時）

`Transformer::forward()` は推論用（`training=false`）、`Transformer::forward_with_training()` は学習用。

## テスト

```
running 169 tests
test dropout::tests::test_dropout_backward_matches_mask ... ok
test dropout::tests::test_dropout_preserves_expected_value ... ok
test dropout::tests::test_dropout_zeros_some_elements ... ok
test dropout::tests::test_dropout_zero_rate ... ok
test dropout::tests::test_dropout_shape_preserved ... ok
test transformer_block::tests::test_forward_with_dropout ... ok
test transformer_block::tests::test_forward_no_dropout_in_eval ... ok
(+ existing tests)
```

## 次回

日本語コーパス（蜘蛛の糸）で改善された学習デモを実行する。
