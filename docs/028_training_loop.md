# 028: 学習ループ

## 全体の流れ

```
1. テキスト → トークン化 → (input, target) ペア
2. Forward: input → logits
3. Loss: cross_entropy(logits, target)
4. Backward: d_logits → 全パラメータの勾配
5. Update: Adam で全パラメータを更新
6. 繰り返し
```

## データ準備

言語モデリングでは、入力系列の「次のトークン」を予測する：

```
テキスト: "hello world"
tokens:   [h, e, l, l, o, _, w, o, r, l, d]

Window 1:
  input:  [h, e, l, l, o, _, w, o]
  target: [e, l, l, o, _, w, o, r]

Window 2:
  input:  [r, l, d, ...]
  ...
```

`prepare_data` は固定長のスライディングウィンドウでデータを分割する。

```rust
pub fn prepare_data(token_ids: &[usize], seq_len: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
    // Non-overlapping windows: stride = seq_len
    // input = tokens[start..start+seq_len]
    // target = tokens[start+1..start+seq_len+1]
}
```

## 学習ステップ

```rust
pub fn train_step(model, adam, input, target) -> f32 {
    let logits = model.forward(input);
    let loss = cross_entropy_loss(&logits, target);
    let d_logits = cross_entropy_loss_backward(&logits, target);
    let grads = model.backward(&d_logits, input);
    model.apply_gradients(&grads, adam);
    loss
}
```

1ステップが forward → loss → backward → update の完全なサイクル。

## 学習結果

小さなコーパス "hello world hello world hello world" での結果：

```
Steps: 400, Loss: 2.3051 -> 0.0052
```

- 初期 loss ≈ 2.3 は `ln(vocab_size=10) ≈ 2.3` に近い（ランダム予測相当）
- 最終 loss ≈ 0.005 はほぼ完全な記憶

学習前後のテキスト生成：

```
Before: "hello" -> "hellolllllllwllllllllllll"  (でたらめ)
After:  "hello" -> "hello worrreeooooooodoeoe"  (パターンを学習)
```

学習後は "hello wor..." のパターンが出現。小さなモデルと短い学習なので完璧ではないが、明らかにコーパスのパターンを学習している。

## テスト

```
running 6 tests
test train::tests::test_prepare_data_basic ... ok
test train::tests::test_prepare_data_exact ... ok
test train::tests::test_prepare_data_too_short ... ok
test train::tests::test_train_step_returns_finite_loss ... ok
test train::tests::test_train_loss_decreases ... ok
test train::tests::test_overfit_tiny_sequence ... ok
```

- `test_train_loss_decreases`: 学習を通じて loss が減少することを検証
- `test_overfit_tiny_sequence`: 小さな系列を暗記できることを検証

## まとめ

これで Decoder-Only Transformer の全実装が完了：

1. **データ構造**: Tensor（行列演算）
2. **モデル構成要素**: Linear, Activation, Softmax, LayerNorm, Embedding, Positional Encoding
3. **アーキテクチャ**: Attention → Multi-Head Attention → FFN → Transformer Block → Transformer
4. **推論**: Tokenizer + Generate（温度サンプリング）
5. **学習**: Loss → Backward（全レイヤー） → Adam → Training Loop

外部依存は `rand` のみ。行列演算含め全て自前実装。155 テストで検証済み。
