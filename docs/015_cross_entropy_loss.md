# 015: Cross Entropy Loss — モデルの予測がどれだけ正しいかを測る

## Loss（損失関数）とは

モデルの出力と正解との「ずれ」を数値化する関数。この値が小さいほど良い予測をしている。学習はこの Loss を最小化する最適化問題として定式化される。

## Cross Entropy Loss の直感

言語モデルは各位置で「次のトークンの確率分布」を出力する。Cross Entropy Loss は「正解トークンに割り当てた確率が高いほどロスが小さい」という性質を持つ。

```
予測確率: [0.7, 0.2, 0.1]  (token 0, 1, 2)
正解:     token 0
loss = -ln(0.7) = 0.357   ← 低い（良い予測）

予測確率: [0.1, 0.2, 0.7]
正解:     token 0
loss = -ln(0.1) = 2.303   ← 高い（悪い予測）
```

## 数式

```
Loss = -1/N × Σ log_softmax(logits[pos])[target[pos]]
```

展開すると：

```
log_softmax(x_i) = x_i - log(Σ exp(x_j))
```

`-log(確率)` は情報理論における「驚き（surprise）」に対応する。確率が低いほど驚きが大きく、Loss も大きくなる。

## 数値安定性

ナイーブに `exp(x)` を計算するとオーバーフローする。Softmax と同じく、最大値を引く技法を使う。

```
log_softmax(x_i) = (x_i - max) - log(Σ exp(x_j - max))
```

数学的に同値だが、`exp` の引数が常に 0 以下になるためオーバーフローしない。

## 均一予測のときの Loss

全 logits が同じ値のとき、確率は `1/vocab_size` に均等分布する。

```
Loss = -ln(1/vocab_size) = ln(vocab_size)
```

vocab_size=17（現在のコーパス）なら `ln(17) ≈ 2.83`。未学習モデルの Loss はこの付近になるはず。これがベースラインであり、学習が進めばここから下がっていく。

## 実装

```rust
pub fn cross_entropy_loss(logits: &Tensor, targets: &[usize]) -> f32 {
    let seq_len = logits.shape[0];
    let vocab_size = logits.shape[1];
    let mut total_loss = 0.0_f32;

    for pos in 0..seq_len {
        let row = &logits.data[pos * vocab_size..(pos + 1) * vocab_size];

        // log-softmax with numerical stability
        let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let log_sum_exp = row.iter().map(|&v| (v - max_val).exp()).sum::<f32>().ln();
        let log_prob = row[targets[pos]] - max_val - log_sum_exp;

        total_loss += -log_prob;
    }

    total_loss / seq_len as f32  // average over positions
}
```

Softmax の計算と `ln` をまとめて行う（log-softmax）ことで、中間で確率値を経由せず直接対数確率を得る。これにより `ln(0)` のような数値的な問題も回避できる。

## テスト

```
running 6 tests
test loss::tests::test_averaged_over_positions ... ok
test loss::tests::test_numerical_stability ... ok
test loss::tests::test_perfect_prediction ... ok
test loss::tests::test_single_position ... ok
test loss::tests::test_uniform_prediction ... ok
test loss::tests::test_worst_prediction ... ok
```

- **perfect_prediction**: 正解に高い logits → Loss ≈ 0
- **worst_prediction**: 正解に低い logits → Loss が大きい
- **uniform_prediction**: 均一 logits → Loss = ln(vocab_size)
- **single_position**: 手計算 (`-log_softmax([1,2,3])[0] ≈ 2.408`) と一致
- **numerical_stability**: logits が 1000 台でもオーバーフローしない
- **averaged_over_positions**: 2位置の平均 = 1位置の値（均一なので同値）

## 次回

逆伝播の基盤を実装する。Loss から各パラメータへの勾配を計算するための仕組み（勾配テンソルと連鎖律）を整備する。
