# 016: 逆伝播の基盤 — 連鎖律と勾配の流れ

## 逆伝播（Backpropagation）とは

ニューラルネットワークの学習は「Loss を最小化するようにパラメータを調整する」こと。そのためには「各パラメータを少し変えたら Loss がどう変わるか」= 勾配（gradient）を知る必要がある。

逆伝播は、Loss から逆順に各層を辿りながら、連鎖律（chain rule）を使って全パラメータの勾配を効率的に計算するアルゴリズム。

## 連鎖律

合成関数の微分法則：

```
y = f(g(x))
dy/dx = dy/dg × dg/dx
```

ニューラルネットワークは関数の合成:

```
Loss = L(FFN(Attention(LayerNorm(Embedding(x)))))
```

逆伝播は出力側（Loss）から入力側に向かって、各層の局所的な微分を掛け合わせていく。

## backward 関数のパターン

各層の backward は以下の統一的なパターンに従う：

```
入力:  d_output  (上流から来た勾配 = ∂Loss/∂output)
出力:  d_input   (下流へ渡す勾配 = ∂Loss/∂input)
副作用: d_weight, d_bias (パラメータの勾配を蓄積)
```

例えば3層のネットワーク `y = C(B(A(x)))` の逆伝播：

```
Forward:  x → [A] → a → [B] → b → [C] → y → Loss
Backward: Loss → d_y → [C.backward] → d_b → [B.backward] → d_a → [A.backward] → d_x
```

各 backward は `d_output` を受け取り、`d_input` を返す。パラメータを持つ層は同時に `d_weight`, `d_bias` も計算する。

## Cross Entropy Loss の backward

逆伝播の出発点。Loss → logits への勾配を計算する。

Cross Entropy Loss + Softmax の勾配は非常にシンプルな形になる：

```
d_logits = (softmax(logits) - one_hot(targets)) / seq_len
```

### なぜこうなるか

Cross Entropy Loss: `L = -Σ t_i × log(s_i)` （`t`: one-hot, `s`: softmax 出力）

Softmax + Cross Entropy を合わせて微分すると：
- 正解クラス `i = target`: `∂L/∂logit_i = s_i - 1`（確率 - 1）
- それ以外: `∂L/∂logit_i = s_i`（確率そのもの）

まとめると `d_logits = softmax - one_hot`。正解に近い予測（`s_i ≈ 1`）なら勾配が小さく、遠い予測なら大きい。

### 勾配の性質

- **各行の合計は 0**: `Σ(softmax) - Σ(one_hot) = 1 - 1 = 0`
- **正解位置は負**: `s_i - 1 < 0`（logits を増やす方向に更新）
- **不正解位置は正**: `s_i > 0`（logits を減らす方向に更新）

## 実装

```rust
pub fn cross_entropy_loss_backward(logits: &Tensor, targets: &[usize]) -> Tensor {
    let seq_len = logits.shape[0];
    let probs = softmax(logits);
    let mut d_logits = probs.data.clone();
    let vocab_size = logits.shape[1];

    for pos in 0..seq_len {
        d_logits[pos * vocab_size + targets[pos]] -= 1.0;
    }

    // Average over positions
    let scale = 1.0 / seq_len as f32;
    for v in d_logits.iter_mut() {
        *v *= scale;
    }

    Tensor::new(d_logits, logits.shape.clone())
}
```

## 数値微分による検証

解析的に求めた勾配が正しいか、有限差分法（numerical gradient）で検証する：

```
numerical_grad[i] = (Loss(logit_i + ε) - Loss(logit_i - ε)) / (2ε)
```

解析的勾配と数値微分が一致すれば、backward の実装が正しいことが保証される。以降の各層でもこのパターンで検証していく。

## テスト

```
running 10 tests
test loss::tests::test_backward_numerical_gradient ... ok
test loss::tests::test_backward_shape ... ok
test loss::tests::test_backward_sums_to_zero_per_row ... ok
test loss::tests::test_backward_target_position_negative ... ok
(+ 6 forward tests)
```

## 次回

Linear 層の backward を実装する。`y = x @ W^T + b` の逆伝播で `d_x`, `d_W`, `d_b` を求める。
