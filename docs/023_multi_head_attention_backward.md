# 023: Multi-Head Attention の backward

## 計算グラフ

Multi-Head Attention の forward：

```
1. Q_full = Wq(x), K_full = Wk(x), V_full = Wv(x)   [線形射影]
2. Q_h, K_h, V_h = split(Q_full, K_full, V_full)      [ヘッドに分割]
3. attn_h = attention(Q_h, K_h, V_h, mask)             [各ヘッドで attention]
4. concat = [attn_0 | attn_1 | ... | attn_H]           [結合]
5. output = Wo(concat)                                  [出力射影]
```

## Backward の流れ（逆順）

### Step 5: Wo の逆伝播

```
(d_concat, d_wo_weight, d_wo_bias) = Wo.backward(d_output, concat)
```

### Step 4-3: ヘッドごとの attention backward

`d_concat` をヘッドに分割し、各ヘッドで `attention_backward` を実行：

```
for each head h:
    d_out_h = d_concat[:, h*d_k:(h+1)*d_k]
    (d_q_h, d_k_h, d_v_h) = attention_backward(d_out_h, q_h, k_h, v_h, mask)
```

### Step 2: ヘッド勾配の結合

各ヘッドの勾配を結合して `d_q_full`, `d_k_full`, `d_v_full` を構築。

### Step 1: Wq, Wk, Wv の逆伝播

```
(d_x_from_q, d_wq_weight, d_wq_bias) = Wq.backward(d_q_full, x)
(d_x_from_k, d_wk_weight, d_wk_bias) = Wk.backward(d_k_full, x)
(d_x_from_v, d_wv_weight, d_wv_bias) = Wv.backward(d_v_full, x)
```

### d_x の合成

Self-attention では Q, K, V すべてが同じ入力 `x` から射影されるため、`d_x` は3つの寄与の合計：

```
d_x = d_x_from_q + d_x_from_k + d_x_from_v
```

## 勾配の構造体

返り値が9個（d_x + 4つの Linear × (weight, bias)）あるため、`MHAGradients` 構造体を定義：

```rust
pub struct MHAGradients {
    pub d_x: Tensor,
    pub d_wq_weight: Tensor, pub d_wq_bias: Tensor,
    pub d_wk_weight: Tensor, pub d_wk_bias: Tensor,
    pub d_wv_weight: Tensor, pub d_wv_bias: Tensor,
    pub d_wo_weight: Tensor, pub d_wo_bias: Tensor,
}
```

## テスト

```
running 9 tests
test multi_head_attention::tests::test_backward_numerical_d_x ... ok
test multi_head_attention::tests::test_backward_numerical_d_wo_weight ... ok
test multi_head_attention::tests::test_backward_shapes ... ok
(+ 6 forward tests)
```

`d_x` と `d_wo_weight` を数値微分で検証。causal mask あり・なしの両方でテスト。

## 次回

Feed-Forward Network の backward を実装する。
