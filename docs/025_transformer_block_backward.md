# 025: Transformer Block の backward

## Pre-LN Transformer Block の復習

```
normed1  = LayerNorm1(x)
attn_out = MHA(normed1)
x1       = x + attn_out         ← 残差接続1

normed2  = LayerNorm2(x1)
ffn_out  = FFN(normed2)
output   = x1 + ffn_out         ← 残差接続2
```

## 残差接続の勾配

残差接続 `y = x + f(x)` の勾配は：

```
d_x = d_y + d_f(x)
```

勾配がそのまま通過する（`d_y`）ことが、深いネットワークでも勾配が消失しにくい理由。

## Backward の流れ

### 第2残差ブランチ（output = x1 + ffn_out）

```
ffn_grads = ffn.backward(d_output, normed2)
(d_x1_from_ln2, d_ln2_gamma, d_ln2_beta) = ln2.backward(ffn_grads.d_x, x1)
d_x1 = d_output + d_x1_from_ln2    ← 残差 + LN2経路
```

### 第1残差ブランチ（x1 = x + attn_out）

```
mha_grads = mha.backward(d_x1, normed1)
(d_x_from_ln1, d_ln1_gamma, d_ln1_beta) = ln1.backward(mha_grads.d_x, x)
d_x = d_x1 + d_x_from_ln1          ← 残差 + LN1経路
```

## 勾配の流れ図

```
d_output ──┬──────────────── d_output (residual)
           │                     │
           ▼                     │
     ffn.backward                │
           │                     │
           ▼                     │
     ln2.backward                │
           │                     │
           ▼                     ▼
        d_x1 = d_output + d_x1_from_ln2
           │
      ┌────┴──────────────── d_x1 (residual)
      │                          │
      ▼                          │
 mha.backward                   │
      │                          │
      ▼                          │
 ln1.backward                   │
      │                          │
      ▼                          ▼
   d_x = d_x1 + d_x_from_ln1
```

残差接続のおかげで、`d_output` が2回の加算を経て `d_x` にほぼそのまま届く。

## テスト

```
running 7 tests
test transformer_block::tests::test_backward_numerical_d_x ... ok
test transformer_block::tests::test_backward_shapes ... ok
(+ 5 forward tests)
```

`d_x` を数値微分で検証。

## 次回

Transformer 全体の backward を実装する。
