# 011: Transformer Block — 全部品を1層にまとめる

## Transformer Block の構造

Transformer Block は、これまで実装した部品を組み合わせた1層分の処理単位である。実際の Transformer はこの Block を N 層積み重ねる。

```
入力 x
  │
  ├──────────────────┐
  │                  │ (残差接続)
  ▼                  │
LayerNorm1           │
  │                  │
  ▼                  │
Multi-Head Attention │
  │                  │
  ▼                  │
  + ◄────────────────┘
  │
  ├──────────────────┐
  │                  │ (残差接続)
  ▼                  │
LayerNorm2           │
  │                  │
  ▼                  │
Feed-Forward Network │
  │                  │
  ▼                  │
  + ◄────────────────┘
  │
  ▼
出力
```

コードで書くと：

```
x = x + MHA(LayerNorm1(x))
x = x + FFN(LayerNorm2(x))
```

## 残差接続 (Residual Connection)

各サブレイヤー（MHA, FFN）の出力に、入力をそのまま足し合わせる。

```
output = x + SubLayer(x)
```

### なぜ必要か

層が深くなると勾配消失が起きやすい。残差接続があれば、最悪の場合でも `SubLayer(x) ≈ 0` を学習すれば入力がそのまま通過する。つまり「何もしない」という選択肢を常に持てる。

逆伝播の観点では、勾配が残差のショートカットを通って直接浅い層に到達できるため、深いネットワークでも学習が安定する。

## Pre-LayerNorm vs Post-LayerNorm

LayerNorm の位置には2つのバリエーションがある。

**Post-LN**（元の論文）:
```
x = LayerNorm(x + MHA(x))
```

**Pre-LN**（GPT-2以降の主流）:
```
x = x + MHA(LayerNorm(x))
```

今回は Pre-LN を採用した。Pre-LN は学習が安定しやすく、learning rate warmup なしでも発散しにくい。サブレイヤーへの入力が正規化されているため、勾配のスケールが安定するため。

## 実装

```rust
pub fn forward(&self, x: &Tensor) -> Tensor {
    // x = x + MHA(LayerNorm1(x))
    let normed1 = self.ln1.forward(x);
    let attn_out = self.mha.forward(&normed1, true);
    let x = x.add(&attn_out);

    // x = x + FFN(LayerNorm2(x))
    let normed2 = self.ln2.forward(&x);
    let ffn_out = self.ffn.forward(&normed2);
    x.add(&ffn_out)
}
```

MHA には `use_causal_mask: true` を渡している。Decoder-Only モデルでは、各トークンは自分より前のトークンにしか注目できないため、常に Causal Mask が必要。

入出力の shape は `[seq_len, d_model]` で変わらない。このおかげで同じ Block を何層でも積み重ねられる。

## テスト

```
running 5 tests
test transformer_block::tests::test_forward_deterministic ... ok
test transformer_block::tests::test_forward_different_inputs_differ ... ok
test transformer_block::tests::test_forward_shape ... ok
test transformer_block::tests::test_rand ... ok
test transformer_block::tests::test_residual_connection ... ok
```

- **shape**: 入出力が `[seq_len, d_model]` で一致
- **residual_connection**: 出力がゼロにならない（残差が入力を保持している）
- **different_inputs_differ**: 異なる入力は異なる出力を生む
- **deterministic**: 同じ入力は同じ出力

## 次回

Transformer 全体を組み立てる。Embedding → Positional Encoding → N × TransformerBlock → LayerNorm → 出力 Linear で、入力トークンIDから次トークンの確率分布を出力する完全なモデルになる。
