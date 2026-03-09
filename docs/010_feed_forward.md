# 010: Feed-Forward Network — 各トークンを独立に変換する

## FFN の役割

Transformer Block の中で、Multi-Head Attention と FFN は異なる役割を担う。

- **Attention**: トークン間の関係を捉える（「どこに注目するか」）
- **FFN**: 各トークンの表現を個別に変換する（「注目した結果をどう解釈するか」）

FFN は各位置（トークン）に対して同じパラメータで独立に適用される。これを position-wise と呼ぶ。トークン A の FFN 計算にトークン B の情報は一切入らない。

## 構造

```
FFN(x) = Linear2(GELU(Linear1(x)))
```

- `Linear1`: `[d_model, d_ff]` — 次元を拡大
- `GELU`: 活性化関数
- `Linear2`: `[d_ff, d_model]` — 元の次元に戻す

典型的には `d_ff = 4 * d_model`。例えば `d_model=512` なら `d_ff=2048`。

### なぜ次元を拡大してから戻すのか

一度高次元に射影し、非線形変換（GELU）を施してから元の次元に戻す。これにより低次元空間では表現できない複雑なパターンを捉えられる。

イメージとしては、2次元では分離できないデータを一旦3次元に持ち上げて分離し、2次元に戻すようなもの。

## 実装

```rust
pub fn forward(&self, x: &Tensor) -> Tensor {
    let hidden = self.linear1.forward(x);   // [seq_len, d_ff]
    let activated = gelu(&hidden);           // [seq_len, d_ff]
    self.linear2.forward(&activated)         // [seq_len, d_model]
}
```

ステップ2（Linear）とステップ3（GELU）を順番に適用するだけ。3行で完結する。

## Position-wise であることの意味

```
入力: [[tok0], [tok1], [tok2]]   (3トークン)

FFN は各行に独立に適用:
  FFN(tok0) → out0
  FFN(tok1) → out1
  FFN(tok2) → out2

出力: [[out0], [out1], [out2]]
```

行列演算の形で書くと `[seq_len, d_model]` の入力に対して同じ重み行列を掛けるので、自動的に各行が独立に処理される。特別な実装は不要。

テストでは同じ入力行を3つ並べたとき、3つの出力行が全て同一になることで position-wise 性を確認している。

## テスト

```
running 5 tests
test feed_forward::tests::test_forward_deterministic ... ok
test feed_forward::tests::test_forward_known_values ... ok
test feed_forward::tests::test_forward_position_wise ... ok
test feed_forward::tests::test_forward_shape ... ok
test feed_forward::tests::test_rand_shapes ... ok
```

- **known_values**: 手計算で `Linear1 → GELU → Linear2` の結果を検証
- **position_wise**: 同じ入力行は同じ出力行になる
- **shape**: `[seq_len, d_model]` の入出力サイズが一致

## 次回

Transformer Block を実装する。MHA と FFN を Layer Norm と残差接続で包んだ、Transformer の1層分の構造。
