# 007: Positional Encoding — Transformer に順序の概念を与える

## なぜ位置情報が必要か

RNN は入力を1トークンずつ順番に処理するため、構造的に順序情報を持つ。一方 Transformer の Attention は全トークンを同時に見るため、入力の順序を区別できない。

例えば「猫が魚を食べた」と「魚が猫を食べた」は同じトークン集合であり、Attention だけでは区別できない。Positional Encoding を Embedding に加算することで、各トークンの位置情報を注入する。

```
input = Embedding(token_ids) + PositionalEncoding
```

## 正弦波による Positional Encoding

"Attention Is All You Need" 論文で提案された方式。各位置 `pos` と各次元 `i` に対して、sin と cos の値を計算する。

```
PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
```

- 偶数次元には `sin`、奇数次元には `cos` を割り当てる
- 次元が深くなるほど `10000^(2i/d_model)` が大きくなり、周波数が低くなる

### 具体例（d_model=4）

```
pos=0: [sin(0), cos(0), sin(0), cos(0)] = [0, 1, 0, 1]
pos=1: [sin(1), cos(1), sin(0.01), cos(0.01)] ≈ [0.841, 0.540, 0.010, 1.000]
```

dim=0,1 は高周波（`sin(pos)`, `cos(pos)`）で位置ごとに大きく変化する。
dim=2,3 は低周波（`sin(pos/100)`, `cos(pos/100)`）でゆっくり変化する。

この多スケールの波の組み合わせにより、各位置が一意のパターンを持つ。

## なぜ sin/cos なのか

この方式には重要な性質がある。任意の固定オフセット `k` に対して、`PE(pos+k)` は `PE(pos)` の線形変換で表現できる。

```
sin(pos+k) = sin(pos)cos(k) + cos(pos)sin(k)
cos(pos+k) = cos(pos)cos(k) - sin(pos)sin(k)
```

つまりモデルは「2つ先のトークン」や「3つ前のトークン」といった相対的な位置関係を、線形演算で学習できる。

## 学習可能な Positional Encoding との比較

GPT-2 などでは、sin/cos の代わりに学習可能な Embedding を位置情報に使う。

| | 正弦波 | 学習可能 |
|---|---|---|
| パラメータ数 | 0 (固定) | seq_len × d_model |
| 未知の長さへの汎化 | 可能 | 学習した長さまで |
| 実装 | 数式から生成 | Embedding と同じ |

今回は正弦波方式を採用した。学習パラメータが不要でシンプル。

## 実装

```rust
pub fn positional_encoding(seq_len: usize, d_model: usize) -> Tensor {
    let mut data = vec![0.0_f32; seq_len * d_model];
    for pos in 0..seq_len {
        for i in 0..d_model {
            let angle = pos as f32
                / (10000.0_f32).powf((i / 2 * 2) as f32 / d_model as f32);
            data[pos * d_model + i] = if i % 2 == 0 {
                angle.sin()
            } else {
                angle.cos()
            };
        }
    }
    Tensor::new(data, vec![seq_len, d_model])
}
```

`i / 2 * 2` は整数除算の切り捨てを利用して、偶数次元のインデックスに丸める。`i=0,1` は同じ周波数で sin/cos のペア、`i=2,3` も同じ周波数のペア、という対応になる。

## テスト

```
running 6 tests
test positional_encoding::tests::test_deterministic ... ok
test positional_encoding::tests::test_different_positions_differ ... ok
test positional_encoding::tests::test_first_dim_is_sin ... ok
test positional_encoding::tests::test_position_zero ... ok
test positional_encoding::tests::test_shape ... ok
test positional_encoding::tests::test_values_in_range ... ok
```

- **position_zero**: pos=0 では `sin(0)=0`, `cos(0)=1` になる
- **values_in_range**: 全ての値が `[-1, 1]` に収まる（sin/cos の性質）
- **different_positions_differ**: 異なる位置は異なるベクトルを持つ
- **first_dim_is_sin**: dim=0 は `sin(pos)` であることを確認
- **deterministic**: 同じ引数で呼べば同じ結果（乱数なし）

## 次回

Scaled Dot-Product Attention と Causal Mask を実装する。Transformer の心臓部である。
