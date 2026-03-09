# 006: Embedding — 整数のトークンIDをベクトルに変換する

## Embedding とは

ニューラルネットワークは浮動小数点数で計算を行う。しかし入力は「単語」や「文字」といった離散的なトークンであり、まず整数のID（例: `"hello"→42`）で表現される。

Embedding はこの整数IDを固定長の実数ベクトルに変換するルックアップテーブルである。

```
vocab_size = 3, embed_dim = 2 のとき:

weight = [[1.0, 2.0],   ← id 0
          [3.0, 4.0],   ← id 1
          [5.0, 6.0]]   ← id 2

forward([1, 2, 0]) → [[3.0, 4.0],
                       [5.0, 6.0],
                       [1.0, 2.0]]
```

各IDに対応する行をそのまま取り出すだけ。計算は行わない。

## なぜ学習可能なのか

Embedding の weight テーブルは学習によって更新される。最初はランダムな値だが、学習が進むにつれて意味的に近い単語のベクトルが近くに配置されるようになる。

例えば「king」と「queen」のベクトルは近くに、「king」と「apple」のベクトルは遠くに配置される。この性質がモデルに言語の意味構造を理解させる基盤になる。

## 実装

forward は weight テーブルから該当行をコピーするだけである。

```rust
pub fn forward(&self, token_ids: &[usize]) -> Tensor {
    let vocab_size = self.weight.shape[0];
    let embed_dim = self.weight.shape[1];
    let mut data = Vec::with_capacity(token_ids.len() * embed_dim);
    for &id in token_ids {
        assert!(id < vocab_size);
        let start = id * embed_dim;
        data.extend_from_slice(&self.weight.data[start..start + embed_dim]);
    }
    Tensor::new(data, vec![token_ids.len(), embed_dim])
}
```

row-major order で格納されているため、ID `i` の埋め込みベクトルは `data[i * embed_dim..(i+1) * embed_dim]` のスライスに対応する。`extend_from_slice` で連続メモリをそのままコピーできる。

### 初期化

ランダム初期化では小さな範囲 `[-0.02, 0.02)` の一様分布を使う。これは GPT-2 の初期化に倣ったもの。

## Transformer での位置

Embedding は Transformer の最初の層であり、入力の入口となる。

```
token_ids → [Embedding] → [+ Positional Encoding] → Transformer Blocks → ...
```

出力の shape は `[seq_len, embed_dim]` で、これが `d_model` として Transformer 全体を流れていく。

## テスト

```
running 7 tests
test embedding::tests::test_forward_duplicate_ids ... ok
test embedding::tests::test_forward_multiple ... ok
test embedding::tests::test_forward_out_of_range - should panic ... ok
test embedding::tests::test_forward_single ... ok
test embedding::tests::test_new ... ok
test embedding::tests::test_new_not_2d - should panic ... ok
test embedding::tests::test_rand_shape ... ok
```

- **single / multiple**: 1個および複数のトークンIDで正しい行が取り出される
- **duplicate_ids**: 同じIDが複数回出ても正しくコピーされる
- **out_of_range**: vocab_size を超えるIDで panic

## 次回

Positional Encoding を実装する。Transformer は入力の順序情報を持たないため、各位置に固有のベクトルを加算して順序を伝える。
