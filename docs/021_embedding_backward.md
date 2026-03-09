# 021: Embedding の backward

## Embedding の復習

Forward は「weight テーブルから token_id に対応する行をコピーする」操作：

```
forward(token_ids) -> weight[token_ids[0]], weight[token_ids[1]], ...
```

入力は離散的な整数（token ID）なので、入力に対する勾配は存在しない。backward で求めるのは `d_weight` のみ。

## d_weight の導出

Forward では weight の特定の行がそのまま出力にコピーされる。逆に backward では、上流の勾配 `d_output` の各行を、対応する token_id の weight 行に「戻す」操作になる。

```
d_weight[id] += d_output[t]   (for each t where token_ids[t] == id)
```

これは scatter-add（分散加算）と呼ばれるパターン。同じ token_id が複数回出現した場合、その勾配が加算される。

### 具体例

```
vocab_size=3, embed_dim=2
token_ids = [0, 0, 1]
d_output = [[0.1, 0.2],   <- position 0, id=0
             [0.3, 0.4],   <- position 1, id=0
             [0.5, 0.6]]   <- position 2, id=1

d_weight = [[0.1+0.3, 0.2+0.4],   <- id 0: two contributions
            [0.5,     0.6    ],   <- id 1: one contribution
            [0.0,     0.0    ]]   <- id 2: unused
         = [[0.4, 0.6],
            [0.5, 0.6],
            [0.0, 0.0]]
```

## 実装

```rust
pub fn backward(&self, d_output: &Tensor, token_ids: &[usize]) -> Tensor {
    let vocab_size = self.weight.shape[0];
    let embed_dim = self.weight.shape[1];
    let mut d_weight = vec![0.0_f32; vocab_size * embed_dim];

    for (t, &id) in token_ids.iter().enumerate() {
        let src = t * embed_dim;
        let dst = id * embed_dim;
        for i in 0..embed_dim {
            d_weight[dst + i] += d_output.data[src + i];
        }
    }

    Tensor::new(d_weight, vec![vocab_size, embed_dim])
}
```

ゼロ初期化した `d_weight` に対して、各 token 位置の勾配を対応する行に加算していく。forward の逆操作として自然な形になっている。

## Linear の backward との違い

Linear の backward は行列積（`d_output^T @ input`）で計算できたが、Embedding は離散的なルックアップなので行列演算ではなく、インデックスベースの scatter-add になる。

## テスト

```
running 12 tests
test embedding::tests::test_backward_duplicate_ids ... ok
test embedding::tests::test_backward_multiple ... ok
test embedding::tests::test_backward_numerical ... ok
test embedding::tests::test_backward_shape ... ok
test embedding::tests::test_backward_simple ... ok
(+ 7 forward tests)
```

重複 ID での加算の正しさと、数値微分との照合の両方で検証。

## 次回

Attention の backward を実装する。
