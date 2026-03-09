# 001: Tensor の基礎 — 多次元配列を1次元で表現する

Decoder-Only Transformer をゼロから Rust で実装するシリーズの第1回。
まずは全ての土台となる **Tensor（テンソル）** を実装する。

## Tensor とは何か

Tensor は「任意の次元を持つ数値の配列」の総称である。

| 次元数 (rank) | 呼び名       | 例                     |
| ------------- | ------------ | ---------------------- |
| 0             | スカラー     | `3.14`                 |
| 1             | ベクトル     | `[1, 2, 3]`           |
| 2             | 行列         | `[[1, 2], [3, 4]]`    |
| 3             | 3階テンソル  | バッチ × 系列長 × 次元 |

Transformer の内部では、データは主に3階テンソル `[batch_size, seq_len, d_model]` として流れる。
これを扱うための基盤が必要になる。

## データ構造の設計

### なぜ `Vec<f32>` + `Vec<usize>` なのか

多次元配列を素直に表現すると `Vec<Vec<Vec<f32>>>` のようなネストになるが、これには問題がある。

- 各行が別々のヒープ領域に配置され、メモリが不連続になる
- キャッシュ効率が悪く、大きな行列演算で遅くなる
- 次元が増えるたびに型が変わり、汎用的に扱えない

そこで、NumPy や PyTorch と同じく **1次元の連続メモリ + shape メタデータ** で表現する。

```rust
#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: Vec<f32>,   // 実データ（1次元、連続メモリ）
    pub shape: Vec<usize>, // 各次元のサイズ
}
```

論理的に 2×3 の行列：

```
[[1, 2, 3],
 [4, 5, 6]]
```

物理的にはこう格納される（row-major order）：

```
data:  [1, 2, 3, 4, 5, 6]
shape: [2, 3]
```

## コンストラクタ

```rust
pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
    let expected_len: usize = shape.iter().product();
    assert_eq!(data.len(), expected_len);
    Self { data, shape }
}

pub fn zeros(shape: Vec<usize>) -> Self {
    let len: usize = shape.iter().product();
    Self {
        data: vec![0.0; len],
        shape,
    }
}
```

`shape` の全要素の積がデータの総数と一致しなければならない。
`[2, 3]` なら `2 × 3 = 6` 個の要素が必要。

## 多次元インデックスから1次元位置への変換

これが Tensor 実装の核心部分である。

`shape: [2, 3]` の行列で `[1, 2]`（2行目の3列目）にアクセスしたい場合：

```
flat_index = 1 * 3 + 2 = 5
data[5] → 6.0
```

一般化すると、`shape: [d0, d1, ..., dn]` のテンソルで `indices: [i0, i1, ..., in]` のとき：

```
flat_index = i0 * (d1 * d2 * ... * dn)
           + i1 * (d2 * d3 * ... * dn)
           + ...
           + in
```

各次元の「stride（刻み幅）」を末尾から掛け合わせて計算する。

```rust
fn flat_index(&self, indices: &[usize]) -> usize {
    let mut idx = 0;
    let mut stride = 1;
    for i in (0..self.shape.len()).rev() {  // 末尾の次元から
        idx += indices[i] * stride;
        stride *= self.shape[i];
    }
    idx
}
```

### 3階テンソルでの具体例

`shape: [2, 2, 2]` のテンソル：

```
[[[1, 2],    data: [1, 2, 3, 4, 5, 6, 7, 8]
  [3, 4]],
 [[5, 6],
  [7, 8]]]
```

`[1, 0, 1]`（2番目のブロック、1行目、2列目）にアクセス：

```
stride の計算（後ろから）:
  dim 2: stride = 1
  dim 1: stride = 1 * 2 = 2
  dim 0: stride = 2 * 2 = 4

flat_index = 1 * 4 + 0 * 2 + 1 * 1 = 5
data[5] → 6.0 ✓
```

## 要素ごとの演算: add / mul

同じ shape のテンソル同士なら、`data` を要素ごとに演算するだけでよい。

```rust
pub fn add(&self, other: &Tensor) -> Tensor {
    assert_eq!(self.shape, other.shape);
    let data = self.data.iter()
        .zip(other.data.iter())
        .map(|(a, b)| a + b)
        .collect();
    Tensor::new(data, self.shape.clone())
}
```

物理メモリ上の並び順が同じなので、多次元のインデックス計算は一切不要。
`zip` で対応する要素同士をペアにして演算する。

## 転置 (transpose)

2D 行列の転置は、行と列を入れ替える操作。

```
[[1, 2, 3],     [[1, 4],
 [4, 5, 6]]  →   [2, 5],
                  [3, 6]]
```

row-major order では、元の `[r, c]` の位置にある要素を `[c, r]` に配置し直す。

```rust
pub fn transpose(&self) -> Tensor {
    let (rows, cols) = (self.shape[0], self.shape[1]);
    let mut data = vec![0.0; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            data[c * rows + r] = self.data[r * cols + c];
        }
    }
    Tensor::new(data, vec![cols, rows])
}
```

元の位置 `r * cols + c` から、転置後の位置 `c * rows + r` へコピーする。
shape は `[rows, cols]` → `[cols, rows]` に変わる。

## 行列積 (matmul)

行列積は Transformer で最も頻繁に実行される演算である。
Attention の `QK^T` や `(Attention)V`、全ての Linear 層で使われる。

`A: [m, k]` × `B: [k, n]` → `C: [m, n]` で、各要素は：

```
C[i][j] = Σ (k=0..k-1) A[i][k] * B[k][j]
```

```rust
pub fn matmul(&self, other: &Tensor) -> Tensor {
    let (m, k1) = (self.shape[0], self.shape[1]);
    let (k2, n) = (other.shape[0], other.shape[1]);
    assert_eq!(k1, k2);
    let mut data = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..k1 {
                sum += self.data[i * k1 + k] * other.data[k * n + j];
            }
            data[i * n + j] = sum;
        }
    }
    Tensor::new(data, vec![m, n])
}
```

これはナイーブな三重ループ実装（計算量 `O(m * n * k)`）。
学習目的にはこれで十分だが、性能を追求する場合はループ順序の変更（キャッシュ効率改善）、SIMD、タイリングなどの最適化がある。

## テスト

全ての関数に対してテストを書き、TDD で実装を進めた。

```
running 13 tests
test tensor::tests::test_add ... ok
test tensor::tests::test_add_shape_mismatch - should panic ... ok
test tensor::tests::test_get_2d ... ok
test tensor::tests::test_get_3d ... ok
test tensor::tests::test_matmul ... ok
test tensor::tests::test_matmul_non_square ... ok
test tensor::tests::test_matmul_shape_mismatch - should panic ... ok
test tensor::tests::test_mul ... ok
test tensor::tests::test_new ... ok
test tensor::tests::test_new_shape_mismatch - should panic ... ok
test tensor::tests::test_set ... ok
test tensor::tests::test_transpose ... ok
test tensor::tests::test_zeros ... ok
```

正常系だけでなく、shape の不一致時に panic することも `#[should_panic]` で検証している。

## 次回

Phase 2 では、この Tensor の上にニューラルネットワークの部品を積み上げていく。

- **Linear 層** — `y = x @ W^T + b`
- **活性化関数** — ReLU, GELU
- **Softmax** — `exp(x_i) / Σexp(x_j)`
- **Layer Normalization** — `(x - μ) / σ * γ + β`
- **Embedding** — トークン ID からベクトルへの変換
