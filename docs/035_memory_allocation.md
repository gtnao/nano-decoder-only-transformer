# 035: メモリ確保の削減

## 背景

Tensor の演算はすべて新しい Vec を確保して返していた。1ステップの forward + backward で数百回のアロケーションが発生し、アロケータへの負荷とキャッシュ効率の低下を招いていた。

## 変更内容

### 1. in-place 演算の追加 (tensor.rs)

```rust
// Allocation-free element-wise addition
pub fn add_inplace(&mut self, other: &Tensor) {
    for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
        *a += b;
    }
}

// Allocation-free scalar multiplication
pub fn scale_inplace(&mut self, s: f32) {
    for v in self.data.iter_mut() {
        *v *= s;
    }
}
```

### 2. attention.rs — mask 加算の in-place 化

```rust
// Before: scores = scores.add(m);  // allocates new Tensor
// After:
scores.add_inplace(m);  // modifies in-place
```

forward と backward 両方で適用。各 head × 各ステップで 2 回の seq_len×seq_len アロケーションを削減。

### 3. softmax.rs — 中間 Vec の削除

```rust
// Before: exps 用の Vec を毎行確保
let exps: Vec<f32> = group.iter().map(|&v| (v - max_val).exp()).collect();

// After: 出力バッファに直接書き込み
let out = &mut data[start..end];
for (i, &v) in group.iter().enumerate() {
    out[i] = (v - max_val).exp();
    sum += out[i];
}
```

行数 × グループ数分の中間 Vec 確保を削減。

### 4. transformer_block.rs — 不要な clone の削除

```rust
// Before: dropout なしでも常に clone
let mut d_ffn_out = d_output.clone();
if let Some((_, ref mask2)) = masks {
    d_ffn_out = dropout_backward(&d_ffn_out, mask2, rate);
}

// After: dropout ありの場合のみ新規確保
let d_ffn_out = if let Some((_, ref mask2)) = masks {
    dropout_backward(d_output, mask2, rate)
} else {
    d_output.clone()
};
```

dropout 使用時は clone + dropout_backward の 2 回が dropout_backward の 1 回に。

## テスト

既存の 169 テストが全て通過。計算結果は変わらない。
