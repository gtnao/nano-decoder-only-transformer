# 034: マルチコア並列化 (rayon)

## 背景

Step 33 で行列積を高速化したが、Attention の head ごとの計算はまだ逐次実行だった。各 head の計算は完全に独立しているため、マルチコアで並列実行できる。

## rayon クレート

`rayon` は Rust のデータ並列ライブラリ。`into_par_iter()` で通常のイテレータを並列イテレータに変換するだけで、work-stealing スケジューラが自動的にスレッドプールに分配する。

```rust
// Before: sequential
for h in 0..n_heads {
    let result = compute_head(h);
    // ...
}

// After: parallel
let results: Vec<_> = (0..n_heads)
    .into_par_iter()
    .map(|h| compute_head(h))
    .collect();
```

## 変更箇所

`multi_head_attention.rs` の3箇所の head ループを並列化:

### 1. forward の head 計算

```rust
let head_results: Vec<(usize, Tensor)> = (0..self.n_heads)
    .into_par_iter()
    .map(|h| {
        // Extract Q, K, V for head h
        // Run scaled_dot_product_attention
        (offset, attn_out)
    })
    .collect();
```

### 2. backward の forward 再計算

同様に `into_par_iter()` で並列化。

### 3. backward の勾配計算

```rust
let head_grads: Vec<(usize, Tensor, Tensor, Tensor)> = (0..self.n_heads)
    .into_par_iter()
    .map(|h| {
        // Extract slices, run attention_backward
        (offset, d_qh, d_kh, d_vh)
    })
    .collect();
```

## 並列化のパターン

1. 各 head の計算を `par_iter().map()` で並列実行
2. 結果を `collect()` で集約
3. 集約後にシーケンシャルに結合（scatter back）

scatter back は並列化しない。データ量が小さく、並列化のオーバーヘッドの方が大きいため。

## 効果

n_heads=4 の場合、理論上は最大4倍の高速化。ただし:
- head 数が少ないとスレッド起動のオーバーヘッドが相対的に大きい
- matmul が支配的な場合、head 並列化の効果は限定的
- コア数が head 数以上ある環境で最大効果

## テスト

既存の 169 テストが全て通過。並列化は計算結果を変えない（浮動小数点の加算順序も変わらない）。
