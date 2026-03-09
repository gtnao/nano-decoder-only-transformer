# 033: 行列積の高速化 (matrixmultiply)

## 背景

Step 32 までの行列積は素朴な3重ループで実装していた。

```rust
for i in 0..m {
    for j in 0..n {
        let mut sum = 0.0;
        for k in 0..k1 {
            sum += a[i * k1 + k] * b[k * n + j];
        }
        c[i * n + j] = sum;
    }
}
```

モデルをスケールアップ（d_model=128, n_layers=4, d_ff=512, seq_len=128）すると、学習に **~90分** かかる状態になった。

## matrixmultiply クレート

`matrixmultiply` は純 Rust の最適化された行列積ライブラリ。

主な最適化手法:
- **SIMD 命令**: CPU のベクトル演算命令（SSE, AVX, NEON 等）を使い、複数の浮動小数点演算を1命令で実行
- **キャッシュタイリング**: 行列をキャッシュに収まるブロックに分割し、メモリアクセスの局所性を最大化
- **カーネル最適化**: ループアンローリング、レジスタ割り当ての最適化

システムライブラリ（OpenBLAS 等）への依存がなく、`cargo add` だけで使える。

## 変更箇所

`tensor.rs` の `matmul` メソッドのみ:

```rust
pub fn matmul(&self, other: &Tensor) -> Tensor {
    // ...assert...
    let mut data = vec![0.0_f32; m * n];
    unsafe {
        matrixmultiply::sgemm(
            m, k1, n,
            1.0,                                    // alpha
            self.data.as_ptr(), k1 as isize, 1,     // A: row-major
            other.data.as_ptr(), n as isize, 1,      // B: row-major
            0.0,                                     // beta
            data.as_mut_ptr(), n as isize, 1,        // C: row-major
        );
    }
    Tensor::new(data, vec![m, n])
}
```

### sgemm のパラメータ

`C = alpha * A @ B + beta * C`

stride パラメータ（`rsc`, `csc`）で行列のメモリレイアウトを指定:
- Row-major (C言語標準): `rsc = 列数, csc = 1`
- Column-major (Fortran標準): `rsc = 1, csc = 行数`

## 効果

| | 素朴な3重ループ | matrixmultiply |
|---|---|---|
| 1ステップ | ~0.74s | ~0.05s |
| ETA (7410 steps) | ~90分 | ~6分 |
| 速度比 | 1x | **~15x** |

## テスト

既存の 169 テストが全て通過。行列積の結果は同一なので、他のモジュールへの影響はない。
