# 036: SIMD 自動ベクトル化

## 背景

行列積は matrixmultiply で最適化したが、element-wise 演算（加算、スケーリング、softmax の exp、LayerNorm 等）は素朴なループのまま。これらも SIMD で高速化できる。

## アプローチ: コンパイラの自動ベクトル化

Rust コンパイラ（LLVM backend）は `--release` ビルドで自動ベクトル化を行う。条件:
- 単純なループ（分岐が少ない）
- データが連続メモリ上にある（`Vec<f32>` のスライス）
- ループ本体が短い

本プロジェクトの element-wise 演算は全てこの条件を満たしている。

## target-cpu=native

`.cargo/config.toml` に以下を追加:

```toml
[build]
rustflags = ["-C", "target-cpu=native"]
```

これにより、ビルドマシンの CPU が対応する最適な命令セットが使われる:

| CPU | 使用される命令セット |
|---|---|
| Intel/AMD (近年) | AVX2 (256-bit), FMA |
| Apple Silicon | NEON (128-bit) |
| 古い Intel | SSE4.2 (128-bit) |

### デフォルトとの違い

デフォルト (`target-cpu=generic`) では SSE2 (128-bit) のみ使用。`native` にすると:
- AVX2 対応 CPU: float 演算が最大 2x 高速（256-bit → 8 個の f32 同時処理）
- Apple Silicon: NEON はデフォルトで有効だが、追加の最適化が効く場合がある

## 自動ベクトル化されやすいパターン

本プロジェクトで使っているパターン:

```rust
// 1. zip + map — add, mul, add_inplace
for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
    *a += b;
}

// 2. iter_mut + scalar — scale_inplace
for v in self.data.iter_mut() {
    *v *= s;
}

// 3. map + collect — new Tensor creation
let data: Vec<f32> = self.data.iter().map(|v| v * s).collect();
```

これらはすべて LLVM が SIMD 命令に変換できる。

## 注意点

- `target-cpu=native` はビルドしたマシンでしか動かないバイナリを生成する
- 配布用バイナリには使わない（クロスコンパイル時は明示的に target-cpu を指定）
- `matrixmultiply` クレートは自前で CPU 検出して最適な命令を使うため、この設定の影響を受けにくい

## テスト

既存の 169 テストが全て通過。
