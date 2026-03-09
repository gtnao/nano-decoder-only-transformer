# Nano Decoder-Only Transformer

Rust でフレームワークなし（rand のみ）で Decoder-Only Transformer をゼロから実装する学習プロジェクト。

## 実装ステップ

1. [x] Tensor基盤 — Tensor構造体, add, mul, transpose, matmul
2. [x] Linear層 — y = x @ W^T + b
3. [ ] 活性化関数 — ReLU, GELU
4. [ ] Softmax
5. [ ] Layer Normalization
6. [ ] Embedding
7. [ ] Positional Encoding
8. [ ] Scaled Dot-Product Attention + Causal Mask
9. [ ] Multi-Head Attention
10. [ ] Feed-Forward Network
11. [ ] Transformer Block — LayerNorm → MHA → 残差接続 → LayerNorm → FFN → 残差接続
12. [ ] Transformer全体 — Embedding → N × Block → LayerNorm → 出力Linear
13. [ ] トークナイザ（簡易版）
14. [ ] 推論ループ
15. [ ] 学習ループ — Cross Entropy Loss, 逆伝播, パラメータ更新

## 各ステップの進め方（必ずこの順序で行う）

1. テストを書く（構造体・シグネチャは `todo!()` で定義）
2. 実装する
3. テスト通過を確認する
4. `docs/NNN_xxx.md` に解説記事を書く
5. commit する
6. CLAUDE.md の該当ステップを `[x]` に更新する

**1つでも飛ばさないこと。**

## 開発方針

- TDD: テスト（と必要な構造体・シグネチャ）を先に書き、実装は後から埋める
- 外部crateは `rand` のみ。行列演算含め全て自前実装
- コード内コメントは英語
- ユーザーへの応答は日本語
