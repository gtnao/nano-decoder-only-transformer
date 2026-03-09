# Nano Decoder-Only Transformer

Rust でフレームワークなし（rand のみ）で Decoder-Only Transformer をゼロから実装する学習プロジェクト。

## 実装ステップ

1. [x] Tensor基盤 — Tensor構造体, add, mul, transpose, matmul
2. [x] Linear層 — y = x @ W^T + b
3. [x] 活性化関数 — ReLU, GELU
4. [x] Softmax
5. [x] Layer Normalization
6. [x] Embedding
7. [x] Positional Encoding
8. [x] Scaled Dot-Product Attention + Causal Mask
9. [x] Multi-Head Attention
10. [x] Feed-Forward Network
11. [x] Transformer Block — LayerNorm → MHA → 残差接続 → LayerNorm → FFN → 残差接続
12. [x] Transformer全体 — Embedding → N × Block → LayerNorm → 出力Linear
13. [x] トークナイザ（簡易版）
14. [x] 推論ループ
15. [x] Cross Entropy Loss
16. [x] 逆伝播の基盤 — 勾配テンソル、連鎖律の仕組み
17. [x] Linear の backward
18. [x] 活性化関数の backward — ReLU, GELU
19. [x] Softmax の backward
20. [x] Layer Normalization の backward
21. [x] Embedding の backward
22. [x] Attention の backward
23. [x] Multi-Head Attention の backward
24. [x] Feed-Forward Network の backward
25. [x] Transformer Block の backward
26. [x] Transformer 全体の backward
27. [x] パラメータ更新 — SGD or Adam
28. [x] 学習ループ — データ準備、ミニバッチ、エポック
29. [x] 勾配クリッピング
30. [x] 学習率スケジューリング — linear warmup + cosine decay
31. [x] Dropout
32. [ ] 日本語コーパスでの学習デモ

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
