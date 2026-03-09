# 038: BPE トークナイザ

## 背景

文字レベルのトークナイザでは、日本語の文法パターン（助詞、活用語尾）を学ぶのに非常に多くのステップが必要だった。seq_len=128文字でも実質40〜60語分しかカバーできない。

## BPE (Byte Pair Encoding) とは

テキストをサブワード単位に分割するアルゴリズム。GPT-2/3/4 で使用されている。

### 学習アルゴリズム

```
1. 初期語彙 = 全ユニーク文字
2. コーパスで最も頻出する隣接2トークンのペアを見つける
3. そのペアを1つの新トークンとして語彙に追加
4. コーパス中の全ての該当ペアをマージ
5. 2-4を目標語彙サイズまで繰り返す
```

例（「ましたましたました」）:
```
初期: [ま, し, た, ま, し, た, ま, し, た]
ペア頻度: (ま,し)=3, (し,た)=3, (た,ま)=2
マージ1: ま+し → まし → [まし, た, まし, た, まし, た]
マージ2: まし+た → ました → [ました, ました, ました]
```

### エンコード（推論時）

学習で得たマージルールを優先順に適用:

```
"ました" → [ま, し, た] → [まし, た] → [ました]
```

## 実装

### bpe_tokenizer.rs

```rust
pub struct BpeTokenizer {
    merges: Vec<(String, String)>,      // Ordered merge rules
    token_to_id: HashMap<String, usize>, // Token -> ID
    id_to_token: Vec<String>,            // ID -> Token
}
```

主要メソッド:
- `train(corpus, vocab_size)` — コーパスからBPE語彙を構築
- `encode(text)` — テキストをマージルール適用でトークンID列に変換
- `decode(ids)` — トークンID列を文字列に復元

### TokenizerTrait

```rust
pub trait TokenizerTrait {
    fn vocab_size(&self) -> usize;
    fn encode(&self, text: &str) -> Vec<usize>;
    fn decode(&self, ids: &[usize]) -> String;
}
```

`Tokenizer`（文字レベル）と `BpeTokenizer` の両方がこのトレイトを実装。`generate()` と `train()` はトレイトオブジェクトを受け取るため、トークナイザの切り替えが容易。

## 効果

vocab_size=1000 の場合:
- 文字レベル: 31K文字 → 31Kトークン、seq_len=128で128文字分の文脈
- BPE: 31K文字 → ~15K-20Kトークン、seq_len=128で200-400文字分の文脈

同じ seq_len でより広い文脈を見られるため、文法パターンの学習が効率的になる。

## テスト

```
181 tests passed (+9 new)
- test_train_basic
- test_train_merges_most_frequent
- test_encode_decode_roundtrip
- test_encode_unknown_chars
- test_encode_empty
- test_merges_reduce_tokens
- test_vocab_size_limit
- test_japanese_roundtrip
- test_japanese_merges
```
