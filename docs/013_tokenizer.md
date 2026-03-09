# 013: トークナイザ — 文字列とトークンIDの相互変換

## トークナイザの役割

Transformer は整数のトークンID列を入力として受け取る。人間が読む文字列とモデルが処理するID列を相互変換するのがトークナイザの仕事。

```
"hello" → [4, 3, 7, 7, 8] → Transformer → logits
```

## 文字レベル vs サブワード

| 方式 | 単位 | 語彙数 | 例 |
|---|---|---|---|
| 文字レベル | 1文字 | 数十〜数百 | h, e, l, l, o |
| サブワード (BPE) | 可変長 | 数万 | hel, lo |
| 単語レベル | 1単語 | 数十万 | hello |

実用的な LLM は BPE（Byte Pair Encoding）などのサブワード方式を使うが、今回は学習目的なので最もシンプルな**文字レベル**を採用。

文字レベルの利点：
- 実装が簡単（数十行）
- 未知語がほぼ発生しない
- 語彙数が小さいのでモデルも小さく済む

欠点：
- 系列長が長くなる（「hello」が5トークン vs BPEなら1〜2トークン）
- 文字間の意味的関係をモデルが学ぶ必要がある

## 特殊トークン

| ID | トークン | 用途 |
|---|---|---|
| 0 | `<PAD>` | パディング（バッチ処理で長さを揃える） |
| 1 | `<UNK>` | 未知文字（語彙に含まれない文字） |
| 2〜 | 実文字 | コーパスから収集した文字 |

## 語彙の構築

コーパス（学習テキスト）から一意な文字を収集し、ソートして決定的な順序を保証する。

```rust
pub fn from_corpus(corpus: &str) -> Self {
    let mut chars: Vec<char> = corpus.chars()
        .collect::<BTreeSet<_>>()  // 重複排除 + ソート
        .into_iter()
        .collect();

    let mut id_to_char = vec!['\0', '\u{FFFD}'];  // PAD, UNK
    id_to_char.append(&mut chars);
    // ...
}
```

`BTreeSet` を使うことで、コーパスの文字出現順序に関係なく同じ語彙が生成される。

## エンコード / デコード

エンコードは各文字を辞書で引くだけ。未知文字は `UNK=1` に置き換える。

```rust
pub fn encode(&self, text: &str) -> Vec<usize> {
    let unk_id = 1;
    text.chars()
        .map(|c| *self.char_to_id.get(&c).unwrap_or(&unk_id))
        .collect()
}
```

デコードはIDから文字への逆引き。

```rust
pub fn decode(&self, ids: &[usize]) -> String {
    ids.iter()
        .map(|&id| {
            if id < self.id_to_char.len() {
                self.id_to_char[id]
            } else {
                self.id_to_char[1] // UNK
            }
        })
        .collect()
}
```

文字レベルなので、encode → decode のラウンドトリップが（未知文字を除き）完全に可逆。

## テスト

```
running 9 tests
test tokenizer::tests::test_corpus_with_spaces_and_newlines ... ok
test tokenizer::tests::test_decode_roundtrip ... ok
test tokenizer::tests::test_decode_special_tokens ... ok
test tokenizer::tests::test_encode_basic ... ok
test tokenizer::tests::test_encode_empty ... ok
test tokenizer::tests::test_encode_unknown_char ... ok
test tokenizer::tests::test_from_corpus ... ok
test tokenizer::tests::test_from_corpus_dedup ... ok
test tokenizer::tests::test_sorted_vocab ... ok
```

- **dedup**: 重複文字がある場合でも語彙数が正しい
- **unknown_char**: 語彙にない文字は UNK(=1) になる
- **roundtrip**: encode → decode で元の文字列に戻る
- **sorted_vocab**: コーパスの文字順に依存しない決定的な語彙

## 次回

推論ループを実装する。モデルの出力 logits から次のトークンを選び、自己回帰的にテキストを生成する。
