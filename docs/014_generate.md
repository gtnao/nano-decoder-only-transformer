# 014: 推論ループ — テキストを自己回帰的に生成する

## 自己回帰生成とは

Decoder-Only Transformer は「これまでのトークン列が与えられたとき、次のトークンは何か」を予測するモデルである。テキスト生成は、この予測を繰り返し行うことで実現する。

```
"he"     → model → 次は "l" と予測
"hel"    → model → 次は "l" と予測
"hell"   → model → 次は "o" と予測
"hello"  → 完了
```

各ステップで、これまでの全トークンをモデルに入力し、最後の位置の出力だけを使って次のトークンを決定する。自分の出力を次の入力に使うため「自己回帰（auto-regressive）」と呼ぶ。

## トークンの選び方

モデルの出力は logits（各トークンの非正規化スコア）。ここからどのトークンを選ぶかにはいくつかの戦略がある。

### Greedy Decoding（argmax）

最もスコアの高いトークンを常に選ぶ。

```rust
pub fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap()
        .0
}
```

決定的で高速だが、常に「最も確率の高い1つ」しか選ばないため、出力が単調で繰り返しに陥りやすい。

### Temperature Sampling

logits を temperature で割ってから Softmax で確率分布に変換し、その分布からランダムにサンプリングする。

```
probs = softmax(logits / temperature)
```

| temperature | 振る舞い |
|---|---|
| → 0 | greedy に近づく（確実に最大を選ぶ） |
| 1.0 | モデル本来の確率分布でサンプリング |
| > 1.0 | 分布が平坦に → より多様（ランダム）な出力 |

```rust
pub fn sample_with_temperature(logits: &[f32], temperature: f32) -> usize {
    let scaled: Vec<f32> = logits.iter().map(|&v| v / temperature).collect();
    let probs = softmax(&Tensor::new(scaled, vec![logits.len()]));

    let mut rng = rand::rng();
    let r: f32 = rng.random_range(0.0..1.0);
    let mut cumulative = 0.0;
    for (i, &p) in probs.data.iter().enumerate() {
        cumulative += p;
        if r < cumulative {
            return i;
        }
    }
    probs.data.len() - 1
}
```

累積分布関数を使ったサンプリング。`[0, 1)` の一様乱数を生成し、累積確率がそれを超えた時点のインデックスを返す。

## 生成ループ

```rust
pub fn generate(
    model: &Transformer,
    tokenizer: &Tokenizer,
    prompt: &str,
    max_new_tokens: usize,
    temperature: f32,
) -> String {
    let mut token_ids = tokenizer.encode(prompt);

    for _ in 0..max_new_tokens {
        let logits = model.forward(&token_ids);

        // Take logits of the last position
        let vocab_size = logits.shape[1];
        let last_pos = logits.shape[0] - 1;
        let last_logits = &logits.data[last_pos * vocab_size..(last_pos + 1) * vocab_size];

        let next_id = if temperature <= 0.01 {
            argmax(last_logits)
        } else {
            sample_with_temperature(last_logits, temperature)
        };

        token_ids.push(next_id);
    }

    tokenizer.decode(&token_ids)
}
```

ポイント：
- `model.forward()` は全位置の logits `[seq_len, vocab_size]` を返すが、次トークン予測には最後の位置だけを使う
- 生成したトークンを `token_ids` に追加し、次のステップでは伸びた系列全体をモデルに再入力する
- temperature が 0.01 以下なら greedy、それ以上なら sampling

### KV Cache について

現在の実装では毎ステップ全系列を再計算している。実用的な推論では KV Cache（過去の Key/Value を保存して再計算を避ける）を使うが、学習目的の今回はシンプルさを優先して省略している。

## テスト

```
running 9 tests
test generate::tests::test_argmax_basic ... ok
test generate::tests::test_argmax_first ... ok
test generate::tests::test_argmax_last ... ok
test generate::tests::test_argmax_negative ... ok
test generate::tests::test_generate_greedy_deterministic ... ok
test generate::tests::test_generate_max_tokens ... ok
test generate::tests::test_generate_returns_string ... ok
test generate::tests::test_sample_low_temperature ... ok
test generate::tests::test_sample_returns_valid_index ... ok
```

- **argmax**: 基本ケース、先頭・末尾・負の値でも正しく最大を返す
- **sample_low_temperature**: 低 temperature で argmax と同等の動作
- **sample_returns_valid_index**: 有効なインデックスが返る
- **generate_returns_string**: プロンプトから始まる文字列が生成される
- **generate_max_tokens**: 生成トークン数が指定通り
- **generate_greedy_deterministic**: greedy 生成は決定的

## 次回

学習ループを実装する。Cross Entropy Loss の計算、逆伝播による勾配計算、パラメータ更新を行い、モデルが実際にテキストを学習できるようにする。
