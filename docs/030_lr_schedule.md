# 030: 学習率スケジューリング

## なぜ必要か

Transformer の学習では固定学習率だと：

- **初期**: ランダムな重みからの大きな勾配で、パラメータが暴走しやすい
- **後半**: 学習率が大きすぎて、最適値の周りで振動して収束しない

学習率スケジューリングは、学習の段階に応じて学習率を動的に調整する。

## Linear Warmup + Cosine Decay

現代の Transformer 学習で標準的なスケジュール：

```
lr
│  ╱╲
│ ╱   ╲
│╱     ‾‾╲
│          ╲
└──────────────── step
  warmup  decay
```

### Warmup フェーズ（最初の10%）

```
lr(t) = base_lr × (t + 1) / warmup_steps
```

0 から `base_lr` まで線形に増加。初期の不安定な勾配に小さな学習率で対応。

### Cosine Decay フェーズ（残り90%）

```
lr(t) = base_lr × 0.5 × (1 + cos(π × progress))
```

`progress` は 0 → 1 に変化。cosine カーブで緩やかに 0 に向かう。

cosine decay の利点：
- 急激に下がらず、学習中盤でもそれなりの学習率を維持
- 線形 decay より最終的な性能が良いことが多い

## 実装

```rust
pub struct LRSchedule {
    pub base_lr: f32,
    pub warmup_steps: usize,
    pub total_steps: usize,
}

impl LRSchedule {
    pub fn get_lr(&self, step: usize) -> f32 {
        if step < self.warmup_steps {
            self.base_lr * (step + 1) as f32 / self.warmup_steps as f32
        } else {
            let progress = (step - self.warmup_steps) as f32
                / (self.total_steps - self.warmup_steps) as f32;
            self.base_lr * 0.5 * (1.0 + (PI * progress).cos())
        }
    }
}
```

## 学習ループへの統合

```rust
let schedule = LRSchedule::new(lr, warmup_steps, total_steps);
for step in 0..total_steps {
    adam.set_lr(schedule.get_lr(step));
    train_step(...);
}
```

Adam の `set_lr` メソッドでステップごとに学習率を更新する。Adam の内部状態（モーメント）はそのまま保持される。

## テスト

```
running 14 tests
test optimizer::tests::test_lr_schedule_warmup ... ok
test optimizer::tests::test_lr_schedule_cosine_decay ... ok
test optimizer::tests::test_lr_schedule_warmup_then_decay ... ok
(+ 11 other tests)
```

warmup の線形増加、cosine decay の減少、両方を合わせた単調性を検証。

## 次回

Dropout を実装する。
