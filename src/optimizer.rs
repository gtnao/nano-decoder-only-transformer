/// Gradient clipping by global norm.
/// Computes L2 norm across all gradient slices. If it exceeds max_norm,
/// scales all gradients down so the total norm equals max_norm.
/// Returns the original norm before clipping.
pub fn clip_grad_norm(grads: &mut [&mut [f32]], max_norm: f32) -> f32 {
    // Compute global L2 norm
    let mut total_sq = 0.0_f32;
    for g in grads.iter() {
        for &v in g.iter() {
            total_sq += v * v;
        }
    }
    let norm = total_sq.sqrt();

    if norm > max_norm {
        let scale = max_norm / norm;
        for g in grads.iter_mut() {
            for v in g.iter_mut() {
                *v *= scale;
            }
        }
    }

    norm
}

/// Learning rate schedule: linear warmup + cosine decay.
pub struct LRSchedule {
    pub base_lr: f32,
    pub warmup_steps: usize,
    pub total_steps: usize,
}

impl LRSchedule {
    pub fn new(base_lr: f32, warmup_steps: usize, total_steps: usize) -> Self {
        Self { base_lr, warmup_steps, total_steps }
    }

    /// Compute learning rate at given step (0-indexed).
    pub fn get_lr(&self, step: usize) -> f32 {
        if step < self.warmup_steps {
            // Linear warmup: 0 -> base_lr
            self.base_lr * (step + 1) as f32 / self.warmup_steps as f32
        } else {
            // Cosine decay: base_lr -> 0
            let decay_steps = self.total_steps - self.warmup_steps;
            if decay_steps == 0 {
                return self.base_lr;
            }
            let progress = (step - self.warmup_steps) as f32 / decay_steps as f32;
            let progress = progress.min(1.0);
            self.base_lr * 0.5 * (1.0 + (std::f32::consts::PI * progress).cos())
        }
    }
}

/// SGD optimizer (Stochastic Gradient Descent)
pub struct SGD {
    pub lr: f32,
}

impl SGD {
    pub fn new(lr: f32) -> Self {
        Self { lr }
    }

    /// Update a single parameter group: param -= lr * grad
    pub fn update(&self, param: &mut [f32], grad: &[f32]) {
        assert_eq!(param.len(), grad.len());
        for (p, g) in param.iter_mut().zip(grad.iter()) {
            *p -= self.lr * g;
        }
    }
}

/// Adam optimizer
pub struct Adam {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub t: usize,
    m: Vec<Vec<f32>>, // first moment per parameter group
    v: Vec<Vec<f32>>, // second moment per parameter group
}

impl Adam {
    /// Create Adam optimizer. param_sizes: size of each parameter group.
    pub fn new(lr: f32, param_sizes: &[usize]) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            t: 0,
            m: param_sizes.iter().map(|&s| vec![0.0; s]).collect(),
            v: param_sizes.iter().map(|&s| vec![0.0; s]).collect(),
        }
    }

    /// Call once per training step before updating parameters.
    pub fn begin_step(&mut self) {
        self.t += 1;
    }

    /// Set learning rate (for use with LRSchedule).
    pub fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }

    /// Update parameter group `idx`.
    pub fn update(&mut self, idx: usize, param: &mut [f32], grad: &[f32]) {
        assert_eq!(param.len(), grad.len());
        let t = self.t as i32;
        let m = &mut self.m[idx];
        let v = &mut self.v[idx];
        let bias_correction1 = 1.0 - self.beta1.powi(t);
        let bias_correction2 = 1.0 - self.beta2.powi(t);

        for i in 0..param.len() {
            m[i] = self.beta1 * m[i] + (1.0 - self.beta1) * grad[i];
            v[i] = self.beta2 * v[i] + (1.0 - self.beta2) * grad[i] * grad[i];
            let m_hat = m[i] / bias_correction1;
            let v_hat = v[i] / bias_correction2;
            param[i] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== LRSchedule ====================

    #[test]
    fn test_lr_schedule_warmup() {
        let sched = LRSchedule::new(0.001, 10, 100);
        // Step 0: lr = 0.001 * 1/10 = 0.0001
        assert!((sched.get_lr(0) - 0.0001).abs() < 1e-7);
        // Step 4: lr = 0.001 * 5/10 = 0.0005
        assert!((sched.get_lr(4) - 0.0005).abs() < 1e-7);
        // Step 9: lr = 0.001 * 10/10 = 0.001 (peak)
        assert!((sched.get_lr(9) - 0.001).abs() < 1e-7);
    }

    #[test]
    fn test_lr_schedule_cosine_decay() {
        let sched = LRSchedule::new(0.001, 0, 100);
        // Step 0: full lr
        assert!((sched.get_lr(0) - 0.001).abs() < 1e-6);
        // Step 50: half lr (cosine at pi/2)
        assert!((sched.get_lr(50) - 0.0005).abs() < 1e-4);
        // Step 99: near 0
        assert!(sched.get_lr(99) < 0.0001);
    }

    #[test]
    fn test_lr_schedule_warmup_then_decay() {
        let sched = LRSchedule::new(0.01, 10, 110);
        // Warmup phase: monotonically increasing
        for i in 0..9 {
            assert!(sched.get_lr(i) < sched.get_lr(i + 1));
        }
        // Peak at end of warmup
        assert!((sched.get_lr(9) - 0.01).abs() < 1e-6);
        // Decay phase: monotonically decreasing
        for i in 10..109 {
            assert!(sched.get_lr(i) >= sched.get_lr(i + 1));
        }
    }

    // ==================== clip_grad_norm ====================

    #[test]
    fn test_clip_grad_norm_no_clip() {
        // norm = sqrt(1+4+9) = sqrt(14) ≈ 3.74, max_norm=5.0 => no clipping
        let mut g = vec![1.0, 2.0, 3.0];
        let norm = clip_grad_norm(&mut [&mut g[..]], 5.0);
        assert!((norm - 14.0_f32.sqrt()).abs() < 1e-5);
        assert!((g[0] - 1.0).abs() < 1e-6); // unchanged
    }

    #[test]
    fn test_clip_grad_norm_clips() {
        // norm = sqrt(9+16) = 5.0, max_norm=1.0 => scale by 1/5
        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut [&mut g[..]], 1.0);
        assert!((norm - 5.0).abs() < 1e-5);
        assert!((g[0] - 0.6).abs() < 1e-5);
        assert!((g[1] - 0.8).abs() < 1e-5);
    }

    #[test]
    fn test_clip_grad_norm_multiple_slices() {
        // Two slices: [3.0] and [4.0], global norm = 5.0, max_norm=2.5
        let mut g1 = vec![3.0];
        let mut g2 = vec![4.0];
        clip_grad_norm(&mut [&mut g1[..], &mut g2[..]], 2.5);
        // scale = 2.5 / 5.0 = 0.5
        assert!((g1[0] - 1.5).abs() < 1e-5);
        assert!((g2[0] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_clip_grad_norm_preserves_direction() {
        let mut g = vec![3.0, 4.0];
        clip_grad_norm(&mut [&mut g[..]], 1.0);
        // After clipping, ratio should be preserved: g[0]/g[1] = 3/4
        assert!((g[0] / g[1] - 0.75).abs() < 1e-5);
        // New norm should be max_norm
        let new_norm = (g[0] * g[0] + g[1] * g[1]).sqrt();
        assert!((new_norm - 1.0).abs() < 1e-5);
    }

    // ==================== SGD ====================

    #[test]
    fn test_sgd_basic() {
        let sgd = SGD::new(0.1);
        let mut param = vec![1.0, 2.0, 3.0];
        let grad = vec![0.5, -1.0, 2.0];
        sgd.update(&mut param, &grad);
        // param = [1.0 - 0.1*0.5, 2.0 - 0.1*(-1.0), 3.0 - 0.1*2.0]
        //       = [0.95, 2.1, 2.8]
        assert!((param[0] - 0.95).abs() < 1e-6);
        assert!((param[1] - 2.1).abs() < 1e-6);
        assert!((param[2] - 2.8).abs() < 1e-6);
    }

    #[test]
    fn test_sgd_zero_gradient() {
        let sgd = SGD::new(0.1);
        let mut param = vec![1.0, 2.0];
        let grad = vec![0.0, 0.0];
        sgd.update(&mut param, &grad);
        assert_eq!(param, vec![1.0, 2.0]);
    }

    #[test]
    fn test_sgd_convergence() {
        // Minimize f(x) = x^2, gradient = 2x
        let sgd = SGD::new(0.1);
        let mut param = vec![5.0];
        for _ in 0..100 {
            let grad = vec![2.0 * param[0]];
            sgd.update(&mut param, &grad);
        }
        assert!(param[0].abs() < 1e-4, "param should converge to 0, got {}", param[0]);
    }

    // ==================== Adam ====================

    #[test]
    fn test_adam_basic() {
        let mut adam = Adam::new(0.01, &[3]);
        let mut param = vec![1.0, 2.0, 3.0];
        let grad = vec![0.5, -1.0, 2.0];
        adam.begin_step();
        adam.update(0, &mut param, &grad);
        // After one step, params should have moved
        assert!(param[0] < 1.0); // positive gradient => param decreases
        assert!(param[1] > 2.0); // negative gradient => param increases
        assert!(param[2] < 3.0); // positive gradient => param decreases
    }

    #[test]
    fn test_adam_convergence() {
        // Minimize f(x) = x^2, gradient = 2x
        let mut adam = Adam::new(0.01, &[1]);
        let mut param = vec![5.0];
        for _ in 0..2000 {
            let grad = vec![2.0 * param[0]];
            adam.begin_step();
            adam.update(0, &mut param, &grad);
        }
        assert!(param[0].abs() < 0.5, "param should converge near 0, got {}", param[0]);
    }

    #[test]
    fn test_adam_multiple_params() {
        let mut adam = Adam::new(0.01, &[2, 3]);
        let mut p1 = vec![1.0, 2.0];
        let mut p2 = vec![3.0, 4.0, 5.0];
        let g1 = vec![0.1, 0.2];
        let g2 = vec![0.3, 0.4, 0.5];

        adam.begin_step();
        adam.update(0, &mut p1, &g1);
        adam.update(1, &mut p2, &g2);

        // All params should have changed
        assert!(p1[0] < 1.0);
        assert!(p2[2] < 5.0);
    }

    #[test]
    fn test_adam_bias_correction() {
        // First few steps should have larger effective learning rate
        // due to bias correction when m and v are near zero
        let mut adam = Adam::new(0.001, &[1]);
        let mut param1 = vec![1.0];
        let grad = vec![1.0];

        adam.begin_step();
        adam.update(0, &mut param1, &grad);
        let step1_delta = (1.0 - param1[0]).abs();

        // Step 1 delta should be approximately lr (due to bias correction)
        assert!(
            (step1_delta - 0.001).abs() < 0.0005,
            "first step delta {} should be ~0.001", step1_delta
        );
    }
}
