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
