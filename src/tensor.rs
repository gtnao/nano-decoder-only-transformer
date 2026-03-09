#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let expected_len: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            expected_len,
            "data length {} does not match shape {:?} (expected {})",
            data.len(),
            shape,
            expected_len,
        );
        Self { data, shape }
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let len: usize = shape.iter().product();
        Self {
            data: vec![0.0; len],
            shape,
        }
    }

    fn flat_index(&self, indices: &[usize]) -> usize {
        assert_eq!(indices.len(), self.shape.len());
        let mut idx = 0;
        let mut stride = 1;
        for i in (0..self.shape.len()).rev() {
            assert!(indices[i] < self.shape[i]);
            idx += indices[i] * stride;
            stride *= self.shape[i];
        }
        idx
    }

    pub fn get(&self, indices: &[usize]) -> f32 {
        self.data[self.flat_index(indices)]
    }

    pub fn set(&mut self, indices: &[usize], value: f32) {
        let idx = self.flat_index(indices);
        self.data[idx] = value;
    }

    // element-wise addition
    pub fn add(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape, "shape mismatch for add");
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        Tensor::new(data, self.shape.clone())
    }

    // element-wise multiplication
    pub fn mul(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape, "shape mismatch for mul");
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .collect();
        Tensor::new(data, self.shape.clone())
    }

    // 2D transpose
    pub fn transpose(&self) -> Tensor {
        assert_eq!(self.shape.len(), 2, "transpose requires 2D tensor");
        let (rows, cols) = (self.shape[0], self.shape[1]);
        let mut data = vec![0.0; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                data[c * rows + r] = self.data[r * cols + c];
            }
        }
        Tensor::new(data, vec![cols, rows])
    }

    // 2D matrix multiplication (optimized via matrixmultiply crate)
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape.len(), 2, "matmul requires 2D tensor");
        assert_eq!(other.shape.len(), 2, "matmul requires 2D tensor");
        let (m, k1) = (self.shape[0], self.shape[1]);
        let (k2, n) = (other.shape[0], other.shape[1]);
        assert_eq!(k1, k2, "matmul inner dimensions mismatch: {} vs {}", k1, k2);
        let mut data = vec![0.0_f32; m * n];
        unsafe {
            matrixmultiply::sgemm(
                m, k1, n,
                1.0,
                self.data.as_ptr(), k1 as isize, 1,
                other.data.as_ptr(), n as isize, 1,
                0.0,
                data.as_mut_ptr(), n as isize, 1,
            );
        }
        Tensor::new(data, vec![m, n])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== constructor ====================

    #[test]
    fn test_new() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        assert_eq!(t.shape, vec![2, 3]);
        assert_eq!(t.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    #[should_panic]
    fn test_new_shape_mismatch() {
        Tensor::new(vec![1.0, 2.0, 3.0], vec![2, 3]);
    }

    #[test]
    fn test_zeros() {
        let t = Tensor::zeros(vec![2, 3]);
        assert_eq!(t.shape, vec![2, 3]);
        assert_eq!(t.data, vec![0.0; 6]);
    }

    // ==================== get / set ====================

    #[test]
    fn test_get_2d() {
        // [[1, 2, 3],
        //  [4, 5, 6]]
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        assert_eq!(t.get(&[0, 0]), 1.0);
        assert_eq!(t.get(&[0, 2]), 3.0);
        assert_eq!(t.get(&[1, 0]), 4.0);
        assert_eq!(t.get(&[1, 2]), 6.0);
    }

    #[test]
    fn test_get_3d() {
        // [[[1, 2], [3, 4]],
        //  [[5, 6], [7, 8]]]
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 2, 2]);
        assert_eq!(t.get(&[0, 0, 0]), 1.0);
        assert_eq!(t.get(&[0, 1, 1]), 4.0);
        assert_eq!(t.get(&[1, 0, 0]), 5.0);
        assert_eq!(t.get(&[1, 1, 1]), 8.0);
    }

    #[test]
    fn test_set() {
        let mut t = Tensor::zeros(vec![2, 3]);
        t.set(&[1, 2], 9.0);
        assert_eq!(t.get(&[1, 2]), 9.0);
    }

    // ==================== add ====================

    #[test]
    fn test_add() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::new(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2]);
        let c = a.add(&b);
        assert_eq!(c.shape, vec![2, 2]);
        assert_eq!(c.data, vec![11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    #[should_panic]
    fn test_add_shape_mismatch() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let b = Tensor::new(vec![1.0, 2.0], vec![2]);
        a.add(&b);
    }

    // ==================== mul ====================

    #[test]
    fn test_mul() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::new(vec![2.0, 3.0, 4.0, 5.0], vec![2, 2]);
        let c = a.mul(&b);
        assert_eq!(c.shape, vec![2, 2]);
        assert_eq!(c.data, vec![2.0, 6.0, 12.0, 20.0]);
    }

    // ==================== transpose ====================

    #[test]
    fn test_transpose() {
        // [[1, 2, 3],
        //  [4, 5, 6]]
        // => [[1, 4],
        //     [2, 5],
        //     [3, 6]]
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let t2 = t.transpose();
        assert_eq!(t2.shape, vec![3, 2]);
        assert_eq!(t2.get(&[0, 0]), 1.0);
        assert_eq!(t2.get(&[0, 1]), 4.0);
        assert_eq!(t2.get(&[1, 0]), 2.0);
        assert_eq!(t2.get(&[1, 1]), 5.0);
        assert_eq!(t2.get(&[2, 0]), 3.0);
        assert_eq!(t2.get(&[2, 1]), 6.0);
    }

    // ==================== matmul ====================

    #[test]
    fn test_matmul() {
        // [1, 2]   [5, 6]   [1*5+2*7, 1*6+2*8]   [19, 22]
        // [3, 4] x [7, 8] = [3*5+4*7, 3*6+4*8] = [43, 50]
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let c = a.matmul(&b);
        assert_eq!(c.shape, vec![2, 2]);
        assert_eq!(c.data, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_matmul_non_square() {
        // [1, 2, 3]   [7,  8 ]   [1*7+2*9+3*11,  1*8+2*10+3*12]   [58,  64]
        // [4, 5, 6] x [9,  10] = [4*7+5*9+6*11,  4*8+5*10+6*12] = [139, 154]
        //              [11, 12]
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = Tensor::new(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]);
        let c = a.matmul(&b);
        assert_eq!(c.shape, vec![2, 2]);
        assert_eq!(c.data, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    #[should_panic]
    fn test_matmul_shape_mismatch() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]);
        a.matmul(&b);
    }
}
