use crate::tensor::Tensor;

/// Generate sinusoidal positional encoding.
/// PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
/// PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
/// Returns Tensor [seq_len, d_model]
pub fn positional_encoding(seq_len: usize, d_model: usize) -> Tensor {
    let mut data = vec![0.0_f32; seq_len * d_model];
    for pos in 0..seq_len {
        for i in 0..d_model {
            let angle = pos as f32 / (10000.0_f32).powf((i / 2 * 2) as f32 / d_model as f32);
            data[pos * d_model + i] = if i % 2 == 0 {
                angle.sin()
            } else {
                angle.cos()
            };
        }
    }
    Tensor::new(data, vec![seq_len, d_model])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape() {
        let pe = positional_encoding(10, 16);
        assert_eq!(pe.shape, vec![10, 16]);
    }

    #[test]
    fn test_position_zero() {
        // At pos=0: sin(0)=0 for even dims, cos(0)=1 for odd dims
        let pe = positional_encoding(1, 4);
        assert!((pe.get(&[0, 0]) - 0.0).abs() < 1e-5); // sin(0)
        assert!((pe.get(&[0, 1]) - 1.0).abs() < 1e-5); // cos(0)
        assert!((pe.get(&[0, 2]) - 0.0).abs() < 1e-5); // sin(0)
        assert!((pe.get(&[0, 3]) - 1.0).abs() < 1e-5); // cos(0)
    }

    #[test]
    fn test_values_in_range() {
        // sin and cos are in [-1, 1]
        let pe = positional_encoding(100, 32);
        for &v in &pe.data {
            assert!(v >= -1.0 && v <= 1.0, "value {} out of range", v);
        }
    }

    #[test]
    fn test_different_positions_differ() {
        let pe = positional_encoding(3, 4);
        // pos 0 and pos 1 should be different
        let row0 = &pe.data[0..4];
        let row1 = &pe.data[4..8];
        assert!(row0 != row1);
    }

    #[test]
    fn test_first_dim_is_sin() {
        // dim 0 (even): sin(pos / 10000^(0/d_model)) = sin(pos)
        let pe = positional_encoding(3, 4);
        let pos1_dim0 = pe.get(&[1, 0]);
        assert!((pos1_dim0 - (1.0_f32).sin()).abs() < 1e-5);
    }

    #[test]
    fn test_deterministic() {
        let pe1 = positional_encoding(5, 8);
        let pe2 = positional_encoding(5, 8);
        assert_eq!(pe1.data, pe2.data);
    }
}
