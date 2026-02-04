"""
Tests for swasd.utils module
"""
import numpy as np
import pytest
from swasd.utils import (
    detrend_samples,
    _flatten_samples,
    compute_scale_from_last_blocks,
)


class TestDetrendSamples:
    """Tests for detrend_samples function"""

    def test_detrend_constant(self):
        """Test that constant signal is unchanged (zero trend)"""
        samples = np.ones((100, 3)) * 5.0
        detrended = detrend_samples(samples)
        # Should be close to zero after removing mean
        assert detrended.shape == samples.shape
        # Mean should be removed
        assert np.allclose(detrended.mean(axis=0), 0, atol=1e-10)

    def test_detrend_linear(self):
        n = 100
        x = np.arange(n, dtype=float)
        samples = np.column_stack([2*x + 3, -x + 5, 0.5*x - 2])
    
        detrended = detrend_samples(samples)
    
        assert detrended.shape == samples.shape
    
        # Fit slope of residual vs time; should be ~0
        X = np.column_stack([x, np.ones(n)])
        slopes = np.linalg.lstsq(X, detrended, rcond=None)[0][0, :]  # first row = slope
        assert np.all(np.abs(slopes) < 1e-10)
    
        # Residual magnitude should be tiny (optional but intuitive)
        assert np.max(np.abs(detrended)) < 1e-8
        
    def test_detrend_shape_preservation(self):
        """Test that output shape matches input shape"""
        for shape in [(50, 1), (100, 5), (200, 10)]:
            samples = np.random.randn(*shape)
            detrended = detrend_samples(samples)
            assert detrended.shape == shape

    def test_detrend_random_noise(self):
        """Test detrending random noise (should be similar to mean-centering)"""
        np.random.seed(42)
        samples = np.random.randn(100, 3)
        detrended = detrend_samples(samples)
        # Mean should be near zero
        assert np.allclose(detrended.mean(axis=0), 0, atol=0.1)


class TestFlattenSamples:
    """Tests for _flatten_samples function"""

    def test_flatten_2d_no_warmup(self):
        """Test flattening 2D array with no warmup"""
        samples = np.random.randn(100, 5)
        result = _flatten_samples(samples, num_warmup=0)
        assert result.shape == (100, 5)
        np.testing.assert_array_equal(result, samples)

    def test_flatten_2d_with_warmup(self):
        """Test flattening 2D array with warmup removal"""
        samples = np.random.randn(100, 5)
        result = _flatten_samples(samples, num_warmup=20)
        assert result.shape == (80, 5)
        np.testing.assert_array_equal(result, samples[20:])

    def test_flatten_3d_no_warmup(self):
        """Test flattening 3D array (multiple chains) with no warmup"""
        samples = np.random.randn(4, 100, 5)  # 4 chains, 100 draws, 5 params
        result = _flatten_samples(samples, num_warmup=0)
        assert result.shape == (400, 5)  # 4 * 100 = 400

    def test_flatten_3d_with_warmup(self):
        """Test flattening 3D array with warmup removal per chain"""
        samples = np.random.randn(4, 100, 5)
        result = _flatten_samples(samples, num_warmup=20)
        # Each chain: 100 - 20 = 80 draws, total = 4 * 80 = 320
        assert result.shape == (320, 5)

    def test_flatten_excessive_warmup(self):
        """Test that excessive warmup returns empty array"""
        samples = np.random.randn(100, 5)
        result = _flatten_samples(samples, num_warmup=150)
        assert result.shape == (0, 5)

    def test_flatten_3d_excessive_warmup_per_chain(self):
        """Test excessive warmup with 3D samples"""
        samples = np.random.randn(4, 50, 5)
        result = _flatten_samples(samples, num_warmup=60)
        assert result.shape == (0, 5)

    def test_flatten_invalid_dimensions(self):
        """Test that invalid dimensions raise error"""
        samples_1d = np.random.randn(100)
        with pytest.raises(ValueError, match="must be a 2D.*or 3D"):
            _flatten_samples(samples_1d)

    def test_flatten_preserves_dtype(self):
        """Test that dtype is preserved"""
        samples = np.random.randn(100, 5).astype(np.float32)
        result = _flatten_samples(samples)
        assert result.dtype == np.float32


class TestComputeScaleFromLastBlocks:
    """Tests for compute_scale_from_last_blocks function"""

    def test_scale_simple_2d(self):
        """Test scale computation with simple 2D array"""
        np.random.seed(42)
        # Create samples with known std dev
        samples = np.random.randn(1000, 3) * np.array([1.0, 2.0, 0.5])
        scale = compute_scale_from_last_blocks(samples, num_blocks=6)
        
        assert scale.shape == (3,)
        assert all(scale > 0)
        # Should be roughly close to the true scales
        assert scale[1] > scale[0]  # Second param has larger scale
        assert scale[2] < scale[0]  # Third param has smaller scale

    def test_scale_with_warmup_2d(self):
        """Test scale computation with warmup removal (2D)"""
        np.random.seed(42)
        samples = np.random.randn(1000, 3)
        scale_no_warmup = compute_scale_from_last_blocks(samples, num_warmup=0)
        scale_with_warmup = compute_scale_from_last_blocks(samples, num_warmup=100)
        
        assert scale_no_warmup.shape == scale_with_warmup.shape
        # Scales might differ due to different data
        assert all(scale_with_warmup > 0)

    def test_scale_3d_samples(self):
        """Test scale computation with 3D samples (multiple chains)"""
        np.random.seed(42)
        samples = np.random.randn(4, 250, 3)  # 4 chains, 250 draws, 3 params
        scale = compute_scale_from_last_blocks(samples, num_blocks=6, num_warmup=50)
        
        assert scale.shape == (3,)
        assert all(scale > 0)

    def test_scale_zero_variance_protection(self):
        """Test that zero variance is replaced with 1.0"""
        samples = np.ones((1000, 3)) * np.array([1.0, 2.0, 3.0])
        scale = compute_scale_from_last_blocks(samples, num_blocks=6)
        
        # All constant, so std should be 0, replaced with 1.0
        np.testing.assert_array_equal(scale, np.ones(3))

    def test_scale_custom_detrend(self):
        """Test with custom detrend function"""
        def custom_detrend(arr):
            # Just subtract mean
            return arr - arr.mean(axis=0, keepdims=True)
        
        samples = np.random.randn(1000, 3)
        scale = compute_scale_from_last_blocks(
            samples, num_blocks=6, detrend_fn=custom_detrend
        )
        assert scale.shape == (3,)
        assert all(scale > 0)

    def test_scale_empty_after_warmup(self):
        """Test behavior when warmup removes all samples"""
        samples = np.random.randn(100, 3)
        scale = compute_scale_from_last_blocks(samples, num_warmup=200)
        # Should return unit scales
        np.testing.assert_array_equal(scale, np.ones(3))

    def test_scale_single_element_per_block(self):
        """Test with very few samples"""
        samples = np.random.randn(12, 3)  # 6 blocks of 2 elements each
        scale = compute_scale_from_last_blocks(samples, num_blocks=6)
        assert scale.shape == (3,)
        assert all(scale > 0)

    def test_scale_last_two_blocks_only(self):
        """Verify that only last two blocks are used"""
        np.random.seed(42)
        # First 4 blocks: small variance, last 2 blocks: large variance
        small_var = np.random.randn(400, 3) * 0.1
        large_var = np.random.randn(200, 3) * 10.0
        samples = np.vstack([small_var, large_var])
        
        scale = compute_scale_from_last_blocks(samples, num_blocks=6)
        # Scale should reflect the large variance from last blocks
        assert all(scale > 1.0)  # Should be much larger than small variance


class TestLoadStanText:
    """Tests for load_stan_text function"""
    
    def test_load_existing_model(self):
        """Test loading the monotone_swd.stan model"""
        from swasd.utils import load_stan_text
        
        stan_code = load_stan_text("monotone_swd.stan")
        assert isinstance(stan_code, str)
        assert len(stan_code) > 0
        assert "data {" in stan_code
        assert "model {" in stan_code

    def test_load_nonexistent_model(self):
        """Test that loading non-existent file raises error"""
        from swasd.utils import load_stan_text
        
        with pytest.raises(FileNotFoundError):
            load_stan_text("nonexistent.stan")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
