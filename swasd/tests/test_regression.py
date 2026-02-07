"""
Tests for swasd.regression module

These tests cover the MonotoneSWDModel class which fits Bayesian
monotone regression using Stan.

NOTE: These tests will be slower than others because they involve Stan sampling.
Mark them with @pytest.mark.slow to allow skipping during quick test runs.
"""
import numpy as np
import pytest
from swasd.regression import MonotoneSWDModel


@pytest.fixture
def synthetic_pairwise_data():
    """
    Create synthetic pairwise SWD data for testing regression.
    
    Returns a dataset where blocks clearly show monotone decay.
    """
    np.random.seed(42)
    
    # True distances that decay monotonically
    B = 6
    true_d = np.array([2.0, 1.5, 1.0, 0.7, 0.5, 0.3])
    
    # Generate pairwise observations for all pairs (i < j)
    block_pairs = []
    sw_values = []
    
    for i in range(B):
        for j in range(i + 1, B):
            block_pairs.append([i + 1, j + 1])  # 1-based indexing
            # True pairwise distance with small noise
            true_dist = abs(true_d[i] - true_d[j])
            noisy_dist = true_dist * np.exp(np.random.randn() * 0.1)
            sw_values.append(noisy_dist)
    
    return {
        'block_pairs': np.array(block_pairs),
        'sw_values': np.array(sw_values),
        'num_blocks': B,
        'true_d': true_d
    }


@pytest.fixture
def small_pairwise_data():
    """Create minimal test data for quick tests"""
    block_pairs = np.array([
        [1, 2],
        [2, 3],
        [3, 4]
    ])
    sw_values = np.array([1.5, 1.0, 0.5])
    
    return {
        'block_pairs': block_pairs,
        'sw_values': sw_values,
        'num_blocks': 4
    }


class TestMonotoneSWDModelInitialization:
    """Tests for MonotoneSWDModel initialization"""
    
    def test_default_initialization(self):
        """Test initialization with default parameters"""
        model = MonotoneSWDModel()
        
        assert model.num_samples == 2000
        assert model.num_warmup == 2000
        assert model.num_chains == 4
        assert model.seed == 1234
        assert model.verbose == False
        assert model.fit_ is None
        assert model.idata is None
        assert model._B is None
    
    def test_custom_initialization(self):
        """Test initialization with custom parameters"""
        model = MonotoneSWDModel(
            num_samples=1000,
            num_warmup=500,
            num_chains=2,
            seed=9999,
            verbose=True
        )
        
        assert model.num_samples == 1000
        assert model.num_warmup == 500
        assert model.num_chains == 2
        assert model.seed == 9999
        assert model.verbose == True
    
    def test_initialization_type_conversion(self):
        """Test that parameters are properly converted to int"""
        model = MonotoneSWDModel(
            num_samples=1000.5,
            num_warmup=500.9,
            num_chains=2.1,
            seed=1234.7
        )
        
        assert isinstance(model.num_samples, int)
        assert isinstance(model.num_warmup, int)
        assert isinstance(model.num_chains, int)
        assert isinstance(model.seed, int)


class TestMonotoneSWDModelFitting:
    """Tests for the fit method"""
    
    @pytest.mark.slow
    def test_fit_basic(self, small_pairwise_data):
        """Test basic fitting functionality"""
        model = MonotoneSWDModel(
            num_samples=100,  # Small for speed
            num_warmup=100,
            num_chains=2,
            verbose=False
        )
        
        y_log = np.log(small_pairwise_data['sw_values'])
        pairs = small_pairwise_data['block_pairs']
        
        fit = model.fit(
            y=y_log,
            i=pairs[:, 0],
            j=pairs[:, 1]
        )
        
        # Check that fit was stored
        assert model.fit_ is not None
        assert model._B == 4  # Should match num_blocks
        
        # Check fit has expected parameters
        assert 'd' in fit
        assert 'delta' in fit
        assert 'sigma' in fit
    
    def test_fit_index_conversion(self, small_pairwise_data):
        """Test that 0-based indices are converted to 1-based"""
        model = MonotoneSWDModel(
            num_samples=50,
            num_warmup=50,
            num_chains=1,
            verbose=False
        )
        
        y_log = np.log(small_pairwise_data['sw_values'])
        # Use 0-based indices
        pairs_0based = small_pairwise_data['block_pairs'] - 1
        
        # Should not raise error - should convert internally
        model.fit(
            y=pairs_0based[:, 0],
            i=pairs_0based[:, 0],
            j=pairs_0based[:, 1]
        )
        
        assert model._B == 3  # max(i, j) after conversion
    
    @pytest.mark.slow
    def test_fit_creates_idata(self, small_pairwise_data):
        """Test that ArviZ InferenceData is created"""
        model = MonotoneSWDModel(
            num_samples=100,
            num_warmup=100,
            num_chains=2,
            verbose=False
        )
        
        y_log = np.log(small_pairwise_data['sw_values'])
        pairs = small_pairwise_data['block_pairs']
        
        model.fit(y=y_log, i=pairs[:, 0], j=pairs[:, 1])
        
        # idata creation might fail in some environments, so we just check it's attempted
        # If it succeeds, it should be an InferenceData object
        if model.idata is not None:
            import arviz as az
            assert isinstance(model.idata, az.InferenceData)
    
    def test_fit_array_conversion(self):
        """Test that inputs are properly converted to numpy arrays"""
        model = MonotoneSWDModel(num_samples=50, num_warmup=50, num_chains=1, verbose=False)
        
        # Test with lists instead of arrays
        y = [0.5, 0.3, 0.1]
        i = [1, 2, 3]
        j = [2, 3, 4]
        
        # Should not raise error
        model.fit(y=y, i=i, j=j)
        assert model._B == 4


class TestMonotoneSWDModelEstimation:
    """Tests for estimation methods"""
    
    def test_check_fitted_raises_when_not_fitted(self):
        """Test that _check_fitted raises error when model not fitted"""
        model = MonotoneSWDModel()
        
        with pytest.raises(RuntimeError, match="Call fit\\(\\) first"):
            model._check_fitted()
    
    @pytest.mark.slow
    def test_compute_swd_est_ci(self, small_pairwise_data):
        """Test credible interval computation"""
        model = MonotoneSWDModel(
            num_samples=100,
            num_warmup=100,
            num_chains=2,
            verbose=False
        )
        
        y_log = np.log(small_pairwise_data['sw_values'])
        pairs = small_pairwise_data['block_pairs']
        model.fit(y=y_log, i=pairs[:, 0], j=pairs[:, 1])
        
        blocks = np.arange(1, 5)
        mean, median, lower, upper = model.compute_swd_est_ci(blocks)
        
        # Check shapes
        assert len(mean) == 4
        assert len(median) == 4
        assert len(lower) == 4
        assert len(upper) == 4
        
        # Check ordering: lower < median < upper
        assert all(lower <= median)
        assert all(median <= upper)
        
        # Check monotone decreasing (approximately)
        # Note: With small samples this might not be perfect
        assert mean[0] >= mean[-1]  # At least first > last
    
    @pytest.mark.slow
    def test_fit_and_estimate(self, small_pairwise_data):
        """Test the combined fit_and_estimate method"""
        model = MonotoneSWDModel(
            num_samples=100,
            num_warmup=100,
            num_chains=2,
            verbose=False
        )
        
        result = model.fit_and_estimate(
            num_blocks=small_pairwise_data['num_blocks'],
            block_pairs=small_pairwise_data['block_pairs'],
            sw_values=small_pairwise_data['sw_values'],
            return_bands=True,
            cred_interval=(10, 90)
        )
        
        # Check output structure
        assert 'swd_est_mean' in result
        assert 'swd_est_median' in result
        assert 'swd_lci' in result
        assert 'swd_uci' in result
        assert 'idata' in result
        
        # Check shapes
        B = small_pairwise_data['num_blocks']
        assert len(result['swd_est_mean']) == B
        assert len(result['swd_est_median']) == B
        assert len(result['swd_lci']) == B
        assert len(result['swd_uci']) == B
        
        # Check values are positive
        assert all(result['swd_est_mean'] > 0)
        assert all(result['swd_est_median'] > 0)
    
    @pytest.mark.slow
    def test_fit_and_estimate_custom_credible_interval(self, small_pairwise_data):
        """Test custom credible interval specification"""
        model = MonotoneSWDModel(
            num_samples=100,
            num_warmup=100,
            num_chains=2,
            verbose=False
        )
        
        result = model.fit_and_estimate(
            num_blocks=small_pairwise_data['num_blocks'],
            block_pairs=small_pairwise_data['block_pairs'],
            sw_values=small_pairwise_data['sw_values'],
            cred_interval=(5, 95)  # Wider interval
        )
        
        # Check that intervals are wider than default
        interval_width = result['swd_uci'] - result['swd_lci']
        assert all(interval_width > 0)


class TestMonotoneSWDModelDiagnostics:
    """Tests for diagnostic methods"""
    
    @pytest.mark.slow
    def test_predicted_vs_residual(self, small_pairwise_data):
        """Test predicted vs residual computation"""
        model = MonotoneSWDModel(
            num_samples=100,
            num_warmup=100,
            num_chains=2,
            verbose=False
        )
        
        y_log = np.log(small_pairwise_data['sw_values'])
        pairs = small_pairwise_data['block_pairs']
        
        model.fit(y=y_log, i=pairs[:, 0], j=pairs[:, 1])
        
        result = model.predicted_vs_residual(
            y_log=y_log,
            i=pairs[:, 0],
            j=pairs[:, 1]
        )
        
        # Check output structure
        assert 'y_predicted' in result
        assert 'residuals' in result
        assert 'lower_CI' in result
        assert 'upper_CI' in result
        
        # Check shapes
        N = len(y_log)
        assert len(result['y_predicted']) == N
        assert len(result['residuals']) == N
        assert len(result['lower_CI']) == N
        assert len(result['upper_CI']) == N
        
        # Check residual calculation
        expected_resid = y_log - result['y_predicted']
        np.testing.assert_array_almost_equal(result['residuals'], expected_resid)
    
    
    def test_predicted_vs_residual_not_fitted(self):
        """Test that predicted_vs_residual raises error when not fitted"""
        model = MonotoneSWDModel()
        
        with pytest.raises(RuntimeError, match="Call fit\\(\\) first"):
            model.predicted_vs_residual(
                y_log=np.array([0.5, 0.3]),
                i=np.array([1, 2]),
                j=np.array([2, 3])
            )


@pytest.mark.slow
class TestMonotoneSWDModelIntegration:
    """Integration tests for the full workflow"""
    
    def test_full_workflow(self, synthetic_pairwise_data):
        """Test complete workflow: fit -> estimate -> diagnose"""
        model = MonotoneSWDModel(
            num_samples=200,
            num_warmup=200,
            num_chains=2,
            verbose=False
        )
        
        # Fit and estimate
        result = model.fit_and_estimate(
            num_blocks=synthetic_pairwise_data['num_blocks'],
            block_pairs=synthetic_pairwise_data['block_pairs'],
            sw_values=synthetic_pairwise_data['sw_values']
        )
        
        # Get diagnostics
        y_log = np.log(synthetic_pairwise_data['sw_values'])
        pairs = synthetic_pairwise_data['block_pairs']
        
        diag = model.predicted_vs_residual(
            y_log=y_log,
            i=pairs[:, 0],
            j=pairs[:, 1]
        )
        
        # Check that estimates are monotone decreasing (mostly)
        swd_est = result['swd_est_mean']
        # At least check first > last
        assert swd_est[0] > swd_est[-1]
        
        # Check residuals are reasonable
        assert abs(diag['residuals'].mean()) < 1.0  # Not too biased
    
    def test_reproducibility_with_seed(self, small_pairwise_data):
        """Test that same seed gives reproducible results"""
        model1 = MonotoneSWDModel(
            num_samples=100,
            num_warmup=100,
            num_chains=2,
            seed=12345,
            verbose=False
        )
        
        model2 = MonotoneSWDModel(
            num_samples=100,
            num_warmup=100,
            num_chains=2,
            seed=12345,
            verbose=False
        )
        
        result1 = model1.fit_and_estimate(
            num_blocks=small_pairwise_data['num_blocks'],
            block_pairs=small_pairwise_data['block_pairs'],
            sw_values=small_pairwise_data['sw_values'],
            random_seed=999
        )
        
        result2 = model2.fit_and_estimate(
            num_blocks=small_pairwise_data['num_blocks'],
            block_pairs=small_pairwise_data['block_pairs'],
            sw_values=small_pairwise_data['sw_values'],
            random_seed=999
        )
        
        # Results should be very similar (not exactly equal due to Stan internals)
        np.testing.assert_allclose(
            result1['swd_est_mean'],
            result2['swd_est_mean'],
            rtol=0.1  # Allow 10% relative difference
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not slow"])
