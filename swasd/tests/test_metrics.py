"""
Tests for swasd.metrics module
"""
import numpy as np
import pytest
from swasd.metrics import MetricComputer


class TestMetricComputer:
    """Tests for MetricComputer class"""

    @pytest.fixture
    def metric_computer(self):
        """Fixture for a basic MetricComputer instance"""
        return MetricComputer(n_projections=50, n_bootstrap=1)

    @pytest.fixture
    def metric_computer_bootstrap(self):
        """Fixture for MetricComputer with bootstrap"""
        return MetricComputer(n_projections=50, n_bootstrap=10)

    def test_initialization(self):
        """Test MetricComputer initialization"""
        mc = MetricComputer(n_projections=100, n_bootstrap=5)
        assert mc.n_projections == 100
        assert mc.n_bootstrap == 5
        assert "swd" in mc._metrics
        assert "max_swd" in mc._metrics

    def test_compute_value_identical_samples(self, metric_computer):
        """Test SWD between identical samples is near zero"""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        Y = X.copy()
        
        dist = metric_computer.compute_value("swd", X, Y, seed=1)
        assert isinstance(dist, float)
        assert dist < 1e-10  # Should be essentially zero

    def test_compute_value_different_samples(self, metric_computer):
        """Test SWD between different samples is positive"""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        Y = np.random.randn(100, 5) + 2.0  # Shifted distribution
        
        dist = metric_computer.compute_value("swd", X, Y, seed=1)
        assert dist > 0

    def test_compute_value_scaled_samples(self, metric_computer):
        """Test SWD with scaled distributions"""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        Y = X * 2.0  # Scaled distribution
        
        dist = metric_computer.compute_value("swd", X, Y, seed=1)
        assert dist > 0

    def test_compute_value_max_swd(self, metric_computer):
        """Test max_swd metric"""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        Y = np.random.randn(100, 5) + 1.0
        
        dist = metric_computer.compute_value("max_swd", X, Y, seed=1)
        assert dist > 0
        assert isinstance(dist, float)

    def test_compute_value_unsupported_metric(self, metric_computer):
        """Test that unsupported metric raises error"""
        X = np.random.randn(100, 5)
        Y = np.random.randn(100, 5)
        
        with pytest.raises(ValueError, match="Unsupported metric"):
            metric_computer.compute_value("invalid_metric", X, Y)

    def test_bootstrap_no_bootstrap(self, metric_computer):
        """Test bootstrap with n_bootstrap=1 (no actual bootstrap)"""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        Y = np.random.randn(100, 5) + 1.0
        
        result = metric_computer.bootstrap("swd", X, Y, seed=1)
        
        assert isinstance(result, dict)
        assert "est" in result
        assert "mean" in result
        assert "std" in result
        assert "ci" in result
        assert result["std"] == 0.0  # No bootstrap, so std is 0
        assert result["est"] == result["mean"]

    def test_bootstrap_with_replicates(self, metric_computer_bootstrap):
        """Test bootstrap with multiple replicates"""
        np.random.seed(42)
        X = np.random.randn(200, 5)
        Y = np.random.randn(200, 5) + 1.0
        
        result = metric_computer_bootstrap.bootstrap("swd", X, Y, seed=1)
        
        assert isinstance(result, dict)
        assert "est" in result
        assert "mean" in result
        assert "std" in result
        assert "ci" in result
        
        # With bootstrap, std should be positive
        assert result["std"] > 0
        # CI should be a tuple
        assert len(result["ci"]) == 2
        assert result["ci"][0] < result["ci"][1]

    def test_bootstrap_reproducibility(self, metric_computer_bootstrap):
        """Test that bootstrap is reproducible with same seed"""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        Y = np.random.randn(100, 5) + 1.0
        
        result1 = metric_computer_bootstrap.bootstrap("swd", X, Y, seed=123)
        result2 = metric_computer_bootstrap.bootstrap("swd", X, Y, seed=123)
        
        assert result1["est"] == result2["est"]
        assert result1["mean"] == result2["mean"]
        np.testing.assert_allclose(result1["ci"], result2["ci"])

    def test_compute_blockwise_adjacent_mode(self, metric_computer):
        """Test blockwise computation with adjacent mode"""
        np.random.seed(42)
        # Create samples with gradual shift to test block differences
        samples = np.vstack([
            np.random.randn(100, 3) + i * 0.1 
            for i in range(6)
        ])
        
        result = metric_computer.compute_blockwise(
            samples, metric="swd", num_blocks=6, mode="adjacent", verbose=False
        )
        
        assert "est_all" in result
        assert "block_pair" in result
        assert len(result["est_all"]) == 5  # 6 blocks -> 5 adjacent pairs
        assert len(result["block_pair"]) == 5
        
        # Check pairs are correct
        expected_pairs = [(i, i+1) for i in range(1, 6)]
        assert result["block_pair"] == expected_pairs

    def test_compute_blockwise_all_pairs_mode(self, metric_computer):
        """Test blockwise computation with all_pairs mode"""
        np.random.seed(42)
        samples = np.random.randn(600, 3)
        
        result = metric_computer.compute_blockwise(
            samples, metric="swd", num_blocks=6, mode="all_pairs", verbose=False
        )
        
        # 6 blocks -> C(6,2) = 15 pairs
        assert len(result["est_all"]) == 15
        assert len(result["block_pair"]) == 15
        
        # Check all pairs are unique
        pairs = result["block_pair"]
        assert len(set(pairs)) == 15

    def test_compute_blockwise_true_mode(self, metric_computer):
        """Test blockwise computation with true mode (reference samples)"""
        np.random.seed(42)
        samples = np.random.randn(600, 3)
        ref_samples = np.random.randn(1000, 3)
        
        result = metric_computer.compute_blockwise(
            samples, 
            metric="swd", 
            num_blocks=6, 
            mode="true", 
            ref_samples=ref_samples,
            verbose=False
        )
        
        # 6 blocks compared to stationary samples
        assert len(result["est_all"]) == 6
        assert len(result["block_pair"]) == 6
        
        # Check pairs format: (block_idx, "S")
        for pair in result["block_pair"]:
            assert pair[1] == "S"

    def test_compute_blockwise_true_mode_no_ref_samples(self, metric_computer):
        """Test that true mode without ref_samples raises error"""
        samples = np.random.randn(600, 3)
        
        with pytest.raises(ValueError, match="Reference stationary samples required"):
            metric_computer.compute_blockwise(
                samples, mode="true", ref_samples=None
            )

    def test_compute_blockwise_3d_samples_raises_error(self, metric_computer):
        """Test that 3D samples raise error"""
        samples = np.random.randn(4, 150, 3)  # 3D array
        
        with pytest.raises(ValueError, match="must be 2D"):
            metric_computer.compute_blockwise(samples)

    def test_compute_blockwise_scaling(self, metric_computer):
        """Test that samples are properly scaled"""
        np.random.seed(42)
        # Create samples with different scales
        samples = np.random.randn(600, 3) * np.array([1.0, 10.0, 100.0])
        
        result = metric_computer.compute_blockwise(
            samples, num_blocks=6, mode="adjacent", verbose=False
        )
        
        # Should compute without error
        assert len(result["est_all"]) == 5
        # Distances should be reasonable (not huge due to large scales)
        assert all(0 <= d < 100 for d in result["est_all"])

    def test_compute_blockwise_with_bootstrap(self, metric_computer_bootstrap):
        """Test blockwise computation with bootstrap"""
        np.random.seed(42)
        samples = np.random.randn(600, 3)
        
        result = metric_computer_bootstrap.compute_blockwise(
            samples, num_blocks=6, mode="adjacent", verbose=False
        )
        
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert len(result["ci_lower"]) == 5
        assert len(result["ci_upper"]) == 5
        
        # Lower CI should be less than upper CI
        for low, high in zip(result["ci_lower"], result["ci_upper"]):
            assert low <= high

    def test_compute_blockwise_small_samples(self, metric_computer):
        """Test with very small samples"""
        samples = np.random.randn(12, 2)  # Only 12 samples, 6 blocks = 2 per block
        
        result = metric_computer.compute_blockwise(
            samples, num_blocks=6, mode="adjacent", verbose=False
        )
        
        # Should still work but may have high variance
        assert len(result["est_all"]) == 5

    def test_different_projections(self):
        """Test that more projections give more stable estimates"""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        Y = np.random.randn(100, 5) + 1.0
        
        mc_10 = MetricComputer(n_projections=10, n_bootstrap=1)
        mc_100 = MetricComputer(n_projections=100, n_bootstrap=1)
        
        dist_10 = mc_10.compute_value("swd", X, Y, seed=1)
        dist_100 = mc_100.compute_value("swd", X, Y, seed=1)
        
        # Both should be positive
        assert dist_10 > 0
        assert dist_100 > 0
        # With same seed, should be somewhat consistent
        assert abs(dist_10 - dist_100) / dist_10 < 0.5  # Within 50%


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
