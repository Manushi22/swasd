import numpy as np
import pytest
from unittest.mock import Mock, MagicMock, patch, call
from collections import defaultdict
from io import StringIO
import sys

# Import the functions to test
from swasd.algorithm import (
    swasd,
    convergence_check,
    _slice_samples,
    _print_summary,
    _geometric_checkpoints,
    _linear_checkpoints,
)


# ==============================================================================
# Test Helper Functions
# ==============================================================================

class TestSliceSamples:
    """Test _slice_samples helper function."""
    
    def test_slice_2d_samples(self):
        """Test slicing 2D samples (T, d)."""
        samples = np.random.randn(1000, 5)
        result = _slice_samples(samples, 500)
        assert result.shape == (500, 5)
        np.testing.assert_array_equal(result, samples[:500, :])
    
    def test_slice_3d_samples(self):
        """Test slicing 3D samples (C, T, d)."""
        samples = np.random.randn(4, 1000, 5)
        result = _slice_samples(samples, 500)
        assert result.shape == (4, 500, 5)
        np.testing.assert_array_equal(result, samples[:, :500, :])
    
    def test_slice_at_boundary(self):
        """Test slicing at exact sample size."""
        samples = np.random.randn(100, 3)
        result = _slice_samples(samples, 100)
        assert result.shape == (100, 3)
        np.testing.assert_array_equal(result, samples)
    
    def test_slice_first_iteration(self):
        """Test slicing to get just first iteration."""
        samples = np.random.randn(1000, 5)
        result = _slice_samples(samples, 1)
        assert result.shape == (1, 5)


class TestGeometricCheckpoints:
    """Test _geometric_checkpoints helper."""
    
    def test_basic_sequence(self):
        """Test basic geometric checkpoint generation."""
        ckpts = _geometric_checkpoints(1000, 100, 1.5)
        assert ckpts[0] == 100
        assert ckpts[-1] == 1000
        assert len(ckpts) > 1
        assert np.all(np.diff(ckpts) > 0)  # Increasing
    
    def test_rate_validation(self):
        """Test that rate <= 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="check_rate must be > 1.0"):
            _geometric_checkpoints(1000, 100, 1.0)
        
        with pytest.raises(ValueError, match="check_rate must be > 1.0"):
            _geometric_checkpoints(1000, 100, 0.5)
    
    def test_ends_at_n_iters(self):
        """Test that sequence always ends at n_iters."""
        ckpts = _geometric_checkpoints(999, 100, 1.2)
        assert ckpts[-1] == 999
    
    def test_no_duplicates(self):
        """Test no duplicate checkpoints."""
        ckpts = _geometric_checkpoints(1000, 100, 1.1)
        assert len(ckpts) == len(np.unique(ckpts))
    
    def test_small_n_iters(self):
        """Test with small iteration count."""
        ckpts = _geometric_checkpoints(50, 10, 1.5)
        assert ckpts[0] == 10
        assert ckpts[-1] == 50


class TestLinearCheckpoints:
    """Test _linear_checkpoints helper."""
    
    def test_basic_range(self):
        """Test basic linear checkpoint generation."""
        ckpts = _linear_checkpoints(1000, 100, 50)
        assert ckpts[0] == 100
        assert ckpts[-1] == 1000
        assert np.all(np.diff(ckpts) <= 50)
    
    def test_with_explicit_stop(self):
        """Test with explicit stop value."""
        ckpts = _linear_checkpoints(1000, 100, 50, stop=500)
        assert ckpts[0] == 100
        assert ckpts[-1] == 500
        assert np.max(ckpts) <= 500
    
    def test_stop_clamping(self):
        """Test that stop is clamped to n_iters."""
        ckpts = _linear_checkpoints(1000, 100, 50, stop=2000)
        assert ckpts[-1] == 1000  # Clamped to n_iters
    
    def test_step_validation(self):
        """Test that step < 1 raises ValueError."""
        with pytest.raises(ValueError, match="step must be >= 1"):
            _linear_checkpoints(1000, 100, 0)
    
    def test_start_greater_than_stop(self):
        """Test empty array when start > stop."""
        ckpts = _linear_checkpoints(1000, 500, 50, stop=300)
        assert len(ckpts) == 0
    
    def test_no_duplicates(self):
        """Test no duplicate checkpoints."""
        ckpts = _linear_checkpoints(1000, 100, 10)
        assert len(ckpts) == len(np.unique(ckpts))
    
    def test_start_equals_stop(self):
        """Test edge case where start equals stop."""
        ckpts = _linear_checkpoints(1000, 500, 50, stop=500)
        assert len(ckpts) == 1
        assert ckpts[0] == 500
    
    def test_large_step(self):
        """Test step larger than range."""
        ckpts = _linear_checkpoints(1000, 100, 2000, stop=500)
        assert ckpts[0] == 100
        assert ckpts[-1] == 500
    
    def test_returns_int_dtype(self):
        """Test that output is integer array."""
        ckpts = _linear_checkpoints(1000, 100, 50)
        assert ckpts.dtype in [np.int32, np.int64, np.int_]


class TestPrintSummary:
    """Test _print_summary helper."""
    
    def test_rhat_convergence_summary(self, capsys):
        """Test summary for Rhat convergence."""
        history = {
            "rhat_best": [3.1, 3.0, 2.9, 1.05],
            "swd_to_stationary": [],
        }
        _print_summary(history, k_conv=1000, reason="rhat", 
                      rhat_threshold=1.1, swd_threshold=1.0)
        
        captured = capsys.readouterr()
        assert "CONVERGENCE DETECTED" in captured.out
        assert "iteration: 1000" in captured.out
        assert "RHAT" in captured.out
        assert "1.05" in captured.out
    
    def test_swd_convergence_summary(self, capsys):
        """Test summary for SWD convergence."""
        history = {
            "rhat_best": [],
            "swd_to_stationary": [5.0, 3.0, 1.5, 0.8],
        }
        _print_summary(history, k_conv=1500, reason="swd",
                      rhat_threshold=1.1, swd_threshold=1.0)
        
        captured = capsys.readouterr()
        assert "CONVERGENCE DETECTED" in captured.out
        assert "iteration: 1500" in captured.out
        assert "SWD" in captured.out
        assert "0.8" in captured.out
    
    def test_combined_metrics_summary(self, capsys):
        """Test summary when both Rhat and SWD checked."""
        history = {
            "rhat_best": [3.0, 2.5, 2.0],
            "swd_to_stationary": [5.0, 3.0, 1.5, 0.9],
        }
        _print_summary(history, k_conv=2000, reason="swd",
                      rhat_threshold=1.1, swd_threshold=1.0)
        
        captured = capsys.readouterr()
        assert "Rhat checks performed: 3" in captured.out
        assert "SWD checks performed: 4" in captured.out
        assert "Best Rhat achieved: 2.0" in captured.out
        assert "Initial SWD: 5.0" in captured.out


# ==============================================================================
# Test convergence_check Function
# ==============================================================================

class TestConvergenceCheck:
    """Test convergence_check function."""
    
    @patch('swasd.algorithm.rhat_check')
    def test_rhat_success_early_return(self, mock_rhat):
        """Test that Rhat success triggers early return."""
        mock_rhat.return_value = (True, 500, 1.02)
        
        samples = np.random.randn(4, 1000, 5)
        xk = _slice_samples(samples, 600)
        history = defaultdict(list)
        metric_comp = Mock()
        
        k_conv, reason = convergence_check(
            xk,
            do_rhat_check=True,
            do_swd_check=True,
            history=history,
            metric_comp=metric_comp,
            rhat_threshold=1.1,
            rhat_wmin=100,
            rhat_method="rank",
            rhat_num_windows=5,
            verbose=0,
        )
        
        assert k_conv == 600
        assert reason == "rhat"
        assert history["rhat_best"][-1] == 1.02
        # SWD should NOT have been computed
        metric_comp.compute_blockwise.assert_not_called()
    
    @patch('swasd.algorithm.rhat_check')
    @patch('swasd.algorithm.MonotoneSWDModel')
    def test_rhat_fail_then_swd_success(self, mock_model_class, mock_rhat):
        """Test SWD check when Rhat fails."""
        mock_rhat.return_value = (False, 500, 2.5)
        
        # Mock SWD computation
        mock_metric = Mock()
        mock_metric.compute_blockwise.return_value = {
            "block_pair": [(2, 3), (3, 4), (4, 5)],
            "est_all": [2.0, 1.5, 1.0],
        }
        
        # Mock regression model
        mock_model = Mock()
        mock_model.fit_and_estimate.return_value = {
            "swd_est_mean": np.array([3.0, 2.0, 1.5, 1.0, 0.5, 0.3]),
        }
        mock_model.predicted_vs_residual.return_value = {
            "y_predicted": np.array([1.0, 1.0, 1.0]),
            "residuals": np.array([0.1, -0.1, 0.0]),
        }
        mock_model_class.return_value = mock_model
        
        samples = np.random.randn(2000, 5)
        xk = _slice_samples(samples, 1500)
        history = defaultdict(list)
        
        k_conv, reason = convergence_check(
            xk,
            do_rhat_check=True,
            do_swd_check=True,
            history=history,
            metric_comp=mock_metric,
            n_blocks=6,
            convg_threshold=0.5,
            rhat_threshold=1.1,
            rhat_wmin=100,
            rhat_method="rank",
            rhat_num_windows=5,
            verbose=0,
        )
        
        # Rhat was checked
        assert len(history["rhat_best"]) == 1
        assert history["rhat_best"][0] == 2.5
        
        # SWD converged
        assert k_conv == 1500
        assert reason == "swd"
        assert history["swd_to_stationary"][-1] == 0.3
    
    @patch('swasd.algorithm.MonotoneSWDModel')
    def test_swd_only_no_convergence(self, mock_model_class):
        """Test SWD check without convergence."""
        mock_metric = Mock()
        mock_metric.compute_blockwise.return_value = {
            "block_pair": [(2, 3), (3, 4)],
            "est_all": [2.0, 1.5],
        }
        
        mock_model = Mock()
        mock_model.fit_and_estimate.return_value = {
            "swd_est_mean": np.array([5.0, 4.0, 3.0, 2.5, 2.0, 1.8]),
        }
        mock_model.predicted_vs_residual.return_value = {
            "y_predicted": np.array([1.0, 1.0]),
            "residuals": np.array([0.1, -0.1]),
        }
        mock_model_class.return_value = mock_model
        
        samples = np.random.randn(1000, 3)
        history = defaultdict(list)
        
        k_conv, reason = convergence_check(
            samples,
            do_rhat_check=False,
            do_swd_check=True,
            history=history,
            metric_comp=mock_metric,
            n_blocks=6,
            convg_threshold=1.0,
            verbose=0,
        )
        
        # No convergence
        assert k_conv is None
        assert reason is None
        # But SWD was computed
        assert len(history["swd_to_stationary"]) == 1
        assert history["swd_to_stationary"][0] == 1.8
    
    def test_skip_both_checks(self):
        """Test when neither check is scheduled."""
        samples = np.random.randn(1000, 3)
        xk = _slice_samples(samples, 500)
        history = defaultdict(list)
        metric_comp = Mock()
        
        k_conv, reason = convergence_check(
            xk,
            do_rhat_check=False,
            do_swd_check=False,
            history=history,
            metric_comp=metric_comp,
            verbose=0,
        )
        
        assert k_conv is None
        assert reason is None
        assert len(history["rhat_best"]) == 0
        assert len(history["swd_to_stationary"]) == 0


# ==============================================================================
# Test Main swasd Function
# ==============================================================================

class TestSWASDValidation:
    """Test input validation in swasd function."""
    
    def test_n_blocks_too_small(self):
        """Test n_blocks < 2 raises ValueError."""
        samples = np.random.randn(1000, 3)
        with pytest.raises(ValueError, match="n_blocks must be >= 2"):
            swasd(samples, n_blocks=1)
    
    def test_min_iters_per_block_invalid(self):
        """Test min_iters_per_block < 1 raises ValueError."""
        samples = np.random.randn(1000, 3)
        with pytest.raises(ValueError, match="min_iters_per_block must be >= 1"):
            swasd(samples, min_iters_per_block=0)
    
    def test_check_rate_invalid(self):
        """Test check_rate <= 1.0 raises ValueError."""
        samples = np.random.randn(1000, 3)
        with pytest.raises(ValueError, match="check_rate should be > 1.0"):
            swasd(samples, check_rate=1.0)
    
    def test_invalid_block_mode(self):
        """Test invalid block_mode raises ValueError."""
        samples = np.random.randn(1000, 3)
        with pytest.raises(ValueError, match="block_mode must be"):
            swasd(samples, block_mode="invalid")
    
    def test_true_swd_without_samples(self):
        """Test true_swd=True without true_samples raises ValueError."""
        samples = np.random.randn(1000, 3)
        with pytest.raises(ValueError, match="true_swd=True requires"):
            swasd(samples, true_swd=True)
    
    def test_n_iters_exceeds_available_2d(self):
        """Test n_iters > available samples raises ValueError."""
        samples = np.random.randn(1000, 3)
        with pytest.raises(ValueError, match="n_iters exceeds"):
            swasd(samples, n_iters=2000)
    
    def test_n_iters_exceeds_available_3d(self):
        """Test n_iters > available samples for 3D."""
        samples = np.random.randn(4, 1000, 3)
        with pytest.raises(ValueError, match="n_iters exceeds"):
            swasd(samples, n_iters=2000)
    
    def test_insufficient_iterations(self, capsys):
        """Test behavior when k0 > n_iters."""
        samples = np.random.randn(100, 3)
        result = swasd(samples, n_blocks=6, min_iters_per_block=50, verbose=1)
        
        assert result["k_conv"] is None
        captured = capsys.readouterr()
        assert "Not enough iterations" in captured.out


class TestSWASDBasicFunctionality:
    """Test basic swasd functionality."""
    
    @patch('swasd.algorithm.convergence_check')
    def test_early_convergence(self, mock_check):
        """Test early convergence detection."""
        # Mock needs to update history dict just like real function
        def mock_convergence(*args, **kwargs):
            history = kwargs['history']
            history['k_conv'] = 600
            history['convergence_reason'] = 'rhat'
            return (600, "rhat")
        
        mock_check.side_effect = mock_convergence
        
        samples = np.random.randn(1000, 3)
        result = swasd(samples, n_blocks=4, min_iters_per_block=50, verbose=0)
        
        assert result["k_conv"] == 600
        assert result["convergence_reason"] == "rhat"
    
    @patch('swasd.algorithm.convergence_check')
    def test_no_convergence(self, mock_check, capsys):
        """Test when convergence is never detected."""
        # Mock needs to update history with SWD data
        call_count = [0]
        
        def mock_convergence(*args, **kwargs):
            history = kwargs['history']
            call_count[0] += 1
            # Simulate SWD checks
            if kwargs.get('do_swd_check'):
                history['swd_to_stationary'].append(2.0)
                xk = args[0]
                k = xk.shape[0] if xk.ndim == 2 else xk.shape[1]
                history['convg_check_iterate'].append(k)
            return (None, None)
        
        mock_check.side_effect = mock_convergence
        
        samples = np.random.randn(1000, 3)
        result = swasd(samples, n_blocks=4, min_iters_per_block=50, verbose=1)
        
        assert result["k_conv"] is None
        captured = capsys.readouterr()
        assert "CONVERGENCE NOT DETECTED" in captured.out
    
    @patch('swasd.algorithm.convergence_check')
    def test_verbose_levels(self, mock_check):
        """Test different verbose levels."""
        mock_check.return_value = (None, None)
        samples = np.random.randn(500, 3)
        
        # Silent
        result = swasd(samples, n_blocks=4, min_iters_per_block=50, verbose=0)
        assert result is not None
        
        # Reset mock for next test
        mock_check.reset_mock()
        
        # Summary
        result = swasd(samples, n_blocks=4, min_iters_per_block=50, verbose=1)
        assert result is not None
        
        # Reset mock for next test
        mock_check.reset_mock()
        
        # Progress bar (harder to test)
        result = swasd(samples, n_blocks=4, min_iters_per_block=50, verbose=2)
        assert result is not None
    
    def test_verbose_bool_conversion(self):
        """Test verbose bool to int conversion."""
        samples = np.random.randn(100, 3)
        
        with patch('swasd.algorithm.convergence_check') as mock_check:
            mock_check.return_value = (None, None)
            
            # verbose=True should be treated as 1
            result = swasd(samples, n_blocks=4, min_iters_per_block=10, verbose=True)
            assert result is not None
            
            # verbose=False should be treated as 0
            result = swasd(samples, n_blocks=4, min_iters_per_block=10, verbose=False)
            assert result is not None


class TestSWASDCheckpointLogic:
    """Test checkpoint creation and merging."""
    
    @patch('swasd.algorithm.convergence_check')
    def test_checkpoint_merge(self, mock_check):
        """Test that rhat and swasd checkpoints are properly merged."""
        # Track which types of checks were called
        rhat_checks = []
        swd_checks = []
        
        def mock_convergence(*args, **kwargs):
            xk = args[0]
            k = xk.shape[0] if xk.ndim == 2 else xk.shape[1]
            if kwargs.get('do_rhat_check'):
                rhat_checks.append(k)
            if kwargs.get('do_swd_check'):
                swd_checks.append(k)
            return (None, None)
        
        mock_check.side_effect = mock_convergence
        
        samples = np.random.randn(2000, 3)
        result = swasd(
            samples, 
            n_blocks=6, 
            min_iters_per_block=100,
            rhat_wmin=100,
            rhat_check_iter=50,
            rhat_stop_factor=1.5,
            check_rate=1.5,
            verbose=0
        )
        
        # Should have called convergence_check multiple times
        assert mock_check.call_count > 0
        
        # Check that some calls had do_rhat_check=True
        assert len(rhat_checks) > 0, "No Rhat checks were performed"
        
        # Check that some calls had do_swd_check=True
        assert len(swd_checks) > 0, "No SWD checks were performed"


class TestSWASDIntegration:
    """Integration tests with minimal mocking."""
    
    @patch('swasd.algorithm.MonotoneSWDModel')
    @patch('swasd.algorithm.rhat_check')
    def test_full_pipeline_2d(self, mock_rhat, mock_model_class):
        """Test full pipeline with 2D samples."""
        # Rhat never converges
        mock_rhat.return_value = (False, 500, 2.0)
        
        # SWD converges eventually
        mock_model = Mock()
        call_count = [0]
        
        def fake_fit(*args, **kwargs):
            call_count[0] += 1
            # Converge on 3rd call
            if call_count[0] >= 3:
                return {"swd_est_mean": np.array([2.0, 1.5, 1.0, 0.8, 0.5, 0.3])}
            else:
                return {"swd_est_mean": np.array([5.0, 4.0, 3.5, 3.0, 2.5, 2.0])}
        
        mock_model.fit_and_estimate.side_effect = fake_fit
        mock_model.predicted_vs_residual.return_value = {
            "y_predicted": np.array([1.0]),
            "residuals": np.array([0.0]),
        }
        mock_model_class.return_value = mock_model
        
        samples = np.random.randn(1500, 5)
        result = swasd(samples, n_blocks=6, min_iters_per_block=100, 
                      convg_threshold=1.0, verbose=0)
        
        assert result["k_conv"] is not None
        assert result["convergence_reason"] == "swd"
        assert len(result["swd_to_stationary"]) >= 3
    
    @patch('swasd.algorithm.MonotoneSWDModel')
    @patch('swasd.algorithm.rhat_check')
    def test_full_pipeline_3d(self, mock_rhat, mock_model_class):
        """Test full pipeline with 3D samples (multiple chains)."""
        mock_rhat.return_value = (True, 500, 1.05)
        
        samples = np.random.randn(4, 1000, 5)
        result = swasd(samples, n_blocks=6, min_iters_per_block=100, verbose=0)
        
        assert result["k_conv"] is not None
        assert result["convergence_reason"] == "rhat"


# ==============================================================================
# Test Edge Cases
# ==============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_minimum_viable_input(self):
        """Test with minimum viable input size."""
        with patch('swasd.algorithm.convergence_check') as mock_check:
            mock_check.return_value = (None, None)
            
            samples = np.random.randn(500, 2)
            result = swasd(samples, n_blocks=2, min_iters_per_block=50, verbose=0)
            assert result is not None
    
    def test_single_dimension(self):
        """Test with single parameter dimension."""
        with patch('swasd.algorithm.convergence_check') as mock_check:
            mock_check.return_value = (None, None)
            
            samples = np.random.randn(1000, 1)
            result = swasd(samples, n_blocks=4, min_iters_per_block=50, verbose=0)
            assert result is not None
    
    def test_many_blocks(self):
        """Test with many blocks."""
        with patch('swasd.algorithm.convergence_check') as mock_check:
            mock_check.return_value = (None, None)
            
            samples = np.random.randn(2000, 3)
            result = swasd(samples, n_blocks=20, min_iters_per_block=50, verbose=0)
            assert result is not None


# ==============================================================================
# Test History Dictionary
# ==============================================================================

class TestHistoryOutput:
    """Test the structure and content of the returned history dict."""
    
    @patch('swasd.algorithm.convergence_check')
    def test_history_structure_no_convergence(self, mock_check):
        """Test history dict when no convergence."""
        def mock_convergence(*args, **kwargs):
            history = kwargs['history']
            # Simulate what convergence_check adds to history
            if kwargs.get('do_swd_check'):
                history['pairwise_swd_results'].append({})
                history['estimated_swd_results'].append({})
                history['regression_results'].append({})
                history['swd_to_stationary'].append(2.0)
                xk = args[0]
                k = xk.shape[0] if xk.ndim == 2 else xk.shape[1]
                history['convg_check_iterate'].append(k)
            return (None, None)
        
        mock_check.side_effect = mock_convergence
        
        samples = np.random.randn(1000, 3)
        result = swasd(samples, n_blocks=4, min_iters_per_block=50, verbose=0)
        
        assert isinstance(result, dict)
        assert result["k_conv"] is None
        assert "pairwise_swd_results" in result
        assert "estimated_swd_results" in result
        assert "regression_results" in result
        assert "swd_to_stationary" in result
        assert "convg_check_iterate" in result
    
    @patch('swasd.algorithm.convergence_check')
    def test_history_with_rhat(self, mock_check):
        """Test history includes Rhat data when checked."""
        def mock_convergence(*args, **kwargs):
            history = kwargs['history']
            xk = args[0]
            k = xk.shape[0] if xk.ndim == 2 else xk.shape[1]
        
            if kwargs.get('do_rhat_check'):
                history['rhat_best'].append(2.0)
                history['rhat_check_iterate'].append(k)
                history['rhat_best_w'].append(500)
            return (None, None)
        
        mock_check.side_effect = mock_convergence
        
        samples = np.random.randn(4, 1000, 5)
        result = swasd(samples, n_blocks=4, min_iters_per_block=50, verbose=0)
        
        assert "rhat_best" in result
        assert "rhat_check_iterate" in result
        assert len(result["rhat_best"]) > 0  # At least some Rhat checks


if __name__ == "__main__":
    pytest.main([__file__, "-v"])