import numpy as np
import pytest
from unittest.mock import Mock, MagicMock, patch
from collections import defaultdict

# Import the functions to test
from swasd.algorithm import (
    check_is_converged,
    check_where_converged,
    _check_is_converged_step,
    _init_history,
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


class TestGeometricCheckpoints:
    """Test _geometric_checkpoints helper."""
    
    def test_basic_sequence(self):
        """Test basic geometric checkpoint generation."""
        ckpts = _geometric_checkpoints(1000, 100, 1.5)
        assert ckpts[0] == 100
        assert ckpts[-1] == 1000
        assert len(ckpts) > 1
    
    def test_rate_validation(self):
        """Test that rate <= 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="check_rate must be > 1.0"):
            _geometric_checkpoints(1000, 100, 1.0)


class TestLinearCheckpoints:
    """Test _linear_checkpoints helper."""
    
    def test_basic_range(self):
        """Test basic linear checkpoint generation."""
        ckpts = _linear_checkpoints(1000, 100, 50)
        assert ckpts[0] == 100
        assert ckpts[-1] == 1000
    
    def test_step_validation(self):
        """Test that step < 1 raises ValueError."""
        with pytest.raises(ValueError, match="step must be >= 1"):
            _linear_checkpoints(1000, 100, 0)


class TestInitHistory:
    """Test _init_history function."""
    
    def test_creates_correct_structure(self):
        """Test that history has all expected keys."""
        history = _init_history()
        
        expected_keys = {
            "k_conv", "convergence_reason",
            "rhat_check_iterate", "rhat_best_w", "rhat_best",
            "pairwise_swd_results", "true_swd_results",
            "estimated_swd_results", "regression_results",
            "swd_to_stationary", "convg_check_iterate",
        }
        
        assert set(history.keys()) == expected_keys
        assert history["k_conv"] is None
        assert history["convergence_reason"] is None
        assert history["rhat_best"] == []


class TestPrintSummary:
    """Test _print_summary helper."""
    
    def test_rhat_summary(self, capsys):
        """Test summary for Rhat convergence."""
        history = _init_history()
        history["rhat_best"] = [3.0, 2.5, 1.05]
        history["swd_to_stationary"] = []
        
        _print_summary(history, k_conv=1000, reason="rhat",
                      rhat_threshold=1.1, swd_threshold=1.0)
        
        captured = capsys.readouterr()
        assert "CONVERGENCE DETECTED" in captured.out
        assert "1000" in captured.out
        assert "RHAT" in captured.out


# ==============================================================================
# Test _check_is_converged_step Function
# ==============================================================================

class TestCheckIsConvergedStep:
    """Test _check_is_converged_step internal function."""
    
    @patch('swasd.algorithm.rhat_check')
    def test_rhat_success(self, mock_rhat):
        """Test Rhat success triggers return."""
        mock_rhat.return_value = (True, 500, 1.02)
        
        samples = np.random.randn(4, 1000, 5)
        history = _init_history()
        metric_comp = Mock()
        
        k_conv, reason = _check_is_converged_step(
            samples,
            do_rhat_check=True,
            do_swd_check=False,
            history=history,
            metric_comp=metric_comp,
            rhat_wmin=200,
        )
        
        assert k_conv == 1000  # k from samples.shape
        assert reason == "rhat"
        assert history["rhat_best"][-1] == 1.02
    
    @patch('swasd.algorithm.MonotoneSWDModel')
    @patch('swasd.algorithm.rhat_check')
    def test_swd_convergence(self, mock_rhat, mock_model_class):
        """Test SWD convergence detection."""
        # Setup mocks
        mock_metric = Mock()
        mock_metric.compute_blockwise.return_value = {
            "block_pair": [(2, 3), (3, 4)],
            "est_all": [2.0, 1.5],
        }
        
        mock_model = Mock()
        mock_model.fit_and_estimate.return_value = {
            "swd_est_mean": np.array([3.0, 2.0, 1.0, 0.5, 0.3, 0.2]),
        }
        mock_model.predicted_vs_residual.return_value = {
            "y_predicted": np.array([1.0, 1.0]),
            "residuals": np.array([0.0, 0.0]),
        }
        mock_model_class.return_value = mock_model
        
        samples = np.random.randn(1500, 5)
        history = _init_history()
        
        k_conv, reason = _check_is_converged_step(
            samples,
            do_rhat_check=False,
            do_swd_check=True,
            history=history,
            metric_comp=mock_metric,
            n_blocks=6,
            convg_threshold=0.5,
        )
        
        assert k_conv == 1500
        assert reason == "swd"
        assert len(history["swd_to_stationary"]) == 1


# ==============================================================================
# Test check_is_converged Function (Single-Shot Check)
# ==============================================================================

class TestCheckIsConverged:
    """Test check_is_converged function."""
    
    @patch('swasd.algorithm._check_is_converged_step')
    def test_rhat_mode_for_small_k(self, mock_step):
        """Test that small k uses Rhat mode."""
        mock_step.return_value = (None, None)
        
        # k=500 < min_k_for_swd=1500 (250*6)
        samples = np.random.randn(500, 3)
        
        history = check_is_converged(
            samples,
            n_blocks=6,
            min_iters_per_block=250,
            verbose=0
        )
        
        # Verify Rhat was used
        call_kwargs = mock_step.call_args[1]
        assert call_kwargs['do_rhat_check'] == True
        assert call_kwargs['do_swd_check'] == False
    
    @patch('swasd.algorithm._check_is_converged_step')
    def test_swd_mode_for_large_k(self, mock_step):
        """Test that large k uses SWD mode."""
        mock_step.return_value = (None, None)
        
        # k=2000 >= min_k_for_swd=1500
        samples = np.random.randn(2000, 3)
        
        history = check_is_converged(
            samples,
            n_blocks=6,
            min_iters_per_block=250,
            verbose=0
        )
        
        # Verify SWD was used
        call_kwargs = mock_step.call_args[1]
        assert call_kwargs['do_rhat_check'] == False
        assert call_kwargs['do_swd_check'] == True
    
    @patch('swasd.algorithm._check_is_converged_step')
    def test_convergence_detected(self, mock_step, capsys):
        """Test when convergence is detected."""
        mock_step.return_value = (1500, "swd")
        
        def side_effect(*args, **kwargs):
            history = kwargs['history']
            history['swd_to_stationary'].append(0.5)
            history['convg_check_iterate'].append(1500)
            history['k_conv'] = 1500  # ✅ Set this!
            history['convergence_reason'] = 'swd'  # ✅ And this!
            return (1500, "swd")
        
        mock_step.side_effect = side_effect
        
        samples = np.random.randn(1500, 3)
        history = check_is_converged(samples, verbose=1)
        
        assert history["k_conv"] == 1500
        captured = capsys.readouterr()
        assert "Converged" in captured.out


# ==============================================================================
# Test check_where_converged Function (Main Algorithm)
# ==============================================================================

class TestCheckWhereConverged:
    """Test check_where_converged main function."""
    
    def test_validation_n_blocks(self):
        """Test n_blocks < 2 raises ValueError."""
        samples = np.random.randn(1000, 3)
        with pytest.raises(ValueError, match="n_blocks must be >= 2"):
            check_where_converged(samples, n_blocks=1)
    
    def test_validation_min_iters(self):
        """Test min_iters_per_block < 1 raises ValueError."""
        samples = np.random.randn(1000, 3)
        with pytest.raises(ValueError, match="min_iters_per_block must be >= 1"):
            check_where_converged(samples, min_iters_per_block=0)
    
    def test_validation_check_rate(self):
        """Test check_rate <= 1.0 raises ValueError."""
        samples = np.random.randn(1000, 3)
        with pytest.raises(ValueError, match="check_rate should be > 1.0"):
            check_where_converged(samples, check_rate=1.0)
    
    def test_insufficient_iterations(self, capsys):
        """Test behavior when not enough iterations."""
        samples = np.random.randn(100, 3)
        history = check_where_converged(
            samples,
            n_blocks=6,
            min_iters_per_block=50,
            verbose=1
        )
        
        assert history["k_conv"] is None
        captured = capsys.readouterr()
        assert "Not enough iterations" in captured.out
    
    @patch('swasd.algorithm._check_is_converged_step')
    def test_early_convergence_rhat(self, mock_step):
        """Test early convergence via Rhat."""
        call_count = [0]
        
        def side_effect(*args, **kwargs):
            call_count[0] += 1
            history = kwargs['history']
            
            # First call (Rhat) converges
            if call_count[0] == 1:
                history['rhat_best'].append(1.02)
                history['rhat_check_iterate'].append(200)
                history['k_conv'] = 200
                history['convergence_reason'] = 'rhat'
                return (200, 'rhat')
            
            return (None, None)
        
        mock_step.side_effect = side_effect
        
        samples = np.random.randn(2000, 3)
        history = check_where_converged(
            samples,
            n_blocks=6,
            min_iters_per_block=100,
            rhat_wmin=100,
            verbose=0
        )
        
        assert history["k_conv"] == 200
        assert history["convergence_reason"] == "rhat"
    
    @patch('swasd.algorithm._check_is_converged_step')
    def test_swd_convergence(self, mock_step):
        """Test SWD convergence."""
        call_count = [0]
        
        def side_effect(*args, **kwargs):
            call_count[0] += 1
            history = kwargs['history']
            
            # Rhat calls don't converge
            if kwargs.get('do_rhat_check'):
                history['rhat_best'].append(2.0)
                return (None, None)
            
            # Third SWD call converges
            if kwargs.get('do_swd_check'):
                if call_count[0] >= 8:
                    history['swd_to_stationary'].append(0.5)
                    history['convg_check_iterate'].append(1200)
                    history['k_conv'] = 1200
                    history['convergence_reason'] = 'swd'
                    return (1200, 'swd')
                else:
                    history['swd_to_stationary'].append(2.0)
                    return (None, None)
            
            return (None, None)
        
        mock_step.side_effect = side_effect
        
        samples = np.random.randn(2000, 3)
        history = check_where_converged(
            samples,
            n_blocks=6,
            min_iters_per_block=100,
            convg_threshold=1.0,
            verbose=0
        )
        
        assert history["k_conv"] == 1200
        assert history["convergence_reason"] == "swd"
    
    @patch('swasd.algorithm._check_is_converged_step')
    def test_no_convergence(self, mock_step, capsys):
        """Test when no convergence detected."""
        def side_effect(*args, **kwargs):
            history = kwargs['history']
            if kwargs.get('do_swd_check'):
                history['swd_to_stationary'].append(2.0)
            return (None, None)
        
        mock_step.side_effect = side_effect
        
        samples = np.random.randn(1000, 3)
        history = check_where_converged(
            samples,
            n_blocks=6,
            min_iters_per_block=100,
            verbose=1
        )
        
        assert history["k_conv"] is None
        captured = capsys.readouterr()
        assert "CONVERGENCE NOT DETECTED" in captured.out
    
    @patch('swasd.algorithm._check_is_converged_step')
    def test_verbose_levels(self, mock_step):
        """Test different verbose levels work."""
        mock_step.return_value = (None, None)
        samples = np.random.randn(800, 3)
        
        # Silent
        history = check_where_converged(samples, n_blocks=4, 
                                       min_iters_per_block=50, verbose=0)
        assert history is not None
        
        # Summary
        history = check_where_converged(samples, n_blocks=4,
                                       min_iters_per_block=50, verbose=1)
        assert history is not None
        
        # Progress bar
        history = check_where_converged(samples, n_blocks=4,
                                       min_iters_per_block=50, verbose=2)
        assert history is not None
    
    @patch('swasd.algorithm._check_is_converged_step')
    def test_checkpoint_logic(self, mock_step):
        """Test that checkpoints are created correctly."""
        rhat_calls = []
        swd_calls = []
        
        def side_effect(*args, **kwargs):
            k = args[0].shape[0] if args[0].ndim == 2 else args[0].shape[1]
            if kwargs.get('do_rhat_check'):
                rhat_calls.append(k)
            if kwargs.get('do_swd_check'):
                swd_calls.append(k)
            return (None, None)
        
        mock_step.side_effect = side_effect
        
        samples = np.random.randn(2000, 3)
        history = check_where_converged(
            samples,
            n_blocks=6,
            min_iters_per_block=100,
            rhat_wmin=100,
            rhat_check_iter=50,
            check_rate=1.5,
            verbose=0
        )
        
        # Should have Rhat checks
        assert len(rhat_calls) > 0
        # Should have SWD checks
        assert len(swd_calls) > 0


# ==============================================================================
# Test Integration
# ==============================================================================

class TestIntegration:
    """Integration tests with minimal mocking."""
    
    @patch('swasd.algorithm.MonotoneSWDModel')
    @patch('swasd.algorithm.rhat_check')
    def test_full_pipeline_rhat(self, mock_rhat, mock_model):
        """Test full pipeline with Rhat convergence."""
        # Rhat converges on second check
        call_count = [0]
        
        def rhat_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] >= 2:
                return (True, 500, 1.02)
            return (False, 500, 2.0)
        
        mock_rhat.side_effect = rhat_side_effect
        
        samples = np.random.randn(4, 1000, 5)
        history = check_where_converged(
            samples,
            n_blocks=6,
            min_iters_per_block=100,
            verbose=0
        )
        
        assert history["k_conv"] is not None
        assert history["convergence_reason"] == "rhat"
    
    @patch('swasd.algorithm.MonotoneSWDModel')
    @patch('swasd.algorithm.rhat_check')
    def test_full_pipeline_swd(self, mock_rhat, mock_model_class):
        """Test full pipeline with SWD convergence."""
        # Rhat never converges
        mock_rhat.return_value = (False, 500, 2.0)
        
        # SWD converges eventually
        call_count = [0]
        
        def fit_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] >= 3:
                return {"swd_est_mean": np.array([2.0, 1.5, 1.0, 0.8, 0.5, 0.3])}
            else:
                return {"swd_est_mean": np.array([5.0, 4.0, 3.5, 3.0, 2.5, 2.0])}
        
        mock_model = Mock()
        mock_model.fit_and_estimate.side_effect = fit_side_effect
        mock_model.predicted_vs_residual.return_value = {
            "y_predicted": np.array([1.0]),
            "residuals": np.array([0.0]),
        }
        mock_model_class.return_value = mock_model
        
        samples = np.random.randn(1500, 5)
        history = check_where_converged(
            samples,
            n_blocks=6,
            min_iters_per_block=100,
            convg_threshold=1.0,
            verbose=0
        )
        
        assert history["k_conv"] is not None
        assert history["convergence_reason"] == "swd"


# ==============================================================================
# Test Edge Cases
# ==============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    @patch('swasd.algorithm._check_is_converged_step')
    def test_single_dimension(self, mock_step):
        """Test with single parameter dimension."""
        mock_step.return_value = (None, None)
        
        samples = np.random.randn(1000, 1)
        history = check_where_converged(
            samples,
            n_blocks=4,
            min_iters_per_block=50,
            verbose=0
        )
        assert history is not None
    
    @patch('swasd.algorithm._check_is_converged_step')
    def test_3d_samples(self, mock_step):
        """Test with 3D samples (multiple chains)."""
        mock_step.return_value = (None, None)
        
        samples = np.random.randn(4, 1000, 3)
        history = check_where_converged(
            samples,
            n_blocks=4,
            min_iters_per_block=50,
            verbose=0
        )
        assert history is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])