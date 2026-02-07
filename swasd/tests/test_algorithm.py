import numpy as np
import pytest
from unittest.mock import patch, call
from collections import defaultdict

from swasd.algorithm import swasd, _geometric_checkpoints, _linear_checkpoints

# ---------------------------------------------------------------------------
# patch target strings  (must match the module where the names are LOOKED UP)
# ---------------------------------------------------------------------------
P_RHAT      = "swasd.algorithm.rhat_check"
P_BLOCKWISE = "swasd.algorithm.MetricComputer.compute_blockwise"
P_FIT_EST   = "swasd.algorithm.MonotoneSWDModel.fit_and_estimate"
P_PVSR      = "swasd.algorithm.MonotoneSWDModel.predicted_vs_residual"


# ---------------------------------------------------------------------------
# deterministic mock return-values  (shared across many tests)
# ---------------------------------------------------------------------------

def _blockwise_result(samples, metric="swd", num_blocks=6, mode="all_pairs", **kw):
    """Mimic MetricComputer.compute_blockwise output."""
    import itertools

    # your test uses 1..B indexing (keep that consistent with the rest of your tests)
    if mode == "adjacent":
        pairs = [(i, i + 1) for i in range(1, num_blocks)]
    else:
        pairs = list(itertools.combinations(range(1, num_blocks + 1), 2))

    est = [abs(a - b) * 0.4 + 0.05 for a, b in pairs]
    return {"block_pair": pairs, "est_all": est, "mean_all": est}


def _fit_and_estimate(num_blocks, block_pairs, sw_values, **kw):
    """Monotone curve ending at 0.3 (below most test thresholds)."""
    mean = np.linspace(2.0, 0.3, num_blocks)
    return dict(swd_est_mean=mean, swd_est_median=mean,
                swd_lci=mean * 0.8, swd_uci=mean * 1.2, idata=None)


def _predicted_vs_residual(y_log, i, j, compute_fit_diagnostic=True):
    out = dict(y_predicted=y_log, residuals=np.zeros_like(y_log),
               lower_CI=y_log - 0.1, upper_CI=y_log + 0.1)
    if compute_fit_diagnostic:
        out["fit_diagnostic"] = 0.05
    return out


# ---------------------------------------------------------------------------
# 1.  _geometric_checkpoints  (pure, unchanged – kept for completeness)
# ---------------------------------------------------------------------------

class TestGeometricCheckpoints:

    def test_basic_sequence(self):
        cps = _geometric_checkpoints(1000, 100, 1.5)
        assert cps[0] == 100
        assert cps[-1] == 1000
        assert all(cps[i] < cps[i + 1] for i in range(len(cps) - 1))

    def test_small_n_iters(self):
        cps = _geometric_checkpoints(100, 10, 2.0)
        assert cps[0] == 10
        assert cps[-1] == 100

    def test_rate_le_1_raises(self):
        with pytest.raises(ValueError, match="check_rate must be > 1.0"):
            _geometric_checkpoints(1000, 100, 1.0)
        with pytest.raises(ValueError, match="check_rate must be > 1.0"):
            _geometric_checkpoints(1000, 100, 0.5)

    def test_no_duplicates(self):
        cps = _geometric_checkpoints(1000, 100, 1.1)
        assert len(cps) == len(set(cps.tolist()))

    def test_always_ends_at_n_iters(self):
        for n in (100, 500, 1000, 5000):
            assert _geometric_checkpoints(n, 50, 1.5)[-1] == n


# ---------------------------------------------------------------------------
# 2.  _linear_checkpoints  (NEW in v2)
# ---------------------------------------------------------------------------

class TestLinearCheckpoints:

    def test_basic_range(self):
        cps = _linear_checkpoints(n_iters=1000, start=100, step=50)
        assert cps[0] == 100
        assert cps[-1] == 1000          # stop defaults to n_iters
        # strictly increasing, gaps ≤ step
        diffs = np.diff(cps)
        assert all(d > 0 for d in diffs)
        assert all(d <= 50 for d in diffs)

    def test_explicit_stop(self):
        cps = _linear_checkpoints(1000, 100, 50, stop=300)
        assert cps[0] == 100
        assert cps[-1] == 300

    def test_stop_clamped_to_n_iters(self):
        """stop > n_iters must be clamped."""
        cps = _linear_checkpoints(200, 50, 30, stop=9999)
        assert cps[-1] == 200

    def test_step_lt_1_raises(self):
        with pytest.raises(ValueError, match="step must be >= 1"):
            _linear_checkpoints(1000, 100, step=0)

    def test_start_gt_stop_returns_empty(self):
        cps = _linear_checkpoints(1000, 500, 10, stop=200)
        assert len(cps) == 0
        assert cps.dtype == int

    def test_no_duplicates(self):
        cps = _linear_checkpoints(1000, 100, 100, stop=600)
        assert len(cps) == len(set(cps.tolist()))

    def test_start_equals_stop(self):
        """Exactly one checkpoint when start == stop."""
        cps = _linear_checkpoints(500, 200, 50, stop=200)
        assert list(cps) == [200]

    def test_step_larger_than_range(self):
        """step > (stop - start) → [start, stop] only."""
        cps = _linear_checkpoints(1000, 100, 9999, stop=400)
        assert list(cps) == [100, 400]

    def test_returns_int_dtype(self):
        cps = _linear_checkpoints(500, 50, 25)
        assert cps.dtype == int


# ---------------------------------------------------------------------------
# 3.  __init__ – new rhat parameters are stored correctly
# ---------------------------------------------------------------------------

class TestInitRhatParams:

    @pytest.fixture
    def samples(self):
        np.random.seed(0)
        return np.random.randn(2000, 3)

    def test_defaults(self, samples):
        det = swasd(samples=samples)
        assert det.rhat_threshold   == 1.01
        assert det.rhat_wmin        == 100
        assert det.rhat_method      == "rank"
        assert det.rhat_num_windows == 5
        assert det.rhat_stop_factor == 1.5
        assert det.rhat_check_iter  == 100

    def test_custom_values(self, samples):
        det = swasd(samples, rhat_threshold=1.1, rhat_wmin=50,
                    rhat_method="split", rhat_num_windows=10,
                    rhat_stop_factor=2.0, rhat_check_iter=25)
        assert det.rhat_threshold   == 1.1
        assert det.rhat_wmin        == 50
        assert det.rhat_method      == "split"
        assert det.rhat_num_windows == 10
        assert det.rhat_stop_factor == 2.0
        assert det.rhat_check_iter  == 25


# ---------------------------------------------------------------------------
# 4.  run() – input-validation  (unchanged logic; kept for regression safety)
# ---------------------------------------------------------------------------

class TestValidation:

    @pytest.fixture
    def samples(self):
        return np.random.randn(2000, 3)

    def test_n_blocks_lt_2(self, samples):
        with pytest.raises(ValueError, match="n_blocks must be >= 2"):
            swasd(samples, n_blocks=1, verbose=False).run()

    def test_min_iters_lt_1(self, samples):
        with pytest.raises(ValueError, match="min_iters_per_block must be >= 1"):
            swasd(samples, min_iters_per_block=0, verbose=False).run()

    def test_check_rate_le_1(self, samples):
        with pytest.raises(ValueError, match="check_rate should be > 1.0"):
            swasd(samples, check_rate=0.9, verbose=False).run()

    def test_bad_block_mode(self, samples):
        with pytest.raises(ValueError, match="block_mode must be"):
            swasd(samples, block_mode="bad", verbose=False).run()

    def test_true_swd_no_ref(self, samples):
        with pytest.raises(ValueError, match="true_swd=True requires"):
            swasd(samples, true_swd=True, true_samples=None, verbose=False).run()

    def test_n_iters_exceeds_2d(self):
        with pytest.raises(ValueError, match="exceeds available"):
            swasd(np.random.randn(100, 3), n_iters=200, verbose=False).run()

    def test_n_iters_exceeds_3d(self):
        with pytest.raises(ValueError, match="exceeds available"):
            swasd(np.random.randn(2, 100, 3), n_iters=200, verbose=False).run()

    def test_insufficient_samples_returns_none(self):
        """k0 = min_iters_per_block * n_blocks > n_iters → immediate None."""
        result = swasd(np.random.randn(50, 3), n_blocks=6,
                       min_iters_per_block=100, verbose=False).run()
        assert result["k_conv"] is None


# ---------------------------------------------------------------------------
# 5.  Checkpoint-merge arithmetic  (new in v2)
#     Reconstruct the same logic run() uses and verify structural properties.
# ---------------------------------------------------------------------------

class TestCheckpointMerge:

    @staticmethod
    def _merge(n_iters, min_ipb, n_blocks, check_rate,
               rhat_wmin, rhat_check_iter, rhat_stop_factor):
        k0 = min_ipb * n_blocks
        swasd_cs = _geometric_checkpoints(n_iters, k0, check_rate)
        rhat_stop = min(n_iters, int(np.ceil(rhat_stop_factor * k0)))
        rhat_cs   = _linear_checkpoints(n_iters, rhat_wmin,
                                        rhat_check_iter, rhat_stop)
        merged = np.unique(np.concatenate([rhat_cs, swasd_cs])).astype(int)
        return merged, set(rhat_cs.tolist()), set(swasd_cs.tolist()), rhat_stop

    def test_merged_sorted_unique(self):
        merged, _, _, _ = self._merge(2000, 50, 6, 1.2, 100, 50, 1.5)
        assert list(merged) == sorted(set(merged.tolist()))

    def test_both_subsets_present(self):
        merged, rset, sset, _ = self._merge(2000, 50, 6, 1.2, 100, 50, 1.5)
        mset = set(merged.tolist())
        assert rset.issubset(mset)
        assert sset.issubset(mset)

    def test_rhat_checkpoints_respect_rhat_stop(self):
        _, rset, _, rhat_stop = self._merge(2000, 50, 6, 1.2, 100, 50, 1.5)
        assert all(k <= rhat_stop for k in rset)

    def test_empty_rhat_when_wmin_too_large(self):
        """rhat_wmin > rhat_stop → rhat_checks is empty."""
        merged, rset, sset, _ = self._merge(2000, 50, 6, 1.2,
                                            rhat_wmin=99999,
                                            rhat_check_iter=50,
                                            rhat_stop_factor=1.5)
        assert len(rset) == 0
        # merged still has the geometric checkpoints
        assert merged.tolist() == sorted(sset)


# ---------------------------------------------------------------------------
# 6.  run() – Rhat early-exit path
# ---------------------------------------------------------------------------

class TestRhatEarlyExit:
    """Every test here patches the full SWD stack so it never touches Stan."""

    # --- helpers -----------------------------------------------------------
    @staticmethod
    def _samples_2d(n=2000, d=3):
        np.random.seed(0)
        return np.random.randn(n, d)

    @staticmethod
    def _samples_3d(chains=4, n=500, d=3):
        np.random.seed(7)
        return np.random.randn(chains, n, d)

    # common kwargs that guarantee rhat checkpoints exist and the w_upper
    # guard passes for at least some of them
    _RHAT_KW = dict(
        n_blocks=6, min_iters_per_block=50,   # k0 = 300
        rhat_wmin=100, rhat_check_iter=50,
        rhat_stop_factor=3.0,                  # rhat_stop = 900
        rhat_threshold=1.01,
        verbose=False,
    )

    # --- 6a.  rhat succeeds → early return ---------------------------------

    @patch(P_BLOCKWISE, side_effect=_blockwise_result)
    @patch(P_FIT_EST,   side_effect=_fit_and_estimate)
    @patch(P_PVSR,      side_effect=_predicted_vs_residual)
    @patch(P_RHAT,      return_value=(True, 150, 1.005))
    def test_convergence_reason_is_rhat(self, mock_rhat, *_mocks):
        result = swasd(self._samples_2d(), **self._RHAT_KW).run()
        assert result["convergence_reason"] == "rhat"
        assert result["k_conv"] is not None
        mock_rhat.assert_called()

    @patch(P_BLOCKWISE, side_effect=_blockwise_result)
    @patch(P_FIT_EST,   side_effect=_fit_and_estimate)
    @patch(P_PVSR,      side_effect=_predicted_vs_residual)
    @patch(P_RHAT,      return_value=(True, 120, 1.002))
    def test_swd_blockwise_never_called_on_rhat_success(self, _rhat, _pvsr,
                                                         _fit, mock_bw):
        swasd(self._samples_2d(), **self._RHAT_KW).run()
        mock_bw.assert_not_called()

    @patch(P_BLOCKWISE, side_effect=_blockwise_result)
    @patch(P_FIT_EST,   side_effect=_fit_and_estimate)
    @patch(P_PVSR,      side_effect=_predicted_vs_residual)
    @patch(P_RHAT,      return_value=(True, 200, 1.0))
    def test_rhat_history_keys_present(self, *_mocks):
        result = swasd(self._samples_2d(), **self._RHAT_KW).run()
        assert "rhat_check_iterate" in result
        assert "rhat_best_w"        in result
        assert "rhat_best"          in result
        # at least one entry
        assert len(result["rhat_check_iterate"]) >= 1

    @patch(P_BLOCKWISE, side_effect=_blockwise_result)
    @patch(P_FIT_EST,   side_effect=_fit_and_estimate)
    @patch(P_PVSR,      side_effect=_predicted_vs_residual)
    @patch(P_RHAT,      return_value=(True, 200, 1.0))
    def test_rhat_early_return_has_no_swd_history(self, *_mocks):
        """SWD history keys are absent (defaultdict never populated them)."""
        result = swasd(self._samples_2d(), **self._RHAT_KW).run()
        # these keys exist only if the SWD branch ran
        assert "swd_to_stationary"       not in result
        assert "pairwise_swd_results"    not in result
        assert "estimated_swd_results"   not in result

    # --- 6b.  rhat_check receives correct arguments ------------------------

    @patch(P_BLOCKWISE, side_effect=_blockwise_result)
    @patch(P_FIT_EST,   side_effect=_fit_and_estimate)
    @patch(P_PVSR,      side_effect=_predicted_vs_residual)
    def test_rhat_called_with_correct_kwargs(self, *_mocks):
        captured = {}
        def _capture(xk, **kw):
            captured.update(kw)
            captured["xk_shape"] = xk.shape
            return (True, 150, 1.0)

        with patch(P_RHAT, side_effect=_capture):
            swasd(self._samples_2d(), **self._RHAT_KW).run()

        assert captured["rhat_threshold"] == 1.01
        assert captured["method"]         == "rank"
        assert "windows" in captured
        # windows is a numpy array of ints
        assert captured["windows"].dtype == int

    @patch(P_BLOCKWISE, side_effect=_blockwise_result)
    @patch(P_FIT_EST,   side_effect=_fit_and_estimate)
    @patch(P_PVSR,      side_effect=_predicted_vs_residual)
    def test_rhat_receives_3d_slice(self, *_mocks):
        """With 3D samples the slice passed to rhat_check keeps dim-0 intact."""
        captured_shapes = []
        def _capture(xk, **kw):
            captured_shapes.append(xk.shape)
            return (True, 150, 1.0)

        with patch(P_RHAT, side_effect=_capture):
            swasd(self._samples_3d(), **self._RHAT_KW).run()

        # first shape that was captured must have 4 chains and 3 params
        assert captured_shapes[0][0] == 4   # chains
        assert captured_shapes[0][2] == 3   # params

    # --- 6c.  w_upper guard prevents the call when k is too small ----------

    @patch(P_BLOCKWISE, side_effect=_blockwise_result)
    @patch(P_FIT_EST,   side_effect=_fit_and_estimate)
    @patch(P_PVSR,      side_effect=_predicted_vs_residual)
    def test_rhat_not_called_when_w_upper_le_wmin(self, *_mocks):
        """rhat_wmin=900, k0=300, rhat_stop=ceil(1.1*300)=330.
        All rhat checkpoints ≤ 330 → 0.95*330 = 313 < 900 → guard fails."""
        with patch(P_RHAT) as mock_rhat:
            mock_rhat.return_value = (False, 0, 9.9)
            swasd(self._samples_2d(),
                  n_blocks=6, min_iters_per_block=50,
                  rhat_wmin=900, rhat_check_iter=50,
                  rhat_stop_factor=1.1,
                  verbose=False).run()
            mock_rhat.assert_not_called()


# ---------------------------------------------------------------------------
# 7.  run() – Rhat fails → falls through to SWD branch
# ---------------------------------------------------------------------------

class TestRhatFailsThenSWD:

    @staticmethod
    def _samples():
        np.random.seed(0)
        return np.random.randn(2000, 3)

    _KW = dict(
        n_blocks=6, min_iters_per_block=50,
        rhat_wmin=100, rhat_check_iter=50,
        rhat_stop_factor=3.0,
        convg_threshold=100.0,          # very loose → SWD converges
        verbose=False,
    )

    @patch(P_BLOCKWISE, side_effect=_blockwise_result)
    @patch(P_FIT_EST,   side_effect=_fit_and_estimate)
    @patch(P_PVSR,      side_effect=_predicted_vs_residual)
    @patch(P_RHAT,      return_value=(False, 150, 1.5))
    def test_swd_history_populated(self, *_mocks):
        result = swasd(self._samples(), **self._KW).run()
        assert len(result["swd_to_stationary"])      >= 1
        assert len(result["convg_check_iterate"])    >= 1
        assert len(result["pairwise_swd_results"])   >= 1

    @patch(P_BLOCKWISE, side_effect=_blockwise_result)
    @patch(P_FIT_EST,   side_effect=_fit_and_estimate)
    @patch(P_PVSR,      side_effect=_predicted_vs_residual)
    @patch(P_RHAT,      return_value=(False, 150, 1.5))
    def test_rhat_history_also_populated(self, *_mocks):
        """Even though rhat failed, its history entries are still written."""
        result = swasd(self._samples(), **self._KW).run()
        assert len(result["rhat_best"]) >= 1

    @patch(P_BLOCKWISE, side_effect=_blockwise_result)
    @patch(P_FIT_EST,   side_effect=_fit_and_estimate)
    @patch(P_PVSR,      side_effect=_predicted_vs_residual)
    @patch(P_RHAT,      return_value=(False, 150, 1.5))
    def test_convergence_reason_is_swd(self, *_mocks):
        result = swasd(self._samples(), **self._KW).run()
        assert result["convergence_reason"] == "swd"


# ---------------------------------------------------------------------------
# 8.  run() – SWD-only path details  (rhat disabled via high wmin)
# ---------------------------------------------------------------------------

class TestSWDPath:
    """rhat is effectively disabled; tests focus on the SWD loop."""

    @staticmethod
    def _samples(n=2000):
        np.random.seed(0)
        return np.random.randn(n, 3)

    # rhat_wmin so high that w_upper never exceeds it
    _KW = dict(n_blocks=6, min_iters_per_block=50,
               rhat_wmin=99999, rhat_stop_factor=1.5,
               verbose=False)

    @patch(P_BLOCKWISE, side_effect=_blockwise_result)
    @patch(P_FIT_EST,   side_effect=_fit_and_estimate)
    @patch(P_PVSR,      side_effect=_predicted_vs_residual)
    def test_history_lengths_consistent(self, *_mocks):
        result = swasd(self._samples(), convg_threshold=100.0, **self._KW).run()
        n = len(result["swd_to_stationary"])
        assert n >= 1
        assert len(result["convg_check_iterate"])   == n
        assert len(result["pairwise_swd_results"])  == n
        assert len(result["estimated_swd_results"]) == n
        assert len(result["regression_results"])    == n

    @patch(P_BLOCKWISE, side_effect=_blockwise_result)
    @patch(P_FIT_EST,   side_effect=_fit_and_estimate)
    @patch(P_PVSR,      side_effect=_predicted_vs_residual)
    def test_no_convergence_when_threshold_too_tight(self, *_mocks):
        """_fit_and_estimate ends at 0.3; set threshold below that."""
        result = swasd(self._samples(), convg_threshold=0.01, **self._KW).run()
        assert result["k_conv"] is None
        assert "convergence_reason" not in result

    @patch(P_BLOCKWISE, side_effect=_blockwise_result)
    @patch(P_FIT_EST,   side_effect=_fit_and_estimate)
    @patch(P_PVSR,      side_effect=_predicted_vs_residual)
    def test_n_iters_caps_all_checkpoints(self, *_mocks):
        result = swasd(self._samples(n=2000), n_iters=800,
                       convg_threshold=100.0, **self._KW).run()
        for k in result["convg_check_iterate"]:
            assert k <= 800

    # -- wo_init_block filtering --------------------------------------------

    @patch(P_BLOCKWISE, side_effect=_blockwise_result)
    @patch(P_PVSR,      side_effect=_predicted_vs_residual)
    def test_wo_init_block_true_filters_block_1(self, _pvsr, _bw):
        """Pairs containing block index 1 must be stripped before regression."""
        captured = {}
        def _cap(num_blocks, block_pairs, sw_values, **kw):
            captured["bp"] = block_pairs.copy()
            return _fit_and_estimate(num_blocks, block_pairs, sw_values, **kw)
        with patch(P_FIT_EST, side_effect=_cap):
            swasd(self._samples(), wo_init_block=True,
                  convg_threshold=100.0, **self._KW).run()
        assert 1 not in captured["bp"].flatten()

    @patch(P_BLOCKWISE, side_effect=_blockwise_result)
    @patch(P_PVSR,      side_effect=_predicted_vs_residual)
    def test_wo_init_block_false_keeps_block_1(self, _pvsr, _bw):
        captured = {}
        def _cap(num_blocks, block_pairs, sw_values, **kw):
            captured["bp"] = block_pairs.copy()
            return _fit_and_estimate(num_blocks, block_pairs, sw_values, **kw)
        with patch(P_FIT_EST, side_effect=_cap):
            swasd(self._samples(), wo_init_block=False,
                  convg_threshold=100.0, **self._KW).run()
        assert 1 in captured["bp"].flatten()

    # -- 3D samples ---------------------------------------------------------

    @patch(P_BLOCKWISE, side_effect=_blockwise_result)
    @patch(P_FIT_EST,   side_effect=_fit_and_estimate)
    @patch(P_PVSR,      side_effect=_predicted_vs_residual)
    def test_3d_samples_complete(self, *_mocks):
        np.random.seed(0)
        s = np.random.randn(4, 500, 3)
        result = swasd(s, convg_threshold=100.0, **self._KW).run()
        assert "k_conv" in result


# ---------------------------------------------------------------------------
# 9.  verbose smoke  (nothing crashes, stdout behaves)
# ---------------------------------------------------------------------------

class TestVerbose:

    @staticmethod
    def _samples():
        np.random.seed(0)
        return np.random.randn(2000, 3)

    _KW = dict(n_blocks=6, min_iters_per_block=50,
               rhat_wmin=99999, convg_threshold=100.0)

    @patch(P_BLOCKWISE, side_effect=_blockwise_result)
    @patch(P_FIT_EST,   side_effect=_fit_and_estimate)
    @patch(P_PVSR,      side_effect=_predicted_vs_residual)
    def test_verbose_true_produces_output(self, _mock_pvsr, _mock_fit, _mock_bw, capsys):
        swasd(self._samples(), verbose=True, **self._KW).run()
        out = capsys.readouterr().out
        assert len(out) > 0

    @patch(P_BLOCKWISE, side_effect=_blockwise_result)
    @patch(P_FIT_EST,   side_effect=_fit_and_estimate)
    @patch(P_PVSR,      side_effect=_predicted_vs_residual)
    def test_verbose_false_is_silent(self, _mock_pvsr, _mock_fit, _mock_bw, capsys):
        swasd(self._samples(), verbose=False, **self._KW).run()
        out = capsys.readouterr().out
        assert out == ""



if __name__ == "__main__":
    pytest.main([__file__, "-v"])
