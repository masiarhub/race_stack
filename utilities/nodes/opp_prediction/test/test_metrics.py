import numpy as np
import pytest

from opp_prediction.metrics import accuracy, calibration, coverage_expected, min_fde, sparsification, spread, summarize

MEMBERS = np.array([0, 0, 1, 1])
S, P, H = 500, 20, 3


def constant_error(value):
    """[samples, particles, horizon, 2], the particles are spread evenly around the given error."""
    offsets = np.linspace(-0.1, 0.1, 4).reshape(1, 4, 1, 1)
    return value + np.broadcast_to(offsets, (6, 4, 2, 2)).copy()


def test_accuracy_is_the_error_of_the_particle_mean():
    result = accuracy(constant_error(1.0))
    assert np.allclose(result['fde'], 1.0)


def test_ade_averages_the_steps_so_far():
    rel = constant_error(0.0)
    rel[:, :, 1] += 3.0
    result = accuracy(rel)
    assert np.allclose(result['ade'][0], 0.0)
    assert np.allclose(result['ade'][1], 1.5)


def test_steps_without_a_true_position_are_ignored():
    rel = constant_error(1.0)
    rel[0, :, 1] = np.nan
    assert np.allclose(accuracy(rel)['fde'], 1.0)


def test_truth_outside_the_particles_lands_in_the_outer_ranks():
    above = calibration(constant_error(5.0))['rank_histogram']  # all particles above the truth
    below = calibration(constant_error(-5.0))['rank_histogram']
    assert above[..., 0].sum() == above.sum()
    assert below[..., -1].sum() == below.sum()


def normal_errors(particle_std, rng):
    truth = rng.standard_normal((S, 1, H, 2))
    return particle_std * rng.standard_normal((S, P, H, 2)) - truth


def test_coverage_of_calibrated_particles_matches_the_expectation():
    result = calibration(normal_errors(1.0, np.random.default_rng(0)))
    for level in ('0.5', '0.9'):
        assert result['coverage'][level] == pytest.approx(np.full((H, 2), coverage_expected(20)[level]), abs=0.05)


def test_overconfident_particles_cover_too_little():
    result = calibration(normal_errors(0.2, np.random.default_rng(0)))
    assert (result['coverage']['0.9'] < 0.7).all()


def test_spread_separates_disagreement_from_noise_within_a_member():
    rel = np.zeros((1, 4, 1, 2))
    rel[0, 2:] = 2.0  # members disagree, no spread within a member
    result = spread(rel, MEMBERS)
    assert np.allclose(result['aleatoric'], 0.0)
    assert np.allclose(result['epistemic'], 1.0)
    assert np.allclose(result['total'], 1.0)


def test_noise_within_a_member_is_aleatoric():
    rel = np.zeros((1, 4, 1, 2))
    rel[0, [1, 3]] = 2.0  # same mean in both members
    result = spread(rel, MEMBERS)
    assert np.allclose(result['epistemic'], 0.0)
    assert result['aleatoric'][0, 0, 0] > 0.0


def test_removing_uncertain_samples_lowers_the_error_when_spread_is_informative():
    rng = np.random.default_rng(0)
    uncertainty = rng.uniform(0.1, 2.0, (2000, 1, 1))
    errors = uncertainty * np.abs(rng.standard_normal((2000, 1, 1)))
    remaining = sparsification(errors, uncertainty, fractions=(0.0, 0.5))
    assert remaining[0, 0, 1] < 0.7 * remaining[0, 0, 0]


def test_uninformative_spread_leaves_the_error_unchanged():
    rng = np.random.default_rng(0)
    errors = np.abs(rng.standard_normal((4000, 1, 1)))
    remaining = sparsification(errors, rng.uniform(size=(4000, 1, 1)), fractions=(0.0, 0.5))
    assert remaining[0, 0, 1] == pytest.approx(remaining[0, 0, 0], rel=0.1)


def test_min_fde_takes_the_closest_particle():
    rel = np.array([[[[3.0, 4.0]], [[0.3, 0.4]]]])
    assert min_fde(rel) == pytest.approx([0.5])


def test_summary_reports_each_requested_step_and_the_baseline():
    rel = constant_error(1.0)
    report = summarize(rel, np.array([0, 0, 1, 1]), np.full((6, 2, 2), 3.0), steps=(1, 2), dt=0.05)
    assert set(report['steps']) == {'1', '2'}
    step = report['steps']['2']
    assert step['time_s'] == pytest.approx(0.1)
    assert step['fde']['s'] == pytest.approx(1.0)
    assert step['baseline_fde']['d'] == pytest.approx(3.0)
    assert step['samples'] == 6
