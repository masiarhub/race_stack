import torch

from opp_prediction.ensemble import Ensemble


def ensemble(members=5):
    return Ensemble(input_columns=[f'x_{i}' for i in range(19)], members=members, hidden=16, layers=2)


def test_distribution_has_one_mean_and_std_per_member_sample_and_action():
    mu, sigma = ensemble().distribution(torch.zeros(7, 19))
    assert mu.shape == sigma.shape == (5, 7, 2)
    assert (sigma > 0).all()


def test_members_differ_by_initialization():
    mu, _ = ensemble(members=2).distribution(torch.ones(1, 19))
    assert not torch.allclose(mu[0], mu[1])


def test_uncertainty_stays_finite_far_from_the_data():
    _, sigma = ensemble().distribution(torch.full((3, 19), 1e6))
    assert torch.isfinite(sigma).all() and (sigma > 0).all()


def test_distribution_is_in_physical_units():
    model = ensemble(members=1)
    model.fit_normalization(torch.zeros(10, 19), torch.tensor([[0.0, 100.0], [0.0, 102.0]]).repeat(5, 1))
    mu, _ = model.distribution(torch.zeros(1, 19))
    assert abs(mu[0, 0, 1].item() - 101.0) < 10.0  # an untrained member stays near the mean of the action


def test_checkpoint_round_trip(tmp_path):
    model = ensemble()
    model.fit_normalization(torch.randn(20, 19), torch.randn(20, 2))
    path = tmp_path / 'ensemble.pt'
    model.save(path)
    loaded = Ensemble.load(path)
    o = torch.randn(4, 19)
    assert torch.equal(model.distribution(o)[0], loaded.distribution(o)[0])
    assert torch.equal(model.distribution(o)[1], loaded.distribution(o)[1])
    assert loaded.input_columns == model.input_columns
