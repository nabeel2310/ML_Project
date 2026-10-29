import torch

from src.crop_disease.losses import supervised_contrastive_loss


def test_contrastive_loss_is_finite_when_labels_have_positive_pairs():
    embeddings = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    labels = torch.tensor([0, 0, 1, 1])
    loss = supervised_contrastive_loss(embeddings, labels)
    assert torch.isfinite(loss)
    assert loss.item() >= 0


def test_contrastive_loss_returns_zero_without_positive_pairs():
    embeddings = torch.eye(3)
    labels = torch.tensor([0, 1, 2])
    loss = supervised_contrastive_loss(embeddings, labels)
    assert torch.isfinite(loss)
    assert loss.item() == 0.0
