"""
test_loader.py – Unit tests for LFWDataset, FewShotEpisodicSampler, and collate_episode.

All tests run fully offline: a ``fake_lfw`` fixture creates real tiny JPEG images in a
temporary directory and patches ``torchvision.datasets.LFWPeople`` so no network access
or dataset download is ever required.
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image

# ── Fixture ────────────────────────────────────────────────────────────────────

N_CLASSES = 10          # total unique people in the fake dataset
IMGS_PER_CLASS = 8      # images per person (must be ≥ k_shot + n_query used in tests)
IMG_SIZE = (112, 112)


def _make_fake_lfw_people(tmp_path: Path) -> MagicMock:
    """Create tiny real JPEG files and return a MagicMock mimicking LFWPeople."""
    image_paths: list[str] = []
    targets: list[int] = []

    for cls_idx in range(N_CLASSES):
        for img_idx in range(IMGS_PER_CLASS):
            img_file = tmp_path / f"cls{cls_idx:02d}_img{img_idx:02d}.jpg"
            # Each class gets a distinct colour so transforms don't collapse them
            colour = (cls_idx * 20 % 255, img_idx * 15 % 255, 100)
            Image.new("RGB", IMG_SIZE, color=colour).save(img_file, format="JPEG")
            image_paths.append(str(img_file))
            targets.append(cls_idx)

    mock = MagicMock()
    mock.targets = targets
    mock.classes = [f"Person_{i:02d}" for i in range(N_CLASSES)]
    mock.data = image_paths
    return mock


@pytest.fixture()
def fake_people(tmp_path: Path) -> MagicMock:
    """A MagicMock LFWPeople backed by real JPEG files."""
    return _make_fake_lfw_people(tmp_path)


@pytest.fixture()
def dataset(fake_people: MagicMock) -> Generator:
    """LFWDataset wrapping the fake LFWPeople (no network)."""
    with patch("src.data.lfw_loader.LFWPeople", return_value=fake_people):
        from src.data.lfw_loader import LFWDataset

        yield LFWDataset(root="/fake/root", download=False)


# ── LFWDataset ────────────────────────────────────────────────────────────────


class TestLFWDataset:
    """Tests for :class:`src.data.lfw_loader.LFWDataset`."""

    def test_len(self, dataset):
        """Total samples = N_CLASSES * IMGS_PER_CLASS when no filtering applied."""
        assert len(dataset) == N_CLASSES * IMGS_PER_CLASS

    def test_num_classes(self, dataset):
        assert dataset.num_classes == N_CLASSES

    def test_classes_list(self, dataset):
        assert len(dataset.classes) == N_CLASSES
        assert dataset.classes[0].startswith("Person_")

    def test_class_to_idx(self, dataset):
        assert len(dataset.class_to_idx) == N_CLASSES
        for name, idx in dataset.class_to_idx.items():
            assert dataset.classes[idx] == name

    def test_targets_length(self, dataset):
        assert len(dataset.targets) == len(dataset)

    def test_targets_range(self, dataset):
        assert all(0 <= t < dataset.num_classes for t in dataset.targets)

    def test_targets_contiguous(self, dataset):
        """Labels must be remapped to a contiguous range [0, num_classes)."""
        assert sorted(set(dataset.targets)) == list(range(dataset.num_classes))

    def test_getitem_returns_three_tuple(self, dataset):
        result = dataset[0]
        assert len(result) == 3, "Expected (image, label, person_name)"

    def test_getitem_image_is_tensor(self, dataset):
        img, _, _ = dataset[0]
        assert isinstance(img, torch.Tensor)

    def test_getitem_image_shape(self, dataset):
        img, _, _ = dataset[0]
        # Default transform → (3, 112, 112)
        assert img.shape == (3, 112, 112)

    def test_getitem_label_is_int(self, dataset):
        _, label, _ = dataset[0]
        assert isinstance(label, int)

    def test_getitem_label_in_range(self, dataset):
        _, label, _ = dataset[0]
        assert 0 <= label < dataset.num_classes

    def test_getitem_person_name_is_str(self, dataset):
        _, _, name = dataset[0]
        assert isinstance(name, str)

    def test_getitem_name_consistent_with_label(self, dataset):
        _, label, name = dataset[0]
        assert dataset.classes[label] == name

    def test_min_images_filter(self, tmp_path: Path):
        """Classes with fewer images than min_images_per_class are excluded."""
        mock = _make_fake_lfw_people(tmp_path)
        # Artificially add a rare class (just 1 image)
        rare_img = tmp_path / "rare.jpg"
        Image.new("RGB", IMG_SIZE, color=(200, 0, 0)).save(rare_img, format="JPEG")
        mock.data.append(str(rare_img))
        mock.targets.append(N_CLASSES)   # new class index
        mock.classes.append("Rare_Person")

        with patch("src.data.lfw_loader.LFWPeople", return_value=mock):
            from src.data.lfw_loader import LFWDataset

            ds_no_filter = LFWDataset(root="/fake", min_images_per_class=1)
            ds_filtered = LFWDataset(root="/fake", min_images_per_class=2)

        assert ds_no_filter.num_classes == N_CLASSES + 1
        assert ds_filtered.num_classes == N_CLASSES
        assert "Rare_Person" not in ds_filtered.classes

    def test_custom_transform(self, fake_people: MagicMock):
        """A custom transform (identity resize) is applied correctly."""
        from torchvision import transforms
        from src.data.lfw_loader import _pil_to_tensor

        custom_tf = transforms.Compose(
            [
                transforms.Resize((64, 64)),
                transforms.Lambda(_pil_to_tensor),
            ]
        )
        with patch("src.data.lfw_loader.LFWPeople", return_value=fake_people):
            from src.data.lfw_loader import LFWDataset

            ds = LFWDataset(root="/fake", transform=custom_tf)
        img, _, _ = ds[0]
        assert img.shape == (3, 64, 64)


# ── FewShotEpisodicSampler ────────────────────────────────────────────────────


class TestFewShotEpisodicSampler:
    """Tests for :class:`src.data.lfw_loader.FewShotEpisodicSampler`."""

    _N_WAY = 3
    _K_SHOT = 2
    _N_QUERY = 3
    _N_EPISODES = 10

    @pytest.fixture()
    def sampler(self, dataset):
        from src.data.lfw_loader import FewShotEpisodicSampler

        return FewShotEpisodicSampler(
            dataset=dataset,
            n_way=self._N_WAY,
            k_shot=self._K_SHOT,
            n_query=self._N_QUERY,
            num_episodes=self._N_EPISODES,
            class_split=None,   # use all classes
            seed=42,
        )

    def test_len(self, sampler):
        assert len(sampler) == self._N_EPISODES

    def test_episode_size_property(self, sampler):
        expected = self._N_WAY * (self._K_SHOT + self._N_QUERY)
        assert sampler.episode_size == expected

    def test_yields_correct_number_of_episodes(self, sampler):
        episodes = list(sampler)
        assert len(episodes) == self._N_EPISODES

    def test_each_episode_has_correct_size(self, sampler):
        expected = self._N_WAY * (self._K_SHOT + self._N_QUERY)
        for ep in sampler:
            assert len(ep) == expected

    def test_episode_indices_are_valid(self, sampler, dataset):
        n = len(dataset)
        for ep in sampler:
            assert all(0 <= idx < n for idx in ep)

    def test_support_and_query_no_overlap(self, sampler):
        """Within a single episode, support and query indices must not overlap."""
        n_support = self._N_WAY * self._K_SHOT
        for ep in sampler:
            support_set = set(ep[:n_support])
            query_set = set(ep[n_support:])
            assert support_set.isdisjoint(query_set)

    def test_reproducibility_same_seed(self, dataset):
        """Two samplers with the same seed must yield identical episode sequences."""
        from src.data.lfw_loader import FewShotEpisodicSampler

        s1 = FewShotEpisodicSampler(dataset, n_way=3, k_shot=2, n_query=2,
                                     num_episodes=5, class_split=None, seed=99)
        s2 = FewShotEpisodicSampler(dataset, n_way=3, k_shot=2, n_query=2,
                                     num_episodes=5, class_split=None, seed=99)
        assert list(s1) == list(s2)

    def test_different_seeds_differ(self, dataset):
        """Different seeds should (very likely) produce different first episodes."""
        from src.data.lfw_loader import FewShotEpisodicSampler

        s1 = FewShotEpisodicSampler(dataset, n_way=3, k_shot=2, n_query=2,
                                     num_episodes=1, class_split=None, seed=1)
        s2 = FewShotEpisodicSampler(dataset, n_way=3, k_shot=2, n_query=2,
                                     num_episodes=1, class_split=None, seed=2)
        assert list(s1)[0] != list(s2)[0]

    def test_split_sizes_populated(self, dataset):
        from src.data.lfw_loader import FewShotEpisodicSampler

        s = FewShotEpisodicSampler(dataset, n_way=2, k_shot=1, n_query=2,
                                    num_episodes=1, class_split="train")
        assert "train" in s.split_sizes
        assert "val" in s.split_sizes
        assert "test" in s.split_sizes
        assert sum(s.split_sizes.values()) == N_CLASSES

    def test_class_splits_are_disjoint(self, dataset):
        """Train / val / test class pools must be mutually exclusive."""
        from src.data.lfw_loader import FewShotEpisodicSampler

        # Use n_way=1 so the sampler can always be constructed regardless of
        # how many classes end up in each split with the small 10-class fixture.
        def pool(split):
            return set(
                FewShotEpisodicSampler(
                    dataset, n_way=1, k_shot=1, n_query=1,
                    num_episodes=1, class_split=split, seed=7,
                )._class_pool
            )

        train_p = pool("train")
        val_p = pool("val")
        test_p = pool("test")
        assert train_p.isdisjoint(val_p)
        assert train_p.isdisjoint(test_p)
        assert val_p.isdisjoint(test_p)

    def test_invalid_class_split_raises(self, dataset):
        from src.data.lfw_loader import FewShotEpisodicSampler

        with pytest.raises(ValueError, match="class_split must be"):
            FewShotEpisodicSampler(dataset, n_way=2, k_shot=1, n_query=1,
                                    class_split="bogus")

    def test_too_few_classes_raises(self, dataset):
        """Requesting more ways than eligible classes must raise ValueError."""
        from src.data.lfw_loader import FewShotEpisodicSampler

        with pytest.raises(ValueError):
            FewShotEpisodicSampler(dataset, n_way=N_CLASSES + 10,
                                    k_shot=1, n_query=1, class_split=None)


# ── collate_episode ────────────────────────────────────────────────────────────


class TestCollateEpisode:
    """Tests for :func:`src.data.lfw_loader.collate_episode`."""

    _N = 3   # ways
    _K = 2   # shots
    _Q = 4   # queries
    _C, _H, _W = 3, 112, 112

    def _make_batch(self) -> list[tuple[torch.Tensor, int, str]]:
        """Simulate the flat batch the sampler+DataLoader would produce.

        Support block: N*K items, query block: N*Q items.
        Labels are contiguous per class within each block.
        """
        batch = []
        # Support block (N classes × K shots, class-contiguous)
        for cls in range(self._N):
            for _ in range(self._K):
                img = torch.zeros(self._C, self._H, self._W)
                batch.append((img, cls, f"Person_{cls:02d}"))
        # Query block (N classes × Q queries, class-contiguous)
        for cls in range(self._N):
            for _ in range(self._Q):
                img = torch.ones(self._C, self._H, self._W)
                batch.append((img, cls, f"Person_{cls:02d}"))
        return batch

    def test_support_shape(self):
        from src.data.lfw_loader import collate_episode

        ep = collate_episode(self._make_batch())
        assert ep.support.shape == (self._N * self._K, self._C, self._H, self._W)

    def test_query_shape(self):
        from src.data.lfw_loader import collate_episode

        ep = collate_episode(self._make_batch())
        assert ep.query.shape == (self._N * self._Q, self._C, self._H, self._W)

    def test_support_labels_shape(self):
        from src.data.lfw_loader import collate_episode

        ep = collate_episode(self._make_batch())
        assert ep.support_labels.shape == (self._N * self._K,)

    def test_query_labels_shape(self):
        from src.data.lfw_loader import collate_episode

        ep = collate_episode(self._make_batch())
        assert ep.query_labels.shape == (self._N * self._Q,)

    def test_local_labels_in_range(self):
        from src.data.lfw_loader import collate_episode

        ep = collate_episode(self._make_batch())
        assert ep.support_labels.min() >= 0
        assert ep.support_labels.max() < self._N
        assert ep.query_labels.min() >= 0
        assert ep.query_labels.max() < self._N

    def test_class_names_count(self):
        from src.data.lfw_loader import collate_episode

        ep = collate_episode(self._make_batch())
        assert len(ep.class_names) == self._N

    def test_class_names_are_strings(self):
        from src.data.lfw_loader import collate_episode

        ep = collate_episode(self._make_batch())
        assert all(isinstance(n, str) for n in ep.class_names)

    def test_is_episode_namedtuple(self):
        from src.data.lfw_loader import Episode, collate_episode

        ep = collate_episode(self._make_batch())
        assert isinstance(ep, Episode)


# ── End-to-end DataLoader integration ─────────────────────────────────────────


class TestEpisodicDataLoader:
    """Integration: DataLoader with FewShotEpisodicSampler + collate_episode."""

    def test_loader_yields_episodes(self, dataset):
        from torch.utils.data import DataLoader

        from src.data.lfw_loader import (
            Episode,
            FewShotEpisodicSampler,
            collate_episode,
        )

        sampler = FewShotEpisodicSampler(
            dataset, n_way=3, k_shot=2, n_query=2,
            num_episodes=4, class_split=None, seed=0,
        )
        loader = DataLoader(dataset, batch_sampler=sampler,
                            num_workers=0, collate_fn=collate_episode)

        episodes = list(loader)
        assert len(episodes) == 4
        for ep in episodes:
            assert isinstance(ep, Episode)

    def test_loader_support_query_shapes(self, dataset):
        from torch.utils.data import DataLoader

        from src.data.lfw_loader import FewShotEpisodicSampler, collate_episode

        N, K, Q = 3, 2, 3
        sampler = FewShotEpisodicSampler(
            dataset, n_way=N, k_shot=K, n_query=Q,
            num_episodes=2, class_split=None, seed=1,
        )
        loader = DataLoader(dataset, batch_sampler=sampler,
                            num_workers=0, collate_fn=collate_episode)
        ep = next(iter(loader))
        assert ep.support.shape[0] == N * K
        assert ep.query.shape[0] == N * Q
        assert ep.support.shape[1:] == (3, 112, 112)
