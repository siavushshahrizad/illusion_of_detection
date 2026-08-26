import torch
import pytest
import numpy as np
from torch.nn.utils.rnn import PackedSequence
from loaders_torch import (
    CustomDataset,
    create_dataloader,
    custom_pad_collate
)


class TestClass:
    @pytest.fixture
    def setup_data(self):
        X = np.array([[1], [2], [3]])
        y = np.array([0, 0, 1])
        return X, y

    def test_custom_dataset(self, setup_data):
        data = CustomDataset(*setup_data)
        x_result, y_result = data[0]
        exp_x = torch.tensor([1])
        exp_y = 0
        assert x_result == exp_x
        assert y_result == exp_y
        assert x_result.device.type == "cpu"
        assert y_result.device.type == "cpu"

    def test_custom_dataset_on_gpu(self, setup_data):
        if not torch.backends.mps.is_available():
            pytest.skip("Unsupported GPU")
            
        device = torch.device("mps")

        data = CustomDataset(*setup_data, device="GPU")
        x_result, y_result = data[0]
        exp_x = torch.tensor([1], device=device)
        exp_y = 0
        assert x_result == exp_x
        assert y_result == exp_y

        # Assumes MAC
        assert x_result.device.type == "mps"
        assert y_result.device.type == "mps"

    def test_create_dataloader_with_batch_1(self, setup_data):
        loader = create_dataloader(
            *setup_data,
            batch_size=1,
            shuffle=False
        )

        iterator = iter(loader)
        x_result, y_result = next(iterator)

        assert isinstance(x_result, torch.Tensor)
        assert isinstance(y_result, torch.Tensor)
        exp_x = torch.tensor([1])
        exp_y = 0
        assert x_result == exp_x
        assert y_result == exp_y

    def test_custom_pad_collate(self):
        batch = [
            (torch.tensor([1]), torch.tensor(0)),
            (torch.tensor([3]), torch.tensor(0)),
            (torch.tensor([4]), torch.tensor(1)),
            (torch.tensor([3]), torch.tensor(1)),
        ]

        packed_seq, labels = custom_pad_collate(batch)
        assert isinstance(packed_seq, PackedSequence)
        assert isinstance(labels, torch.Tensor)

    def test_create_dataloader_with_batch_2(self, setup_data):
        loader = create_dataloader(
            *setup_data,
            batch_size=2,
            shuffle=False
        )

        iterator = iter(loader)
        x_result, y_result = next(iterator)
        exp_x = torch.tensor([1, 2])
        exp_y = torch.tensor([0, 0])
        assert isinstance(x_result, PackedSequence)
        assert torch.equal(x_result.data, exp_x)
        assert torch.equal(y_result, exp_y)
