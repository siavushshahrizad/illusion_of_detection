import pytest
import numpy as np
from lstm import LSTMModel
from loaders_torch import create_dataloader

class TestClass:
    @pytest.fixture
    def setup_data(self):
        X = np.array([
            [[1], [2], [3]],
            [[3], [2], [4]],
            [[14], [1], [3]]

        ])
        y = np.array([0, 0, 1])
        return X, y

    def test_train_lstm(self, setup_data):
        loader = create_dataloader(
            *setup_data,
            batch_size=1,
            shuffle=False
        )

        lstm = LSTMModel(input_size=1)
        result = lstm.model_train(loader)
        assert isinstance(result, dict)
        assert isinstance(result["loss"], float)
        assert isinstance(result["labels"], list)
        assert isinstance(result["preds"], list)

        assert len(result["preds"]) == len(loader)
        assert len(result["labels"]) == len(loader)

    def test_train_lstm_batch_2(self, setup_data):
        loader = create_dataloader(
            *setup_data,
            batch_size=2,
            shuffle=False
        )

        lstm = LSTMModel(input_size=1)
        result = lstm.model_train(loader)
        assert isinstance(result, dict)

    def test_overfit_3_samples(self, setup_data):
        loader = create_dataloader(
            *setup_data,
            batch_size=1,
            shuffle=False
        )

        lstm = LSTMModel(input_size=1, learning_rate=1e-2)
        result = lstm.model_train(loader)
        initial_loss = result["loss"]

        for _ in range(500):
            result = lstm.model_train(loader)

        final_loss = result["loss"]
        assert final_loss < initial_loss
        assert final_loss < initial_loss * 0.1  # Less than a tenth of initial loss
