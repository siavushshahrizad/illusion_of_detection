"""
Implements the LSTM class.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from constants import (
    INPUT_SIZE,
    HIDDEN_SIZE,
    NUM_LAYERS_RNN,
    OUTPUT_DIM,
    LEARNING_RATE,
)


class LSTMModel(nn.Module):
    def __init__(
            self,
            input_size: int = INPUT_SIZE,
            hidden_size: int = HIDDEN_SIZE,
            output_size: int = OUTPUT_DIM,
            num_layers: int = NUM_LAYERS_RNN,
            bidirectional: bool = False,
            learning_rate = LEARNING_RATE,
            weights: torch.Tensor | None = None,
            device: str = "CPU"
        ):
        super(LSTMModel, self).__init__() 
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._output_size = output_size
        self._num_layers = num_layers
        self._bidirectional = bidirectional
        self._num_directions: int = 2 if self._bidirectional else 1
        self._lr = learning_rate 
        self._weights = weights
        
        # Move to right device
        if device not in ["GPU", "CPU"]:
            raise ValueError("Device must be 'GPU' or 'CPU'.")
        if device == "GPU":
            self._device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        else:
            self._device = torch.device('cpu')

        # Important to turn on batch_first as X coming in as
        # B x T x C, with B = batch, T = timestep, and C = coefficient
        self._lstm = nn.LSTM(
            self._input_size, 
            self._hidden_size, 
            self._num_layers, 
            batch_first=True, 
            bidirectional=self._bidirectional
        )
        self._linear = nn.Linear(self._hidden_size * self._num_directions, self._output_size)
        self._lstm = self._lstm.to(self._device)
        self._linear = self._linear.to(self._device)
    
        if self._weights is not None:
            self._weights = self._weights.to(self._device)

        self._softmax = nn.Softmax(dim=-1)           # Last dim which is the classes, so get probs across samples
        self._optimiser = torch.optim.Adam(self.parameters(), lr=self._lr)
        self._loss_fn = nn.CrossEntropyLoss(weight=self._weights, reduction="mean")       # Weights for class imbalance

    def forward(self, inputs, hidden=None, logits_only=False) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Assumption is hidden and cell are none and let the LSTM
        # API deal with that
        _, (final_hidden, _) = self._lstm(inputs, hidden)
        logits = self._linear(final_hidden[-1])    # Get the last RNN layers hidden state (batch, hidden_dim)
        
        # Branching logic so predict method avoids unneeded computation 
        if logits_only:
            return logits, None

        probs = self._softmax(logits)                
        preds = torch.argmax(probs, dim=-1)     # Gets highest probability class across samples
        return logits, preds

    def model_train(self, data: DataLoader) -> dict:
        labels = []
        preds = []
        avg_loss = 0    

        for X, y in data:
            logits, pred = self.forward(X)
            loss = self._loss_fn(logits, y)
            avg_loss += loss.item()

            preds.extend(pred.cpu().tolist())
            labels.extend(y.cpu().tolist())

            loss.backward()
            self._optimiser.step()
            self._optimiser.zero_grad()

        results = {
            "loss": avg_loss / len(data),
            "labels": labels,
            "preds": preds
        }
        return results

    def model_predict(self, data: DataLoader) -> dict:
        labels = []
        preds = []
        avg_loss = 0    

        with torch.no_grad():
            for X, y in data:
                logits, pred = self.forward(X)
                preds.extend(pred.cpu().tolist())
                labels.extend(y.cpu().tolist())
                loss = self._loss_fn(logits, y)
                avg_loss += loss.item()

        results = {
            "loss": avg_loss / len(data),
            "labels": labels,
            "preds": preds
        }
        return results
