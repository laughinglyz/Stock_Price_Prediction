import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

 
class HSI_lstm(nn.Module):
    """
    This is the class that creates a LSTM network for stock price prediction.
    
    Network architecture:
    - A one-layer LSTM
    - Output layer: a neuron fully connected to the output of LSTM

    Inputs: 
    input_size, hidden_size, num_layers

    """

    def __init__(self, input_size, hidden_size, num_layers):
        super(HSI_lstm, self).__init__()  
        self.rnn = nn.LSTM(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first= True)
        self.output = nn.Linear(hidden_size,1)

    def forward(self, input):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)  
        # h_c shape (n_layers, batch, hidden_size)
        input = input.float()
        r_out, (h_n, h_c) = self.rnn(input)
        return self.output(r_out[:,-1,:])

class HSI_gru(nn.Module):
    """
    This is the class that creates a GRU network for stock price prediction.
    
    Network architecture:
    - A one-layer GRU
    - Output layer: a neuron fully connected to the output of GRU

    Inputs: 
    input_size, hidden_size, num_layers

    """

    def __init__(self, input_size, hidden_size, num_layers):
        super(HSI_gru, self).__init__()
        self.rnn = nn.GRU(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first= True)
        self.output = nn.Linear(hidden_size,1)

    def forward(self, input):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size) 
        input = input.float()
        r_out, h_n = self.rnn(input)
        return self.output(r_out[:,-1,:])
