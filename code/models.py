import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
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

    def __init__(self, input_size, hidden_size, num_layers, drop_out):
        super(HSI_lstm, self).__init__()  
        self.rnn = nn.LSTM(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first= True,
            dropout = drop_out)
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
            batch_first= True,
            dropout = drop_out)
        self.output = nn.Linear(hidden_size,1)

    def forward(self, input):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size) 
        input = input.float()
        r_out, h_n = self.rnn(input)
        return self.output(r_out[:,-1,:])


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_size, z_size):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc21 = nn.Linear(hidden_size, z_size)
        self.fc22 = nn.Linear(hidden_size, z_size)
        self.fc3 = nn.Linear(z_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, input_dim)

    def encode(self, x):
        x = x.float()
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar