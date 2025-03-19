import math
import warnings
import numbers
import weakref
from typing import List, Tuple, Optional, overload

import torch
from torch import Tensor
from torch.nn import Linear, BatchNorm1d
from torch.nn import functional as F   


class liGRUCell(torch.nn.Module):
    """ 
        Light-Gated Recurrent Units (liGRU) cell.
    """

    def __init__(self, input_size, hidden_size, bias=False,
                 nonlinearity="relu", dropout=0.0, normalization = "batch-norm"):
        super(liGRUCell, self).__init__()
        self.hidden_size = int(hidden_size)
        self.input_size = int(input_size)
        self.dropout = dropout
        self.W = Linear(self.input_size, 2*self.hidden_size, bias=bias)
        self.U = Linear(self.hidden_size, 2*self.hidden_size, bias=bias)
        self.reset_parameters()

        if normalization == "batch-norm":
            self.norm = BatchNorm1d(2*self.hidden_size)
        else:
            self.norm = None

        if nonlinearity == "leaky-relu":
            self.nonlinearity = torch.nn.LeakyReLU()
        else:
            self.nonlinearity = torch.nn.ReLU()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
    
    def forward(self, x, hidden):
        
        x = x.view(-1, x.size(1))
        
        gate_x = self.W(x)
        gate_h = self.U(hidden)
        
        if self.norm is not None:
            gates = self.norm(gate_x.squeeze()) + gate_h.squeeze()
        else:
            gates = gate_x.squeeze() + gate_h.squeeze()

        activ_t, zt = gates.chunk(2, 1)
        zt = torch.sigmoid(zt)
        h_tilde = self.nonlinearity(activ_t)
        ht = zt * hidden + (1 - zt) * h_tilde

        return ht
    

class MGUCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MGUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Update gate
        self.W_z = Linear(input_size + hidden_size, hidden_size)
        # Candidate hidden state
        self.W_h = Linear(input_size + hidden_size, hidden_size)
        # Activation function
        self.activation = torch.nn.Tanh()

        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        """
        Forward pass of the MGU.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
            hidden (torch.Tensor): Previous hidden state of shape (batch_size, hidden_size).

        Returns:
            torch.Tensor: Updated hidden state of shape (batch_size, hidden_size).
        """
        # Concatenate input and previous hidden state
        combined = torch.cat((x, hidden), dim=1)

        # Compute update gate
        z = torch.sigmoid(self.W_z(combined))

        # Compute candidate hidden state
        h_tilde = self.activation(self.W_h(combined))

        # Compute new hidden state
        h_next = (1 - z) * hidden + z * h_tilde

        return h_next

  
