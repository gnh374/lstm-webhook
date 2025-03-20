import torch
from torch import nn

class Predictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super().__init__()
        
        self.lstm_1 = nn.LSTM(
            input_size=input_size,
            hidden_size=64,
            batch_first=True,
            dropout=0.2,
            num_layers=2,
            bidirectional=True,            
        )
        self.lstm_2 = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,  # Note: This is the size for each direction
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        
        # Fully connected layers
        # The LSTM output size is doubled due to bidirectionality
        self.fc = nn.Sequential(  
            nn.Dropout(),
            nn.Linear(in_features=hidden_size*2, out_features=64),
            nn.LeakyReLU(),
            nn.Linear(in_features=64, out_features=output_size), 
        )
        
    def forward(self, x):
        # Pass input through the LSTM
        out, _ = self.lstm_1(x)  # out shape: (batch_size, seq_len, hidden_size * 2)
        
        output = out[:, -1, :]  
        
        output = output.unsqueeze(1)
        
        out, _ = self.lstm_2(output)  # out shape: (batch_size, seq_len, hidden_size * 2)
        
        output = out[:, -1, :]  
        
        output = self.fc(output)
        
        return output
    
    