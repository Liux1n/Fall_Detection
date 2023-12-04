from torch import nn

class ConvLSTM(nn.Module):
    def __init__(self):
        super(ConvLSTM, self).__init__()
        
        self.conv1 = nn.Sequential(nn.Conv1d(in_channels=9, out_channels=64, kernel_size=3, padding=1),
                                   nn.BatchNorm1d(64),
                                   nn.ReLU(),
                                   nn.MaxPool1d(2))
        self.conv2 = nn.Sequential(nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                                   nn.BatchNorm1d(64),
                                   nn.ReLU(),
                                   nn.MaxPool1d(2))
        self.conv3 = nn.Sequential(nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                                   nn.BatchNorm1d(64),
                                   nn.ReLU(),
                                   nn.MaxPool1d(2))

        # LSTM layers
        self.lstm1 = nn.LSTM(input_size=64, hidden_size=64, batch_first=True)
        self.dropout1 = nn.Dropout(0.5)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=64, batch_first=True)
        self.dropout2 = nn.Dropout(0.5)

        self.fc = nn.Linear(64, 2)  # No need for softmax here when using nn.CrossEntropyLoss
    
    def forward(self, x):
        # Convolutional layers
        x = x.transpose(1, 2)  # Transpose to have the correct dimensions for Conv1d (batch, channels, length)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Prepare for LSTM
        x = x.transpose(1, 2)  # Transpose back to (batch, seq_len, features)
        
        # LSTM layers
        x, _ = self.lstm1(x)  # Only take the output, ignore hidden states
        x = self.dropout1(x)
        x, _ = self.lstm2(x)  # Only take the output, ignore hidden states
        x = self.dropout2(x)
        
        # Take the outputs of the last time step
        x = x[:, -1, :]
        
        # Fully connected layer
        x = self.fc(x)
        
        return x