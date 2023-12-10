from keras import layers, models

def conv1d_3_block():
    block = models.Sequential([
        layers.Conv1D(filters=64, kernel_size=3, padding='same'),
        layers.Conv1D(filters=64, kernel_size=3, padding='same'),
        layers.Conv1D(filters=64, kernel_size=3, padding='same'),
        layers.ReLU(),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2, strides=2, padding='same')
    ])
    return block

def ConvLSTM_VGG():
    model = models.Sequential([
        # The Conv1D-3 Block is repeated three times
        conv1d_3_block(),
        conv1d_3_block(),
        conv1d_3_block(),
        # Followed by two LSTM layers
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.5),
        layers.LSTM(64),
        layers.Dropout(0.5),
        
        # Flatten the output to feed into the Dense layers
        layers.Flatten(),
        # Two Dense layers with ReLU activation
        layers.Dense(32, activation='relu'),
        layers.Dense(2, activation='relu'),
        layers.Softmax()
    ], name='ConvLSTM_VGG')
    return model
