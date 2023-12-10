from keras import layers, models

def ConvLSTM():
    model = models.Sequential([
        # Conv1
        layers.Conv2D(filters=64, kernel_size=(1, 7), padding='same', input_shape=(1, 50, 9)),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(pool_size=(1, 2)),

        # Conv2
        layers.Conv2D(filters=64, kernel_size=(1, 7), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(pool_size=(1, 2)),

        # Conv3
        layers.Conv2D(filters=64, kernel_size=(1, 7), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(pool_size=(1, 2)),

        # LSTM layers
        layers.Reshape((6, 64)),
        # quantize_annotate_layer(CustomLSTM(64, return_sequences=True), DefaultLSTMQuantizeConfig()),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.5),
        # quantize_annotate_layer(CustomLSTM(64), DefaultLSTMQuantizeConfig()),
        layers.LSTM(64),
        layers.Dropout(0.5),
        
        # Fully connected layer
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dense(2, activation='relu'),
        
        # softmax
        layers.Softmax()
    ], name='ConvLSTM')
    return model
