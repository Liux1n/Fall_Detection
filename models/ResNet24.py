import keras
from keras import layers

def ResNet24(axis_num=9):
    inputs = keras.Input(shape=(50, axis_num))
    x = layers.Reshape((1, 50, axis_num))(inputs)
    x = layers.Conv2D(filters=64, kernel_size=(1, 3))(x)
    x = layers.Conv2D(filters=64, kernel_size=(1, 3))(x)
    x = layers.MaxPooling2D(pool_size=(1, 2))(x)

    # ConvBlock
    residual = layers.Conv2D(filters=64, kernel_size=(1, 1))(x)
    x = layers.Conv2D(filters=16, kernel_size=(1, 1))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters=16, kernel_size=(1, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters=64, kernel_size=(1, 1))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, residual])
    x = layers.ReLU()(x)

    # ConvBlock
    residual = layers.Conv2D(filters=64, kernel_size=(1, 1))(x)
    x = layers.Conv2D(filters=16, kernel_size=(1, 1))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters=16, kernel_size=(1, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters=64, kernel_size=(1, 1))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, residual])
    x = layers.ReLU()(x)

    # IdentityBlock
    residual = x
    x = layers.Conv2D(filters=16, kernel_size=(1, 1))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters=16, kernel_size=(1, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters=64, kernel_size=(1, 1))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, residual])
    x = layers.ReLU()(x)

    # ConvBlock
    residual = layers.Conv2D(filters=64, kernel_size=(1, 1))(x)
    x = layers.Conv2D(filters=16, kernel_size=(1, 1))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters=16, kernel_size=(1, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters=64, kernel_size=(1, 1))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, residual])
    x = layers.ReLU()(x)

    # IdentityBlock
    residual = x
    x = layers.Conv2D(filters=16, kernel_size=(1, 1))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters=16, kernel_size=(1, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters=64, kernel_size=(1, 1))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, residual])
    x = layers.ReLU()(x)

    # ConvBlock
    residual = layers.Conv2D(filters=64, kernel_size=(1, 1))(x)
    x = layers.Conv2D(filters=16, kernel_size=(1, 1))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters=16, kernel_size=(1, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters=64, kernel_size=(1, 1))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, residual])
    x = layers.ReLU()(x)

    # IdentityBlock
    residual = x
    x = layers.Conv2D(filters=16, kernel_size=(1, 1))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters=16, kernel_size=(1, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters=64, kernel_size=(1, 1))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, residual])
    x = layers.ReLU()(x)

    x = layers.AveragePooling2D(pool_size=(1, 2))(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(2, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="ResNet24")
    return model
