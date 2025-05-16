from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from nn import NN


class CNN(NN):

    def __init__(self, channels, time_samples, param):

        self.model = Sequential()

        # Conv layer to learn spatial filters (over channels)
        self.model.add(Conv2D(filters=16, kernel_size=(channels, 1), activation='elu',
                              input_shape=(channels, time_samples, 1), padding='valid'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.3))

        # Conv layer to learn temporal filters
        self.model.add(Conv2D(filters=32, kernel_size=(1, 8), activation='elu', padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(1, 4)))
        self.model.add(Dropout(0.3))

        self.model.add(Flatten())
        self.model.add(Dense(64, activation='elu'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(2, activation='softmax')) 

        self.param = param
        self.compile()
