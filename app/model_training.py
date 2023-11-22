import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from app import app
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Constants
test_data_path = app.config['STATIC_PATH'] / 'data/emnist-letters-test.csv'
train_data_path = app.config['STATIC_PATH'] / 'data/emnist-letters-train.csv'
class_mapping = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt'
num_classes = 47
img_size = 28
epochs = 15


def img_label_load(data_path, num_classes):
    data = pd.read_csv(data_path, header=None)
    data_rows = len(data)

    # Reshape the data
    imgs = data.iloc[:, 1:].values.reshape(data_rows, img_size, img_size, 1)
    labels = keras.utils.to_categorical(data.iloc[:, 0], num_classes)

    return imgs / 255.0, labels


def create_model(input_shape, num_classes):
    # model = Sequential()
    # model.add(Conv2D(filters=12, kernel_size=(5, 5), strides=2, activation='relu', input_shape=input_shape))
    # model.add(Dropout(0.5))
    # model.add(Conv2D(filters=18, kernel_size=(3, 3), strides=2, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Conv2D(filters=24, kernel_size=(2, 2), activation='relu'))
    # model.add(Flatten())
    # model.add(Dense(units=150, activation='relu'))
    # model.add(Dense(units=num_classes, activation='softmax'))
    # optimizer = Adam()
    # model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_model(model, training_data_generator, validation_data_generator, epochs):
    best_model = app.config["STATIC_PATH"] / 'Best_points.h5'
    MCP = ModelCheckpoint(str(best_model), verbose=1, save_best_only=True, monitor='val_accuracy', mode='max')
    ES = EarlyStopping(monitor='val_accuracy', min_delta=0, verbose=0, restore_best_weights=True, patience=3,
                       mode='max')
    RLP = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.2, min_lr=0.0001)
    history = model.fit_generator(training_data_generator,
                                  epochs=epochs,
                                  validation_data=validation_data_generator,
                                  callbacks=[MCP, ES, RLP])
    model.save(app.config['STATIC_PATH'] / 'mnist.h5')
    return history


def evaluate_model(model, test_data_generator):
    return model.evaluate_generator(test_data_generator)


def run_prediction(model, X_test, test_data, idx, class_mapping):
    result = np.argmax(model.predict(X_test[idx:idx + 1]))
    print('Prediction: ', result, ', Char: ', class_mapping[result])
    print('Label: ', test_data.values[idx, 0])


def main():
    X, y = img_label_load(train_data_path, num_classes)
    input_shape = (img_size, img_size, 1)
    model = create_model(input_shape, num_classes)

    data_generator = keras.preprocessing.image.ImageDataGenerator(validation_split=0.2)
    training_data_generator = data_generator.flow(X, y, subset='training')
    validation_data_generator = data_generator.flow(X, y, subset='validation')

    history = train_model(model, training_data_generator, validation_data_generator, epochs=epochs)

    test_X, test_y = img_label_load(test_data_path, num_classes)
    test_data_generator = data_generator.flow(test_X, test_y)

    evaluation_result = evaluate_model(model, test_data_generator)
    print("Evaluation Result:", evaluation_result)

    for _ in range(1, 10):
        idx = np.random.randint(0, 47 - 1)
        run_prediction(model, test_X, pd.read_csv(test_data_path, header=None), idx, class_mapping)

    # from app.gui import predict_digit
    # from PIL import Image
    #
    # pillow_img_obj = Image.open(
    #     "/home/jaydeep/Desktop/college/term1/test_project/digit_recognition/app/static/images/test_d.png")
    # digit, acc = predict_digit(pillow_img_obj)
    # print(digit, acc)


if __name__ == "__main__":
    main()
