import os
import numpy as np
import pandas as pd
from random import shuffle
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import preprocess_input, InceptionV3
from tensorflow.keras.applications.xception import decode_predictions,Xception
from keras.applications.resnet import ResNet50
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from tensorflow.keras import optimizers
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from collections import Counter
import tensorflow as tf
import matplotlib.pyplot as plt


os.getcwd()


class craiglistClassifier:

    def __init__(self, learning_rate=0.001, dropout=0.5,
                 file_path="weights.best.hdf5", epoch_val=10):
        self.image_folder = os.path.join(os.getcwd(), 'AUD_Image_Folder' + os.sep)
        self.rows = 224
        self.cols = 224
        self.batch_size = 16
        self.other_threshold = 0.8
        self.train_folder = 'train_images/'
        self.val_folder = 'validation_images/'
        self.class_mode = 'categorical'
        self.learning_rate = learning_rate
        self.epochs = epoch_val
        self.dropout = dropout
        self.file_path = file_path
        self.patience = 2

    @staticmethod
    def get_image_paths(image_folder):
        list_paths = []
        for subdir, dirs, files in os.walk(image_folder):
            for file in files:
                filepath = subdir + os.sep + file
                list_paths.append(filepath)
        return list_paths

    @staticmethod
    def get_train_val_image_paths(image_paths):
        list_train = [filepath for filepath in image_paths if "train" in filepath]
        list_val = [filepath for filepath in image_paths if "validation" in filepath]
        print(f"Train shape is: {len(list_train)}", '\n', f"Val shape is: {len(list_val)}")
        print(Counter([os.path.dirname(filepath).split(os.sep)[-1] for filepath in list_train]))
        return list_train, list_val

    def get_train_val_image_gen(self):
        train_idg = ImageDataGenerator(
            vertical_flip=True,
            horizontal_flip=True,
            height_shift_range=0.1,
            width_shift_range=0.1,
            rotation_range=5,
            preprocessing_function=preprocess_input
        )
        self.train_gen = train_idg.flow_from_directory(
            os.path.join(self.image_folder + self.train_folder),
            target_size=(self.rows, self.cols),
            batch_size=self.batch_size,
            shuffle=True,
            class_mode=self.class_mode
        )

        val_idg = ImageDataGenerator(preprocessing_function=preprocess_input)
        self.val_gen = val_idg.flow_from_directory(
            os.path.join(self.image_folder + self.val_folder),
            target_size=(self.rows, self.cols),
            batch_size=self.batch_size,
            shuffle=False,
            class_mode=self.class_mode,
        )

    def classification_model(self):
        input_shape = (self.rows, self.cols, 3)
        nclass = len(self.train_gen.class_indices)

        base_model = InceptionV3(
            weights='imagenet',
            include_top=False,
            input_shape=(self.rows, self.cols, 3)
        )
        # base_model = Xception(
        #                         weights='imagenet',
        #                         include_top=False,
        #                         input_shape=(self.rows, self.cols, 3)
        #                         )
        # base_model = ResNet50(
        #                         weights='imagenet',
        #                         include_top=False,
        #                         input_shape=(self.rows, self.cols, 3)
        #                         )
        base_model.trainable = False

        self.model = Sequential()
        self.model.add(base_model)
        self.model.add(GlobalAveragePooling2D())
        self.model.add(Dropout(self.dropout))
        self.model.add(Dense(nclass,
                             activation='softmax'  # softmax
                             ))

        self.model.compile(loss='categorical_crossentropy',
                           optimizer=optimizers.Adam(learning_rate=self.learning_rate),
                           metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
        self.model.summary()

    def predict_images(self):
        predicts_values = self.model.predict(self.val_gen,
                                             verbose=True, workers=2)
        predicts = np.argmax(predicts_values, axis=1)
        label_index = {v: k for k, v in self.train_gen.class_indices.items()}
        predict_label = [label_index[p] for p in predicts]
        df = pd.DataFrame(columns=['fname', 'applicance_type'])
        df['fname'] = [os.path.basename(x) for x in self.val_gen.filenames]
        df['applicance_type'] = predict_label
        return df, predicts_values

    def predict_test_images(self):
        test_idg = ImageDataGenerator(preprocessing_function=preprocess_input)
        test_gen = test_idg.flow_from_directory(
            os.path.join(self.image_folder),
            target_size=(self.rows, self.cols),
            batch_size=self.batch_size,
            shuffle=False,
            class_mode=None,
            classes=['test_images']
        )
        predicts_values = self.model.predict(test_gen,
                                             verbose=True, workers=2)
        predicts_idx = np.argmax(predicts_values, axis=1)
        predicts_val = np.amax(predicts_values, axis=1)
        label_index = {v: k for k, v in myclass.train_gen.class_indices.items()}
        predict_label = [label_index[p] for p in predicts_idx]
        df = pd.DataFrame(columns=['fname', 'predicted_appliance_class', 'class_prob', 'modified_prediction'])
        df['fname'] = [os.path.basename(x) for x in test_gen.filenames]
        df['predicted_appliance_class'] = predict_label
        df['class_prob'] = predicts_val
        df['modified_prediction'] = df['predicted_appliance_class']
        df.loc[df.class_prob < self.other_threshold, 'modified_prediction'] = "other"
        return df

    def loss_plot(self):
        epoch_stopped = self.early_stopping_monitor.stopped_epoch
        epoch_val = self.epochs if epoch_stopped == 0 else epoch_stopped
        loss_train = self.history.history['loss']
        loss_val = self.history.history['val_loss']
        epochs = [*range(1, epoch_val + 1)]
        plt.plot(epochs, loss_train, 'g', label='Training Loss')
        plt.plot(epochs, loss_val, 'b', label='Validation Loss')
        plt.title('Training and Validation Categorical crossentropy Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def main(self):
        checkpoint = ModelCheckpoint(self.file_path, monitor='val_accuracy',
                                     verbose=1, save_best_only=True, mode='max'
                                     )

        self.early_stopping_monitor = EarlyStopping(monitor="val_loss", mode="min", patience=self.patience)
        callbacks_list = [checkpoint, self.early_stopping_monitor]
        image_paths = self.get_image_paths(self.image_folder)
        list_train, list_val = self.get_train_val_image_paths(image_paths)

        self.get_train_val_image_gen()
        self.classification_model()
        self.history = self.model.fit(self.train_gen,
                                      epochs=self.epochs,
                                      shuffle=True,
                                      validation_data=self.val_gen,
                                      verbose=True,
                                      callbacks=callbacks_list
                                      )
        test_pred = self.predict_test_images()
        return test_pred

if __name__ == '__main__':
  myclass = craiglistClassifier(epoch_val=5)
  test_pred = myclass.main()
  plt.rcParams["figure.figsize"] = (10, 5)
  myclass.loss_plot()
  print(test_pred.head(20))


