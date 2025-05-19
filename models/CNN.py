from sparkdl import DeepImageFeaturizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from pyspark.sql.types import StructType, StructField, IntegerType
from pyspark.ml.linalg import VectorUDT
import numpy as np

class DeepImageCNN:
    def __init__(self, num_classes=10):
        
        self.keras_model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        self.featurizer = DeepImageFeaturizer(
            inputCol="image",
            outputCol="features",
            modelName="CustomCNN"
        )
        self.schema = StructType([
            StructField("image", VectorUDT(), True),
            StructField("label", IntegerType(), True)
        ])

    def configure_model(self, configs):
        self.keras_model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return self.featurizer

    def featurize(self, df, raw_model, path):
        
        features_df = raw_model.transform(df)
        
        model, predictions, accuracy, loss, precision, recall, f1 = self.train_keras(features_df)
        return model, predictions, accuracy, loss, precision, recall, f1

    def train_keras(self, df):
        return self.keras_model, [], 0.0, 0.0, 0.0, 0.0, 0.0

    def predict(self, df, raw_model, path=None):
        features_df = raw_model.transform(df)
        return [], 0.0, 0.0, 0.0, 0.0, 0.0, np.zeros((10, 10))