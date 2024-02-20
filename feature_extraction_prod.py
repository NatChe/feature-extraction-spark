import io
import numpy as np
import pandas as pd
from PIL import Image
from typing import Iterator

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, broadcast, udf, element_at, split, pandas_udf
from pyspark.sql.types import ArrayType, FloatType
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.feature import PCA

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras import Model

PATH = 's3://natche-p8'
PATH_DATA_INPUT = f'{PATH}/input'
PATH_DATA_OUTPUT = f'{PATH}/output'

def main():
    print('Initializing SparkSession')
    spark = SparkSession.builder.appName('SparkP8').master("local[*]").getOrCreate()

    print('Initializing model')
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    # Freeze existing MobileNetV2 already trained weights
    base_model.trainable = False
    mobilenet_model = Model(inputs=base_model.inputs, outputs=base_model.output)
    mobilenet_weights = mobilenet_model.get_weights()
    # Broadcast the weights
    broadcast_mobilenet_weights = spark.sparkContext.broadcast(mobilenet_weights)

    # Create a model function
    def create_model():
        # Load MobileNetV2 model
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    
        # Freeze existing MobileNetV2 already trained weights
        base_model.trainable = False
        mobilenet_model = Model(inputs=base_model.inputs, outputs=base_model.output)
    
        # Use the broadcasted weights
        mobilenet_model.set_weights(broadcast_mobilenet_weights.value)
        
        return mobilenet_model

    print('Loading images')
    images = spark.read.format("binaryFile") \
        .option("dropInvalid", True) \
        .option("pathGlobFilter", "*.jpg") \
        .option("recursiveFileLookup", "true") \
        .load(PATH_DATA_INPUT)

    # extract label
    images = images.withColumn('label', element_at(split(images['path'], '/'),-2))
    
    print(images.printSchema())
    print(images.select('path').show(truncate=False))

    print('Extracting features')
    def preprocess(content):
        """
        Preprocesses raw image bytes for prediction.
        """
        img = Image.open(io.BytesIO(content)).resize([224, 224])
        arr = img_to_array(img)
        
        return preprocess_input(arr)
    
    def featurize_series(model, content_series):
        """
        Featurize a pd.Series of raw images using the input model.
        :return: a pd.Series of image features
        """
        input = np.stack(content_series.map(preprocess))
        preds = model.predict(input)
        
        # For some layers, output features will be multi-dimensional tensors.
        # We flatten the feature tensors to vectors for easier storage in Spark DataFrames.
        output = [p.flatten() for p in preds]
        
        return pd.Series(output)
    
    @pandas_udf(ArrayType(FloatType()))
    def featurize_udf(content_series_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
        '''
        This method is a Scalar Iterator pandas UDF wrapping our featurization function.
        The decorator specifies that this returns a Spark DataFrame column of type ArrayType(FloatType).
    
        :param content_series_iter: This argument is an iterator over batches of data, where each batch
                                  is a pandas Series of image data.
        '''
        # With Scalar Iterator pandas UDFs, we can load the model once and then re-use it
        # for multiple data batches.  This amortizes the overhead of loading big models.

        model = create_model()
        
        for content_series in content_series_iter:
            yield featurize_series(model, content_series)


    features_df = images.repartition(24).select(col("path"), col("label"), featurize_udf("content").alias("features"))
    features_df.select('features').show(1, True)

    # Convert features to dense vector
    to_vector_udf = udf(lambda features: Vectors.dense(features), VectorUDT())
    features_df = features_df.withColumn("features_vector", to_vector_udf("features"))

    print('Performing PCA')
    pca = PCA(k=50, inputCol="features_vector", outputCol="pca_features")
    pca_model = pca.fit(features_df)
    df_pca = pca_model.transform(features_df)
    df_pca.printSchema()

    print('Writing files')
    df_pca.write.mode("overwrite").parquet(PATH_DATA_OUTPUT)



if __name__ == "__main__":
    main()