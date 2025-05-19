import numpy as np
from pyspark.streaming.dstream import DStream
from pyspark.streaming import StreamingContext
from pyspark.sql import SQLContext
from pyspark.context import SparkContext
from pyspark.ml.linalg import DenseVector
from transforms import Transforms
import os
import glob
from PIL import Image


class DataLoader:
    def __init__(self, sc: SparkContext, ssc: StreamingContext, sqlContext: SQLContext, spark_config, transforms):
        self.sc = sc
        self.ssc = ssc
        self.sqlContext = sqlContext
        self.spark_config = spark_config
        self.transforms = transforms
        self.cinic_root = './cinic-10'  # Đường dẫn tới thư mục CINIC-10

    def streamCINICDataset(self, dataset_type='cinic'):

        CINIC_SETS = ['train', 'valid', 'test']
        
        
        if self.spark_config.split == 'train':
            selected_set = ['train']
        elif self.spark_config.split == 'valid':
            selected_set = ['valid']
        elif self.spark_config.split == 'test':
            selected_set = ['test']
        else:
            raise ValueError("spark_config.split must be 'train', 'valid', or 'test'")
        
        
        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        
       
        data = []
        for set_name in selected_set:
            set_path = os.path.join(self.cinic_root, set_name)
            for class_idx, class_name in enumerate(classes):
                image_paths = glob.glob(os.path.join(set_path, class_name, '*.png'))
                for img_path in image_paths:
                    img = Image.open(img_path)
                    img_array = np.array(img)  
                    label = class_idx
                    data.append([img_array, label])
        
        
        rdd_queue = []
        batch_size = self.spark_config.batch_size
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            rdd = self.sc.parallelize(batch)
            rdd_queue.append(rdd)
        
        return self.ssc.queueStream(rdd_queue)

    def parse_stream(self):
        stream = self.streamCINICDataset()
        if self.transforms:
            stream = stream.map(lambda x: [self.transforms(x[0]), x[1]])
        return stream