import time
import json
import pickle
import socket
import argparse
import numpy as np
from tqdm import tqdm
import os
from PIL import Image

parser = argparse.ArgumentParser(
    description='Streams CINIC-10 dataset to a Spark Streaming Context')
parser.add_argument('--file', '-f', help='Dataset to stream', required=False, type=str, default="cinic-10")
parser.add_argument('--batch-size', '-b', help='Batch size', required=False, type=int, default=128)
parser.add_argument('--endless', '-e', help='Enable endless stream', required=False, type=bool, default=False)
parser.add_argument('--split', '-s', help="train, valid, or test split", required=False, type=str, default='train')
parser.add_argument('--sleep', '-t', help="streaming interval", required=False, type=int, default=3)

TCP_IP = "localhost"
TCP_PORT = 6100

class Dataset:
    def __init__(self) -> None:
        self.data = []
        self.labels = []
        self.epoch = 0
        self.label_map = {
            'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4,
            'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9
        }

    def data_generator(self, split, batch_size):
        base_path = f"cinic-10/{split}"
        images = []
        labels = []

       
        for class_name in self.label_map:
            class_path = os.path.join(base_path, class_name)
            if not os.path.exists(class_path):
                continue
            for img_name in os.listdir(class_path):
                if img_name.endswith('.png'):
                    img_path = os.path.join(class_path, img_name)
                    img = Image.open(img_path).convert('RGB')
                    img_array = np.array(img).flatten()  
                    images.append(img_array)
                    labels.append(self.label_map[class_name])

       
        images = np.array(images)
        labels = np.array(labels)

        
        batches = []
        for ix in range(0, len(images) - batch_size + 1, batch_size):
            batch_images = images[ix:ix + batch_size].tolist()
            batch_labels = labels[ix:ix + batch_size].tolist()
            batches.append([batch_images, batch_labels])

       
        self.data = images[ix + batch_size:].tolist()
        self.labels = labels[ix + batch_size:].tolist()

        return batches

    def sendCINICBatchToSpark(self, tcp_connection, split, batch_size):
        pbar = tqdm(total=int((9e4 // batch_size) + 1))  
        data_received = 0
        batches = self.data_generator(split, batch_size)

        for batch in batches:
            image, labels = batch
            received_shape = (len(image), 32, 32, 3)  

            feature_size = len(image[0])  

            payload = dict()
            for mini_batch_index in range(len(image)):
                payload[mini_batch_index] = dict()
                for feature_index in range(feature_size):
                    payload[mini_batch_index][f'feature{feature_index}'] = image[mini_batch_index][feature_index]
                payload[mini_batch_index]['label'] = labels[mini_batch_index]

            send_batch = (json.dumps(payload) + '\n').encode()
            try:
                tcp_connection.send(send_batch)
            except BrokenPipeError:
                print("Either batch size is too big for the dataset or the connection was closed")
                break
            except Exception as error_message:
                print(f"Exception thrown but was handled: {error_message}")
                continue

            data_received += 1
            pbar.update(1)
            pbar.set_description(f"epoch: {self.epoch} it: {data_received} | received: {received_shape} images")
            time.sleep(sleep_time)

        
        if self.data:
            image = self.data
            labels = self.labels
            received_shape = (len(image), 32, 32, 3)

            feature_size = len(image[0])
            payload = dict()
            for mini_batch_index in range(len(image)):
                payload[mini_batch_index] = dict()
                for feature_index in range(feature_size):
                    payload[mini_batch_index][f'feature{feature_index}'] = image[mini_batch_index][feature_index]
                payload[mini_batch_index]['label'] = labels[mini_batch_index]

            send_batch = (json.dumps(payload) + '\n').encode()
            try:
                tcp_connection.send(send_batch)
            except BrokenPipeError:
                print("Either batch size is too big for the dataset or the connection was closed")
            except Exception as error_message:
                print(f"Exception thrown but was handled: {error_message}")

            data_received += 1
            pbar.update(1)
            pbar.set_description(f"epoch: {self.epoch} it: {data_received} | received: {received_shape} images")
            self.data = []
            self.labels = []
            time.sleep(sleep_time)

        pbar.close()
        self.epoch += 1

    def connectTCP(self):   
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((TCP_IP, TCP_PORT))
        s.listen(1)
        print(f"Waiting for connection on port {TCP_PORT}...")
        connection, address = s.accept()
        print(f"Connected to {address}")
        return connection, address

    def streamCINICDataset(self, tcp_connection, dataset_type='cinic-10'):
        self.sendCINICBatchToSpark(tcp_connection, train_test_split, batch_size)

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)

    input_file = args.file
    batch_size = args.batch_size
    endless = args.endless
    sleep_time = args.sleep
    train_test_split = args.split
    dataset = Dataset()
    tcp_connection, _ = dataset.connectTCP()

    if input_file == "cinic-10":
        _function = dataset.streamCINICDataset
    if endless:
        while True:
            _function(tcp_connection, input_file)
    else:
        _function(tcp_connection, input_file)

    tcp_connection.close()