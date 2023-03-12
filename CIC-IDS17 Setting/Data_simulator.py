#
#   Created: 2023
#   Author: Seraphin Zunzer
#   License: MIT License
# ---------------------------------------------------------------


# Data simulator, which reads in the datasets and sends
# the samples to the client one after the other.


import json
import math
import socket
import sys
import pandas as pd
import time

HOST = "127.0.0.1"  # The server's hostname or IP address
CLIENT_PORT = None  # The port used by the client, set by cli


def load_dataset(path):
    client_df = pd.read_csv(path)  # read data

    client_labels = client_df[' Label']  # separate labels

    del client_df[' Label']  # delete labels in dataset

    client_data = client_df.to_numpy()  # convert data into array

    return client_data, client_labels


def send_sample_to_client(packet, label):
    """Send new weights back to server"""
    bytes_weights_data = json.dumps([packet, [label]]).encode()    # put data int lists and convert them into json
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, CLIENT_PORT))
            s.sendall(bytes_weights_data) # send data to connected client
    except Exception as e:
        print(f"Data Source failed to connect to Client {CLIENT_ID}, {CLIENT_PORT}")
        time.sleep(20)


if __name__ == "__main__":
    CLIENT_ID = int(sys.argv[1])
    CLIENT_PORT = int(sys.argv[2])
    DATASET_PATH = f'datasets/MachineLearningCVE/clients_separated/{CLIENT_ID}.csv'

    packets, labels = load_dataset(DATASET_PATH)
    time.sleep(40)
    for i in range(len(packets)):
        sample = packets[i]
        label = labels[i]
        if all([isinstance(item, float) and math.isfinite(item) for item in sample]):
            send_sample_to_client(sample.tolist(), label)
        else:
            print("Convert Sample")
            sample_new= [0 if not isinstance(value, float) or not math.isfinite(value) else value for value in sample]
            send_sample_to_client(sample_new, label)

        if i % 20000 == 0:
            print("PROCESSED 20000 DATA SAMPLES:", i," \n\n\n\n")

    print("FINISHED SIMULATOR\n")
