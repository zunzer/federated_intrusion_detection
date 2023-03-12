#
#   Created: 2023
#   Author: Seraphin Zunzer
#   License: MIT License
# ---------------------------------------------------------------


# Federated learning client that receives the global model weights,
# detects attacks in the arriving datastreams from the data simulator,
# and trains the local model afterward.
# This model is transmitted to the Server for weight aggregation using FedAVG

import socket
import threading
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as K
import keras
import sys, time
from datetime import datetime, timedelta
from Crypto.Cipher import AES
import logging

logging.basicConfig(level=logging.DEBUG)

HOST = "127.0.0.1"  # The server's hostname or IP address
SERVER_PORT = 65432  # The port of the server
CLIENT_PORT = None  # The port used by the client, set by cli
DATA_RECEIVING_PORT = None  # port for incoming data
CLIENT_ID = None  # will be set with command line arguments
encryption = True

lr = 0.001  ###
NUM_EPOCHS = 3  ###
BATCHSIZE = 32
DATA_STEPSIZE = 32
DROPOUT = 0.5

LOSS = 'mse'
METRICS = []
OPTIMIZER = keras.optimizers.Adam(lr=lr)
DATASET_SIZE = 78
THRESHOLD_MAD = 70

X = [0]
FN = [np.nan]
TN = [np.nan]
FP = [np.nan]
TP = [np.nan]

global latest_tp
global latest_tn
global latest_fp
global latest_fn
global GLOBAL_WEIGHTS
global train_count


class SimpleMLP:
    """Creates the keras model that is passed to the clients"""

    @staticmethod
    def build():
        encoder = Sequential([
            keras.layers.Dense(DATASET_SIZE, input_shape=(DATASET_SIZE,)),
            keras.layers.Dense(39),
            keras.layers.Dense(20),
        ])
        decoder = Sequential([
            keras.layers.Dense(39, input_shape=(20,)),
            keras.layers.Dense(DATASET_SIZE),
        ])
        return encoder, decoder



class Data_Receiving_Thread(threading.Thread):
    """Data Receiving & Prediction Thread"""

    def __init__(self):
        self.cohda_host = "127.0.0.1"
        self.cohda_port = DATA_RECEIVING_PORT
        self.prediction_data_batch = []  # container for data that should be predicted
        self.prediction_data_label = []
        self.prediction_time = []  # times needed for prediction

        self.train_data_batch = []  # container for data that should be trained with
        self.train_data_label = []

        self.data_container = []
        self.label_container = []
        self.time_container = []

        # initialize socket
        self.data_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.data_socket.bind((self.cohda_host, self.cohda_port))
        self.data_socket.listen()

        # initialize metrics
        self.metrics = [0, 0, 0, 0]

        # initialize prediction model and thread
        input_format = keras.layers.Input(shape=(DATASET_SIZE,))
        enc, dec = SimpleMLP.build()
        self.autoencoder = tf.keras.models.Model(inputs=input_format, outputs=dec(enc(input_format)))
        self.autoencoder.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=METRICS)
        threading.Thread.__init__(self)

    def run(self):
        self.collect_samples()

    def recv_single_sample(self):
        """Receive one data sample from data simulator"""
        conn, addr = self.data_socket.accept()
        with conn:
            while True:
                data = conn.recv(1024)
                if not data:
                    break
                conn.close()
                return json.loads(data)[0], json.loads(data)[1]  # convert data from binary json to list

    def collect_samples(self):
        # initialize timer to measure how long it takes to receive first prediction container
        start = datetime.now()
        # num = 0
        while 1:
            try:
                # receive one sample and add it to containier and store receiving time
                x, y = self.recv_single_sample()
                self.data_container.append(x)
                self.label_container.append(y)
                self.time_container.append(datetime.now())
                # num +=1
            except:
                logging.error("Error occured while Data receiving!")

            if len(self.data_container) > DATA_STEPSIZE:
                # if enough values are collected
                # if train_count==0:
                # print time needed to collect first container
                # print(f"Needed {(datetime.now()-start).total_seconds()} for collecting data")
                # start = datetime.now()
                # store data for prediction
                self.prediction_data_batch.append(
                    self.data_container)  # add values to new batch, to start with prediction
                self.prediction_data_label.append(self.label_container)
                self.prediction_time.append(self.time_container)

                # empty containers
                self.data_container = []
                self.label_container = []
                self.time_container = []

                # start prediction
                self.start_prediction()
            # if num == 1000:
            #    print("Time needed for 1000:", (datetime.now()-start).total_seconds() )
            # if num == 5000:
            #    print("Time needed for 5000:", (datetime.now()-start).total_seconds() )
            #    start = datetime.now()
            #    num = 0

    def start_prediction(self):
        # if enough trainingrounds made
        if train_count > -1:

            # get datasets and make prediction
            pred_dataset = np.array(self.prediction_data_batch.pop(0))
            pred_labels = np.array(self.prediction_data_label.pop(0))
            pred_time = np.array(self.prediction_time.pop(0))
            self.make_predictions(GLOBAL_WEIGHTS, pred_dataset, pred_labels, pred_time)

            # append data and labels to training data
            self.train_data_batch.append(pred_dataset)
            self.train_data_label.append(pred_labels)
        else:
            self.train_data_batch.append(self.prediction_data_batch.pop(0))
            self.train_data_label.append(self.prediction_data_label.pop(0))

    def make_predictions(self, prediction_weights, dataset, label, pred_time):
        """make predictions for the new incoming data"""

        # set local model weight to the weight of the global model
        self.autoencoder.set_weights(prediction_weights)

        # get reconstruction errors
        reconstructions = self.autoencoder.predict(dataset)

        # calculate mse between reconstructions and initial data
        mse = np.mean(np.power(dataset - reconstructions, 2), axis=1)

        # calculate deviation of mean
        def mad_score(points):
            m = np.median(points)
            ad = np.abs(points - m)
            mad = np.median(ad)
            return 0.6745 * ad / mad

        z_scores = mad_score(mse)

        # check if deviations is higher than pre set threshold
        outliers = z_scores > THRESHOLD_MAD

        # get lists of the predicted and the true labels
        prediction = ["ANOMALY" if l else "BENIGN" for l in outliers]
        labels_true = ["BENIGN" if k == "BENIGN" else "ANOMALY" for k in label]

        # calculate the time needed for all traffic
        time_needed = [(datetime.now() - pred_time[j]).total_seconds() for j in range(len(label))]

        def find_indices(list_to_check, item_to_find):
            indices = []
            for idx, value in enumerate(list_to_check):
                if value == item_to_find:
                    indices.append(idx)
            return indices

        for i in list(set([item for sublist in label for item in sublist])):
            if not (i == "BENIGN"):
                time_deltas = []
                count = 0
                for k in find_indices(label, i):
                    time_deltas.append((datetime.now() - pred_time[k]).total_seconds())
                    count += 1
                print(f'\nTime for {i}: {sum(time_deltas) / count}\n')

        # calculate confusion matrice
        fp, tp, fn, tn = self.confusion_matrice(prediction, labels_true)
        try:
            f1 = (2 * tp) / ((2 * tp) + fp + fn)
            # logging.info(f1)
        except ZeroDivisionError:
            # logging.info("F1 0")
            f1 = 0

        # store values in metrics to send them to server
        self.metrics = [x + y for x, y in zip(self.metrics, [fp, tp, fn, tn])]

        logging.info(
            f'Client {CLIENT_ID}: False positives: {fp}, False negatives: {fn}, True positives: {tp}, True negatives: {tn}, F1: {f1}')

        # print outliers
        for i in range(0, len(label)):
            if not label[i][0] == "BENIGN":
                print(label[i], outliers[i], round(z_scores[i], 7), time_needed[i])
                if outliers[i] == True:
                    with open(f'results/times/times_{label[i]}.txt', "a+", newline='') as f:
                        f.write(f'{time_needed[i]}\n')  # write times in to file

        # self.analyze_lr(fp, tp, fn, tn)
        # self.analyze_MAD(z_scores, labels_true)
        # self.process_anomalys(threshold_prediction, dataset, label)

    def analyze_lr(self, fp, tp, fn, tn):
        fpr = fp / (fp + tn)
        # tpr = tp / (tp+fn)
        with open(f'results/results_lr.txt', "a+", newline='') as f:
            f.write(f'{fpr}\n')

    def analyze_MAD(self, z_scores, labels_true):
        tpr_i = []
        i = 0
        while i < 500:
            outliers = z_scores > i
            pred = ["BENIGN" if not l else "ANOMALY" for l in outliers]
            fp, tp, fn, tn = self.confusion_matrice(pred, labels_true)
            fpr = fp / (fp + tn)
            tpr_i.append(fpr)
            i += 1
        print("WRITING TO FILE FPR VS THRESHOLD MAD ")
        with open('results/mad.txt', 'w') as file:
            for i, tpr in enumerate(tpr_i):
                file.write(f'{i},{tpr}\n')

    def confusion_matrice(self, prediction, labels_true):
        """calculate confusion matrice"""
        fp = 0
        tp = 0
        fn = 0
        tn = 0
        for i in range(0, len(prediction)):
            if (prediction[i] == "ANOMALY" and labels_true[i] == "BENIGN"):
                fp += 1
            elif (prediction[i] == "ANOMALY" and labels_true[i] == "ANOMALY"):
                tp += 1
            elif (prediction[i] == "BENIGN" and labels_true[i] == "ANOMALY"):
                fn += 1
            elif (prediction[i] == "BENIGN" and labels_true[i] == "BENIGN"):
                tn += 1

        return fp, tp, fn, tn

    def process_anomalys(self, predictions, dataset, label):
        now = datetime.now()
        timestamp = now.strftime("%m/%d/%Y, %H:%M:%S")

        with open(f"results_{CLIENT_ID}.txt", "a") as txt_file:
            for i in range(len(predictions)):
                if predictions[i] == "anomaly":
                    txt_file.write(f" {timestamp} - {label[i]}  \n")  # dataset[i], label[i])


# -----------------------------------------  Training & Main Thread ----------------------------


def store_loss(loss_hist):
    """function to write loss to file"""
    # if CLIENT_ID == 1:
    with open(f'results/clientlosses/loss_{CLIENT_ID}.txt', 'a') as file:
        file.write(f'{loss_hist[0]}\n')


def client_update(old_server_weights, dataset):
    """Performs training using the received server model weights on the client's dataset"""

    # initialize and compile model
    input_format = keras.layers.Input(shape=(DATASET_SIZE,))
    enc, dec = SimpleMLP.build()
    autoencoder = tf.keras.models.Model(inputs=input_format, outputs=dec(enc(input_format)))
    autoencoder.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=METRICS)
    autoencoder.set_weights(old_server_weights)  # set local model weight to the weight of the global model

    # fit local model with client's data
    history = autoencoder.fit(dataset, dataset, verbose=2, epochs=NUM_EPOCHS, batch_size=BATCHSIZE)

    store_loss(history.history['loss'])  # save loss in file

    weights = autoencoder.get_weights()  # get new weights
    return weights


def recv_global_weights_from_server(s):
    """Receive global weights from the server"""
    binary_data = b''
    s.settimeout(300.0)  # after 5min, assume the server is down
    conn, addr = s.accept()
    with conn:
        while True:
            recv_data = conn.recv(30000)  # receive bytes. value needs to be increase if model is significantly larger
            if not recv_data:
                break
            binary_data = b"".join([binary_data, recv_data])
    conn.close()

    binary_data = aes_decrypt(binary_data)  # decrypt weights
    received_data = json.loads(binary_data)  # convert data from binary json to list
    new_global_weights = unpack_weights(received_data)  # extract global weights from list
    return new_global_weights


def aes_decrypt(encrypted):
    """decrypt bytestring based on aes"""
    if encryption:
        obj = AES.new('This is a key123'.encode("utf8"), AES.MODE_CFB, 'This is an IV456'.encode("utf8"))
        return obj.decrypt(encrypted)
    else:
        return encrypted


def aes_encrypt(decrypted):
    """encrypt bytestring based on aes"""
    if encryption:
        obj = AES.new('This is a key123'.encode("utf8"), AES.MODE_CFB, 'This is an IV456'.encode("utf8"))
        return obj.encrypt(decrypted)
    else:
        return decrypted


def unpack_weights(received_data):
    """unpack local weigths from json format """
    weight_list = []
    for layer in range(0, len(received_data)):
        weight_list.append(np.array(received_data[layer]))
    return weight_list


def pack_weights(global_weights, metrics):
    """pack local weigths into json format for transmission"""
    list_of_weights = [f'client_{CLIENT_ID}', [metrics[0]], [metrics[1]], [metrics[2]], [metrics[3]]]
    for layer in range(0, len(global_weights)):
        list_of_weights.append(global_weights[layer].tolist())
    return json.dumps(list_of_weights).encode()


def send_weights_to_server(weights, metrics):
    """Send local weights back to server"""
    bytes_weights_data = pack_weights(weights, metrics)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, SERVER_PORT))
        s.sendall(aes_encrypt(bytes_weights_data))


def register_client(ID, Client_Port, Server_port):
    """Register client at the server"""
    logging.info("Try registration: ")
    bytes_weights_data = json.dumps(
        ["registration", ID, Client_Port, Server_port]).encode()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, SERVER_PORT))
        s.sendall(aes_encrypt(bytes_weights_data))
    return True


def no_data(ID, Client_Port, Server_port):
    """Send message with no data flag to the server"""
    bytes_weights_data = json.dumps(
        ["no data", ID, Client_Port, Server_port]).encode()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, SERVER_PORT))
        s.sendall(aes_encrypt(bytes_weights_data))
    return


if __name__ == "__main__":
    logging.info(f"START CLIENT {sys.argv[1]}")

    # initialize metrics with 0
    latest_fn = 0
    latest_fp = 0
    latest_tn = 0
    latest_tp = 0

    # get client ID and ports
    CLIENT_ID = int(sys.argv[1])
    CLIENT_PORT = int(sys.argv[2])
    DATA_RECEIVING_PORT = int(sys.argv[3])

    # bind to ports
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # open socket
    s.bind((HOST, CLIENT_PORT))  # bind process to port
    s.listen()

    # initialize variables
    registered = False
    train_count = 0
    last_registration = datetime.now() - timedelta(seconds=90)

    # create data receiving thread
    data_thread = Data_Receiving_Thread()
    data_thread.start()

    while 1:

        # client is not registered yet and last request for registration is 15sec ago
        if (not registered) and ((datetime.now() - last_registration).total_seconds() > 15):
            try:
                # try registration and update time
                last_registration = datetime.now()
                registered = register_client(CLIENT_ID, CLIENT_PORT, SERVER_PORT)
                if (registered):
                    logging.info(f"Registration successful! {CLIENT_ID} {CLIENT_PORT} {DATA_RECEIVING_PORT}")
            except Exception as e:
                logging.warning(f"Client {CLIENT_ID}: Failed to register at server. Retry in 10sec.")
                registered = False

        # client is registered
        if registered:
            try:
                # try to receive global weights from server
                GLOBAL_WEIGHTS = recv_global_weights_from_server(s)
            except socket.timeout:
                # timeout when serve is not responding with weights. set client to unregistered for trying to register again
                registered = False
                logging.warning("SERVER NOT AVAILABLE")
                continue

            # check if client has data available
            if (len(data_thread.train_data_batch) > 1):  # registered == True and
                logging.info('_' * 50)
                logging.info(f"TRAINING CLIENT:  {CLIENT_ID}")

                # get all traindata from data thread
                train_dataset = np.array([item for sublist in data_thread.train_data_batch for item in
                                          sublist])  # np.array(data_thread.train_data_batch.pop(0))

                print("Training data used for training ", len(train_dataset))

                # train local model
                client_weights = client_update(GLOBAL_WEIGHTS, train_dataset)

                data_thread.train_data_label = []
                data_thread.train_data_batch = []

                # add 1 to traincount. Is done to skip prediction in the first iterations where the client has not been trained yet
                train_count += 1
                try:
                    # send local weights to server
                    send_weights_to_server(client_weights, data_thread.metrics)

                    # save tpr and fpr for each client in file
                    fp = data_thread.metrics[0]
                    tp = data_thread.metrics[1]
                    fn = data_thread.metrics[2]
                    tn = data_thread.metrics[3]
                    fpr = (fp + tn) and fp / (fp + tn)
                    tpr = (tp + fn) and tp / (tp + fn)
                    fnr = (fn + tp) and fn / (fn + tp)
                    ac = (tp + tn) / (tp + tn + fp + fn)
                    with open(f'results/clientmetrics/client_{CLIENT_ID}tpr_fpr.txt', 'a') as file:
                        file.write(f'{tpr},{fpr},{fnr},{ac}\n')

                    data_thread.metrics = [0, 0, 0, 0]
                except:
                    logging.error("Could not send weights back!")

                logging.info("Finished Training")
                K.clear_session()  # clear model
            else:
                # if no data available, send no data message
                no_data(CLIENT_ID, CLIENT_PORT, SERVER_PORT)

            print(f"Client {CLIENT_ID}: Collected: Training: ",
                  len(data_thread.train_data_batch),
                  "New Batch: [" + '█' * int(len(data_thread.data_container) / 100) + ' ' * int((DATA_STEPSIZE - int(
                      len(data_thread.data_container))) / 100) + "]")
