from keras.models import Sequential
from keras.layers import Dense, Activation, Input,merge
from keras.optimizers import SGD
import datetime
import numpy as np
from keras.models import load_model, Model
import h5py
from keras import callbacks
import logging
import os
from keras.callbacks import EarlyStopping, ModelCheckpoint
import sys
from keras import optimizers

class printbatch(callbacks.Callback):
    def on_epoch_begin(self, epoch, logs={}):
        logging.info("Epoch: " + str(epoch))
    def on_epoch_end(self, epoch, logs={}):
        logging.info(logs)

class LoggerWriter:
    def __init__(self, level):
        self.level = level

    def write(self, message):
        if message != '\n':
            self.level(message)

    def flush(self):
        self.level(sys.stderr)

def learn(data, labels, stacks, tiers, run_id, output_path, shared_layer_multipliers, layer_multipliers,
          batch_size, learning_rate):
    print("Training with {0} examples. Problem size: {1} tiers {2} stacks.".format(len(data), tiers, stacks))

    shared_layer_multipliers = [x for x in shared_layer_multipliers if x != 0]

    inputArray = []
    for t in range(stacks):
        inputArray.append(Input(shape=(tiers,)))

    layer = inputArray
    for i in range(len(shared_layer_multipliers)):
        shared_dense = Dense(tiers*shared_layer_multipliers[i],activation='relu')
        layerArray = []
        for t in range(stacks):
            layerArray.append(shared_dense(layer[t]))
        layer = layerArray


    merged_vector = merge(layer, mode='concat', concat_axis=-1)

    layer = merged_vector
    for i in range(len(layer_multipliers)):
        layer = Dense(tiers*stacks*layer_multipliers[i],activation='relu')(layer)
    output_layer = Dense(1)(layer)
    model = Model(input=inputArray, output=output_layer)

    logging.info("Value Network Summary:")
    orig_stdout = sys.stdout
    log = logging.getLogger()
    sys.stdout = LoggerWriter(log.info)
    print(model.summary())
    sys.stdout = orig_stdout

    adam = optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=adam, loss='mse', metrics=['mae'])
    logging.info("Start training value model")
    now = datetime.datetime.now()
    model.fit(np.hsplit(data, stacks), labels, nb_epoch=1000, batch_size=batch_size, validation_split=0.2, verbose=2,
              callbacks=[printbatch(), EarlyStopping(monitor='val_loss', patience=50, verbose=0), ModelCheckpoint(os.path.join(output_path, "models",
                            "pm_dnn_value_model_" + str(stacks) + "x" + str(tiers-2) +"_"+ str(now.day) + "." + str(now.month) + "." + str(now.year) + "_"
                            + str(run_id) + "_{epoch:02d}-{val_loss:.2f}"
                            + ".h5"), monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto')])
    model.save(os.path.join(output_path, "models",
                            "pm_dnn_value_model_" + str(stacks) + "x" + str(tiers-2) +"_"+ str(now.day) + "." + str(now.month) + "." + str(now.year) + "_"
                            + str(run_id)
                            + ".h5"))
    logging.info("Finished training. Saved model.")
    return model