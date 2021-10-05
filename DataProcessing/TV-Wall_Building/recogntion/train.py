from __future__ import print_function
import os
from shutil import copyfile

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

import keras
from util import get_sample_ws_from_multiple_y_per_seg_with_one_hot, get_cl_for_timestep_one_hot



def train_classification(model, X_train, Y_train,
                  model_name, model_dir, log_dir,
                  batch_size = 32, epochs = 5, patience = 15, all_classes = None):


  # Do the training.
  # logging = keras.callbacks.TensorBoard(log_dir= os.path.join(log_dir, model_name, "log"),
  #                                       histogram_freq=1,
  #                                       batch_size=batch_size,
  #                                       write_graph=True,
  #                                       write_grads=True,
  #                                       write_images=True,
  #                                       embeddings_freq=0,
  #                                       embeddings_layer_names=None,
  #                                       embeddings_metadata=None)

  model_path = os.path.join(model_dir, model_name + "weights")
  saving = keras.callbacks.ModelCheckpoint(model_path, monitor='val_acc', verbose=0, save_best_only=True,
                                           save_weights_only=True, mode='auto', period=1)

  earlyStopping = keras.callbacks.EarlyStopping(monitor='val_acc',
                                                patience=patience,
                                                verbose=1, mode='auto')




  y_train_one_per_win = get_cl_for_timestep_one_hot(Y_train, all_classes)
  from sklearn.model_selection import train_test_split
  X, X_val, y, y_val = train_test_split(X_train, Y_train, stratify=y_train_one_per_win, test_size=0.1, random_state=42)
  sample_ws = get_sample_ws_from_multiple_y_per_seg_with_one_hot(y)

  if not os.path.exists(model_path + "Done"):
      model.fit(x=X, y=y,
                callbacks=[earlyStopping, saving],
                batch_size=batch_size,
                epochs=epochs,
                validation_split=0.2,
                sample_weight=sample_ws,
                validation_data=(X_val, y_val),
                shuffle=True)


      # sample_ws = get_sample_ws_from_multiple_y_per_seg_with_one_hot(Y_train)
      # model.fit(x=X_train, y=Y_train,
      #             callbacks=[earlyStopping, saving],
      #             batch_size=batch_size,
      #             epochs=epochs,
      #             validation_split=0.2,
      #             sample_weight = sample_ws,
      #             # validation_data=(X_test, y_test),
      #             shuffle=True)

      # Move the current file to the "done" stage, so that
      copyfile(model_path, model_path + "Done")
      os.remove(model_path)

  # # Actually use the best model.
  model.load_weights(model_path + "Done")

  # Return Model and procesing function.
  return model