from __future__ import print_function
import numpy as np
from numpy.lib.stride_tricks import as_strided as ast
import keras

def make_one_hot(y, lst_ordered_classes):
  one_hot = np.zeros( (len(y), len(lst_ordered_classes)) )
  for yi, yv in enumerate(y):
    one_hot[yi][lst_ordered_classes.index(yv)] = 1.0
  return one_hot

def undo_sliding_window(X_win, win_shift):

  if len(X_win.shape) != 2:
    n_wins, win_size, n_feats = X_win.shape
  else:
    n_wins, win_size = X_win.shape
    n_feats = 1
    X_win = X_win.reshape( (n_wins, win_size, 1) )
  original_number_of_steps = win_size + (n_wins - 1) * win_shift

  undone = np.zeros(shape=(original_number_of_steps, n_feats))

  undone[0:win_size] = X_win[0]
  undone[win_size:] = X_win[1:,-win_shift:,:].reshape((original_number_of_steps - win_size, n_feats))

  return undone

def norm_shape(shape):
    '''
    Normalize numpy array shapes so they're always expressed as a tuple,
    even for one-dimensional shapes.

    Parameters
        shape - an int, or a tuple of ints

    Returns
        a shape tuple
    '''
    try:
        i = int(shape)
        return (i,)
    except TypeError:
        # shape was not a number
        pass

    try:
        t = tuple(shape)
        return t
    except TypeError:
        # shape was not iterable
        pass

    raise TypeError('shape must be an int, or a tuple of ints')


def sliding_window(a,ws,ss = None,flatten = True):
    '''
    Return a sliding window over a in any number of dimensions

    Parameters:
        a  - an n-dimensional numpy array
        ws - an int (a is 1D) or tuple (a is 2D or greater) representing the size
             of each dimension of the window
        ss - an int (a is 1D) or tuple (a is 2D or greater) representing the
             amount to slide the window in each dimension. If not specified, it
             defaults to ws.
        flatten - if True, all slices are flattened, otherwise, there is an
                  extra dimension for each dimension of the input.

    Returns
        an array containing each n-dimensional window from a
    '''

    if None is ss:
        # ss was not provided. the windows will not overlap in any direction.
        ss = ws
    ws = norm_shape(ws)
    ss = norm_shape(ss)

    # convert ws, ss, and a.shape to numpy arrays so that we can do math in every
    # dimension at once.
    ws = np.array(ws)
    ss = np.array(ss)
    shape = np.array(a.shape)


    # ensure that ws, ss, and a.shape all have the same number of dimensions
    ls = [len(shape),len(ws),len(ss)]
    if 1 != len(set(ls)):
        raise ValueError(\
        'a.shape, ws and ss must all have the same length. They were %s' % str(ls))

    # ensure that ws is smaller than a in every dimension
    if np.any(ws > shape):
        raise ValueError(\
        'ws cannot be larger than a in any dimension.\
 a.shape was %s and ws was %s' % (str(a.shape),str(ws)))

    # how many slices will there be in each dimension?
    newshape = norm_shape(((shape - ws) // ss) + 1)
    # the shape of the strided array will be the number of slices in each dimension
    # plus the shape of the window (tuple addition)
    newshape += norm_shape(ws)
    # the strides tuple will be the array's strides multiplied by step size, plus
    # the array's strides (tuple addition)
    newstrides = norm_shape(np.array(a.strides) * ss) + a.strides
    strided = ast(a,shape = newshape,strides = newstrides)
    if not flatten:
        return strided

    # Collapse strided so that it has one more dimension than the window.  I.e.,
    # the new array is a flat list of slices.
    meat = len(ws) if ws.shape else 0
    firstdim = (np.product(newshape[:-meat]),) if ws.shape else ()
    dim = firstdim + (newshape[-meat:])
    # remove any dimensions with size 1
    dim = filter(lambda i : i != 1,dim)
    return strided.reshape(dim)

# This one uses numpy stride magic to sliding windows in place.
def make_sliding_window_in_place(X, y, win_size, win_step, strategy):
  """

  :param X: the X numpy array (n_instances, n_features, ...), here we allow more than one feature dimension
    since this may be useful for other models... maybe.
  :param y: the labels for the X array. This will be transformed in the same way as the X array,
            but a single label will be chosen for each window by majority voting. None
  :param win_size:
    the size of the window.
  :param batch_size:
    the batch size for the model.
  :param strategy:
    "majority" : label for a window is the one with highest count.
    "last" : label for a window is decided by its last label.
    "all"    : keep all labels of the window.
    ignored if y is None.
  :return:
    X as a numpy array in the form (n_windows, window_size, n_dims_from_sensors)
    Y as a numpy array with the labels
  """
  # Calculate X.
  if X is not None:
    n_features = len(X[0])
    n_windows = ((len(X) - win_size) / win_step) + 1
    new_X = np.zeros( (n_windows, win_size, n_features) )
    for w_i in xrange(len(new_X)):
      new_X[w_i] = X[w_i*win_step:w_i*win_step + win_size,:]
  else:
    new_X = None

  # Calculate Y.
  if y is not None:
    sequence_Y = sliding_window(y, ws=(win_size), ss=(win_step))
    # sequence_Y = []
    # for w_i in xrange(len(new_X)):
    #   sequence_Y.extend( [ y[w_i*win_step:w_i*win_step + win_size]] )
    # sequence_Y = np.array(sequence_Y)

    f_decide_label = None
    if strategy == "majority":
      def f_highest_count(yw):
         values, counts = np.unique(yw,return_counts=True)
         return values[np.argmax(counts)]
      f_decide_label = f_highest_count
    elif strategy == "last":
      f_decide_label = lambda yw : yw[-1]
    elif strategy == "all":
      f_decide_label = lambda yw: yw
    elif strategy == "no-null":
      f_decide_label = lambda yw: "NULL" if ("NULL" in yw) else yw[-1]
    else:
      raise Exception("unknown label determination strategy " + str(strategy))

    new_Y = [f_decide_label(y_window_data) for y_window_data in sequence_Y]
  else:
    new_Y = None

  return new_X, new_Y


def make_one_label_per_timestep_one_hot(y_train, all_classes):
  y_train_one_hot = [ [all_classes.index(l) for l in y] for y in y_train]
  y_train_one_hot = keras.utils.to_categorical(y_train_one_hot, len(all_classes))
  return y_train_one_hot


def f_highest_count(yw, axis=None):
  values, counts = np.unique(yw, return_counts=True, axis=axis)
  return values[np.argmax(counts)]

def get_cl_for_timestep_one_hot(one_hot_ts, lst_ordered_classes):
  return [f_highest_count([lst_ordered_classes[i]
                    for i in np.argmax(t, axis=1)]) for t in one_hot_ts]



def get_cl_from_probabilities(probs, lst_ordered_classes):
    return [lst_ordered_classes[np.argmax(p)] for p in probs]


def get_sample_ws_from_multiple_y_per_seg_with_one_hot(Y):
  lst_ordered_classes = [str(i) for i in range(Y.shape[-1])]
  Y_cls = [ [lst_ordered_classes[i] for i in np.argmax(t, axis=1)] for t in Y]
  Y_fl = np.vstack(Y_cls)
  Y_fl = np.array(Y_fl).reshape( (len(Y_fl)*len(Y_fl[0]) ) )

  cl_names, cl_counts = np.unique(Y_fl, return_counts=True, axis=0)
  map_counts = { cl_name : float(cl_count) for cl_name, cl_count in zip(cl_names, cl_counts) }
  map_ws = {}
  for cl_i, cl_name in enumerate(lst_ordered_classes):
    map_ws[cl_name] = len(Y_fl) / float(map_counts[cl_name])
  print(lst_ordered_classes)
  print(map_ws)


  sample_ws = np.zeros( len(Y_cls) )
  for i, y in enumerate(Y_cls):
    for l in y:
      sample_ws[i] += map_ws[l]

  return sample_ws


# This function is when labels are strings.
def get_sample_ws_from_multiple_y_per_seg(Y, lst_ordered_classes):
  Y_fl = np.array(Y).reshape( (len(Y)*len(Y[0]) ) )

  cl_names, cl_counts = np.unique(Y_fl, return_counts=True, axis=0)
  map_counts = { cl_name : float(cl_count) for cl_name, cl_count in zip(cl_names, cl_counts) }
  map_ws = {}
  for cl_i, cl_name in enumerate(lst_ordered_classes):
    map_ws[cl_name] = len(Y_fl) / float(map_counts[cl_name])
  print(lst_ordered_classes)
  print(map_ws)


  sample_ws = np.zeros( len(Y) )
  for i, y in enumerate(Y):
    for l in y:
      sample_ws[i] += map_ws[l]

  return sample_ws


def convert_lbls_from_one_per_ts_to_one_per_ts(y_test):
    y_one_per_win = []
    for y in y_test:
        y_in_w, c_in_w = np.unique(y, return_counts=True)
        y_one_per_win.append(y_in_w[np.argmax(c_in_w)])
    return y_one_per_win

def smooth_predictions_using_soft_voting(probs, window_size):

    half_win = window_size / 2

    probs_smoothed = np.zeros(probs.shape)
    for i in xrange(len(probs_smoothed)):
        probs_smoothed[i] = np.sum(probs[max(0,i-half_win):i+half_win], axis=0)
        probs_smoothed[i] /= np.sum(probs_smoothed[i])
    return probs_smoothed


def smooth_predictions_using_hard_voting(probs, window_size, all_classes):

    lst_all_classes = list(all_classes)

    cls_before_smooth = get_cl_from_probabilities(probs, all_classes)
    half_win = window_size / 2
    probs_smoothed = np.zeros(probs.shape)
    for i in xrange(len(probs_smoothed)):
        cls_window = cls_before_smooth[max(0, i - half_win):i + half_win]
        cl, cnts = np.unique(cls_window, return_counts=True)
        cl_won  = cl[np.argmax(cnts)]
        probs_smoothed[i, lst_all_classes.index(cl_won)] = 1.0
    return probs_smoothed






