
#%%
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
#import torchinfo


from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.metrics import cohen_kappa_score

import models 
from preprocess import get_data
#from keras.utils.vis_utils import plot_model
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

#%%
def draw_learning_curves(history, results_path):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'val'], loc='upper left')
    #plt.show()
    plt.savefig(results_path + "/validation_accuracy.png")
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'val'], loc='upper left')
    #plt.show()
    plt.savefig(results_path + "/validation_loss.png")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, sub, results_path, classes, normalize=True, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    accuracy = accuracy_score(y_true, y_pred)
    F1_note = f1_score(y_true, y_pred, average='macro')
    title = "Sub: " + str(sub) + " Macro F1: " + str(round(F1_note*100.0,2)) + "  " + "Accuracy: " + str(round(accuracy*100.0,2))
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=7)
    fig.tight_layout()
    #plt.savefig(confusion_img, dpi=500)
    plt.savefig( results_path + "/cm_subject_" + str(sub) + '.png')
    plt.close(fig)
    return ax


def draw_performance_barChart(num_sub, metric, label):
    fig, ax = plt.subplots()
    x = list(range(1, num_sub+1))
    ax.bar(x, metric, 0.5, label=label)
    ax.set_ylabel(label)
    ax.set_xlabel("Subject")
    ax.set_xticks(x)
    ax.set_title('Model '+ label + ' per subject')
    ax.set_ylim([0,1])
    

#######  sample weigth
def generate_sample_weights(training_data, class_weight_dictionary):
    sample_weights = [class_weight_dictionary[np.where(one_hot_row == 1)[0][0]] for one_hot_row in training_data]
    return np.asarray(sample_weights)

def scheduler(epoch, lr):
    decay_rate = 0.5  #
    decay_step = 200
    if epoch % decay_step == 0 and epoch:
        return lr * decay_rate
    return lr


#%% Training 
def train(dataset_conf, train_conf, results_path):
    # Get the current 'IN' time to calculate the overall training time
    in_exp = time.time()
    # Create a file to store the path of the best model among several runs
    best_models = open(results_path + "/best models.txt", "w")
    # Create a file to store performance during training
    log_write = open(results_path + "/log.txt", "w")
    # Create a .npz file (zipped archive) to store the accuracy and kappa metrics 
    # for all runs (to calculate average accuracy/kappa over all runs)
    perf_allRuns = open(results_path + "/perf_allRuns.npz", 'wb')
    
    # Get dataset paramters
    dataset = dataset_conf.get('name')
    n_sub = dataset_conf.get('n_sub')
    data_path = dataset_conf.get('data_path')
    # Get training hyperparamters
    batch_size = train_conf.get('batch_size')
    epochs = train_conf.get('epochs')
    lr = train_conf.get('lr')
    LearnCurves = train_conf.get('LearnCurves') # Plot Learning Curves?
    n_train = train_conf.get('n_train')

    # Initialize variables
    acc = np.zeros((n_sub, n_train))
    kappa = np.zeros((n_sub, n_train))

    for sub in range(0,n_sub):  # (num_sub): for all subjects, (i-1,i): for the ith subject.

        # Get the current 'IN' time to calculate the subject training time
        in_sub = time.time()
        print('\nTraining on subject ', sub+1)
        log_write.write( '\nTraining on subject '+ str(sub+1) +'\n')
        # Initiating variables to save the best subject accuracy among multiple runs.
        BestSubjAcc = 0 
        bestTrainingHistory = [] 
        # Get training and test data
        X_train, _, y_train_onehot, X_test, _, y_test_onehot = get_data(data_path, sub+1, dataset)
        
        # Iteration over multiple runs 
        for train in range(n_train): # How many repetitions of training for subject i.
            # Get the current 'IN' time to calculate the 'run' training time
            tf.random.set_seed(train+1)
            np.random.seed(train+1)

            in_run = time.time()
            # Create folders and files to save trained models for all runs
            filepath = results_path + '/saved models/run-{}'.format(train+1)
            if not os.path.exists(filepath):
                os.makedirs(filepath)        
            filepath = filepath + '/subject-{}.h5'.format(sub+1)
            
            # Create the model
            model = getModel(dataset_conf)

            # Compile and train the model
            model.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate=lr), metrics=['accuracy'])          
            model.summary()

            callbacks = [
                ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=True, mode='max'),
                LearningRateScheduler(scheduler)
            ]
            class_weight_dictionary = {0: 17.34, 1: 16.14, 2: 18.45, 3:18.89, 4: 15.50, 5: 1.0, 6: 9.79, 7: 46.91, 8: 12.4, 9: 10.2, 10: 9.14, 11: 9.1}
            history = model.fit(X_train, y_train_onehot, validation_data=(X_test, y_test_onehot), epochs=epochs, batch_size=batch_size, sample_weight=generate_sample_weights(y_train_onehot, class_weight_dictionary), callbacks=callbacks, verbose=1)

            # Evaluate the performance of the trained model. 
            # Here we load the Trained weights from the file saved in the hard 
            # disk, which should be the same as the weights of the current model.
            model.load_weights(filepath)
            y_pred = model.predict(X_test).argmax(axis=-1)
            labels = y_test_onehot.argmax(axis=-1)
            acc[sub, train]  = accuracy_score(labels, y_pred)
            kappa[sub, train] = cohen_kappa_score(labels, y_pred)
              
            # Get the current 'OUT' time to calculate the 'run' training time
            out_run = time.time()
            # Print & write performance measures for each run
            info = 'Subject: {}   Train no. {}   Time: {:.1f} m   '.format(sub+1, train+1, ((out_run-in_run)/60))
            info = info + 'Test_acc: {:.4f}   Test_kappa: {:.4f}'.format(acc[sub, train], kappa[sub, train])
            print(info)
            log_write.write(info +'\n')
            # If current training run is better than previous runs, save the history.
            if(BestSubjAcc < acc[sub, train]):
                 BestSubjAcc = acc[sub, train]
                 bestTrainingHistory = history
        
        # Store the path of the best model among several runs
        best_run = np.argmax(acc[sub,:])
        filepath = '/saved models/run-{}/subject-{}.h5'.format(best_run+1, sub+1)+'\n'
        best_models.write(filepath)
        # Get the current 'OUT' time to calculate the subject training time
        out_sub = time.time()
        # Print & write the best subject performance among multiple runs
        info = '----------\n'
        info = info + 'Subject: {}   best_run: {}   Time: {:.1f} m   '.format(sub+1, best_run+1, ((out_sub-in_sub)/60))
        info = info + 'acc: {:.4f}   avg_acc: {:.4f} +- {:.4f}   '.format(acc[sub, best_run], np.average(acc[sub, :]), acc[sub,:].std() )
        info = info + 'kappa: {:.4f}   avg_kappa: {:.4f} +- {:.4f}'.format(kappa[sub, best_run], np.average(kappa[sub, :]), kappa[sub,:].std())
        info = info + '\n----------'
        print(info)
        log_write.write(info+'\n')
        # Plot Learning curves 
        if (LearnCurves == True):
            print('Plot Learning Curves ....... ')
            draw_learning_curves(bestTrainingHistory, results_path)
          
    # Get the current 'OUT' time to calculate the overall training time
    out_exp = time.time()
    info = '\nTime: {:.1f} h   '.format( (out_exp-in_exp)/(60*60) )
    print(info)
    log_write.write(info+'\n')
    
    # Store the accuracy and kappa metrics as arrays for all runs into a .npz 
    # file format, which is an uncompressed zipped archive, to calculate average
    # accuracy/kappa over all runs.
    np.savez(perf_allRuns, acc = acc, kappa = kappa)
    
    # Close open files 
    best_models.close()   
    log_write.close() 
    perf_allRuns.close() 


#%% Evaluation 
def test(model, dataset_conf, results_path, allRuns = True):
    # Open the  "Log" file to write the evaluation results 
    log_write = open(results_path + "/log.txt", "a")
    # Open the file that stores the path of the best models among several random runs.
    best_models = open(results_path + "/best models.txt", "r")   
    
    # Get dataset paramters
    dataset = dataset_conf.get('name')
    n_classes = dataset_conf.get('n_classes')
    n_sub = dataset_conf.get('n_sub')
    data_path = dataset_conf.get('data_path')
    classes_labels = dataset_conf.get('cl_labels')
    
    # Initialize variables
    acc_bestRun = np.zeros(n_sub)
    kappa_bestRun = np.zeros(n_sub)  
    cf_matrix = np.zeros([n_sub, n_classes, n_classes])

    # Calculate the average performance (average accuracy and K-score) for 
    # all runs (experiments) for each subject.
    if(allRuns): 
        # Load the test accuracy and kappa metrics as arrays for all runs from a .npz 
        # file format, which is an uncompressed zipped archive, to calculate average
        # accuracy/kappa over all runs.
        perf_allRuns = open(results_path + "/perf_allRuns.npz", 'rb')
        perf_arrays = np.load(perf_allRuns)
        acc_allRuns = perf_arrays['acc']
        kappa_allRuns = perf_arrays['kappa']
    
    # Iteration over subjects 
    # for sub in range(n_sub-1, n_sub): # (num_sub): for all subjects, (i-1,i): for the ith subject.
    #for sub in range(n_sub): # (num_sub): for all subjects, (i-1,i): for the ith subject.
    labels_all = np.array([])
    predic_all = np.array([])
    #for sub in range(0, 2):  # (num_sub): for all subjects, (i-1,i): for the ith subject.
    for sub in range(0, n_sub):  # (num_sub): for all subjects, (i-1,i): for the ith subject.

        # Load data
        _, _, _, X_test, _, y_test_onehot = get_data(data_path, sub+1, dataset)
        
        # Load the best model out of multiple random runs (experiments).
        filepath = best_models.readline()
        model.load_weights(results_path + filepath[:-1])
        # Predict MI task
        y_pred = model.predict(X_test).argmax(axis=-1)
        # Calculate accuracy and K-score
        labels = y_test_onehot.argmax(axis=-1)
        acc_bestRun[sub] = accuracy_score(labels, y_pred)
        kappa_bestRun[sub] = cohen_kappa_score(labels, y_pred)
        # Calculate and draw confusion matrix
        cf_matrix[sub, :, :] = confusion_matrix(labels, y_pred, normalize='true')
        plot_confusion_matrix(labels, y_pred, sub+1, results_path, classes_labels)

        ## save labels and y_pred for post-processing
        np.save(results_path + "/test/Y_truth_Sub_" + str(sub+1) + ".npy", labels)
        np.save(results_path + "/test/Y_pred_Sub_" + str(sub+1) + ".npy", y_pred)
        labels_all = np.concatenate((labels_all, labels), axis = 0)
        predic_all = np.concatenate((predic_all, y_pred), axis = 0)

        # Print & write performance measures for each subject
        info = 'Subject: {}   best_run: {:2}  '.format(sub+1, (filepath[filepath.find('run-')+4:filepath.find('/sub')]) )
        info = info + 'acc: {:.4f}   kappa: {:.4f}   '.format(acc_bestRun[sub], kappa_bestRun[sub] )
        if(allRuns): 
            info = info + 'avg_acc: {:.4f} +- {:.4f}   avg_kappa: {:.4f} +- {:.4f}'.format(
                np.average(acc_allRuns[sub, :]), acc_allRuns[sub,:].std(),
                np.average(kappa_allRuns[sub, :]), kappa_allRuns[sub,:].std() )
        print(info)
        log_write.write('\n'+info)
      
    # Print & write the average performance measures for all subjects     
    info = '\nAverage of {} subjects - best runs:\nAccuracy = {:.4f}   Kappa = {:.4f}\n'.format(
        n_sub, np.average(acc_bestRun), np.average(kappa_bestRun)) 
    if(allRuns): 
        info = info + '\nAverage of {} subjects x {} runs (average of {} experiments):\nAccuracy = {:.4f}   Kappa = {:.4f}'.format(
            n_sub, acc_allRuns.shape[1], (n_sub * acc_allRuns.shape[1]),
            np.average(acc_allRuns), np.average(kappa_allRuns)) 
    print(info)
    log_write.write(info)
    
    # Draw a performance bar chart for all subjects 
    draw_performance_barChart(n_sub, acc_bestRun, 'Accuracy')
    draw_performance_barChart(n_sub, kappa_bestRun, 'K-score')
    # Draw confusion matrix for all subjects (average)
    plot_confusion_matrix(labels_all, predic_all, "All", results_path, classes_labels)
    # Close open files     
    log_write.close() 
    
    
#%%
def getModel(dataset_conf):
    
    n_classes = dataset_conf.get('n_classes')
    n_channels = dataset_conf.get('n_channels')
    in_samples = dataset_conf.get('in_samples')

    model = models.Post_Fusion(
        # Dataset parameters
        n_classes=n_classes,
        in_chans=n_channels,
        in_samples=in_samples,
        # Sliding window (SW) parameter
        n_windows=4,  ## 5 8
        # Convolutional (CV) block parameters
        F1=32,
        D=4,
        kernelSize=20,
        dropout=0.1,
        # Dilated block parameters
        di_kernelSize=3,
        di_filters=32,
        di_dropout=0.1,
        di_activation='elu'
    )
    return model
    
    
#%%
def run():
    # Define dataset parameters
    dataset = "Gym_Cap"

    in_samples = 80
    n_channels = 1
    n_sub = 2
    n_classes = 12
    classes_labels = ["Adductor", "ArmCurl", "BenchPress", "LegCurl", "LegPress", "Null", "Riding", "RopeSkipping", "Running", "Squat", "StairClimber", "Walking"]
    data_path = "/datasets/sibian/cap/Deep_wrist_All.csv"
    #data_path = "/datasets/sibian/cap/Deep_leg_All.csv"
    #data_path = "/datasets/sibian/cap/Deep_pocket_All.csv"

    # Create a folder to store the results of the experiment
    results_path = os.getcwd() + "/results"
    #results_path = os.getcwd() + "/results_imu"
    #results_path = os.getcwd() + "/results_cap"
    #results_path = os.getcwd() + "/results_leg"
    #results_path = os.getcwd() + "/results_leg_imu"
    #results_path = os.getcwd() + "/results_leg_cap"
    #results_path = os.getcwd() + "/results_pocket"
    #results_path = os.getcwd() + "/results_pocket_imu"
    #results_path = os.getcwd() + "/results_pocket_cap"

    if not  os.path.exists(results_path):
      os.makedirs(results_path)   # Create a new directory if it does not exist 
      
    # Set dataset paramters 
    dataset_conf = { 'name': dataset, 'n_classes': n_classes, 'cl_labels': classes_labels, 'n_sub': n_sub, 'n_channels': n_channels, 'in_samples': in_samples, 'data_path': data_path}
    # Set training hyperparamters
    train_conf = { 'batch_size': 256, 'epochs': 1000, 'lr': 0.0001, 'LearnCurves': True, 'n_train': 3}
           
    # Train the model
    train(dataset_conf, train_conf, results_path)

    # Evaluate the model based on the weights saved in the '/results' folder
    model = getModel(dataset_conf)
    test(model, dataset_conf, results_path)    
    
#%%
if __name__ == "__main__":
    run()
    
