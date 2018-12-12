import sys
from config import *
from sound_models import Sound_Models
from dataGenerator import audio_norm
import importlib
# importlib.reload(Sound_Models)
import os
import shutil
import importlib
from sklearn.model_selection import StratifiedKFold
from keras import losses, models, optimizers
from keras.activations import relu, softmax
from keras.callbacks import (EarlyStopping, LearningRateScheduler,
                             ModelCheckpoint, TensorBoard, ReduceLROnPlateau)
from keras.utils import Sequence, to_categorical
from keras.layers import (Convolution1D, Dense, Dropout, GlobalAveragePooling1D,
                          GlobalMaxPool1D, Input, MaxPool1D, concatenate)
from dataGenerator import DataGenerator
import pandas as pd



def init_config():
    print('initializing system parameters')
    max_epochs = None
    num_folds = None
    lr = None
    audio_duration = None
    for v in sys.argv:
        v = v.split('=')
        if 'max_epochs' in v[0]:
            max_epochs = int(v[1])
        if 'num_folds' in v[0]:
            num_folds = int(v[1])
        if 'lr' in v[0] or 'learning_rate' in v[0]:
            lr = float(v[1])
        if 'audio_duration' in v[0]:
            audio_duration = int(v[1])
    return max_epochs, num_folds, lr,audio_duration


def train_network(name, the_model, config, train, test, LABELS, num_cores):
    PREDICTION_FOLDER = 'models_for_top_five_cat' #"predictions_1d_conv_output_2"
    MODEL_PATH = PREDICTION_FOLDER + '/'+name+'/'
    if not os.path.exists(PREDICTION_FOLDER):
        os.mkdir(PREDICTION_FOLDER)
    if not os.path.exists(PREDICTION_FOLDER+'/'+name):
        os.mkdir(PREDICTION_FOLDER+'/'+name)
    if os.path.exists('../logs/' + PREDICTION_FOLDER+name+'/'):
        shutil.rmtree('../logs/' + PREDICTION_FOLDER+name+'/')
    skf = StratifiedKFold(n_splits=config.n_folds)
    splits = skf.split(train.label, train.actual_label_idx)
    fold_acc = []
    for i, (train_split, val_split) in enumerate(splits):
        train_set = train.iloc[train_split]
        val_set = train.iloc[val_split]
        checkpoint = ModelCheckpoint(MODEL_PATH + 'best_%d.h5' % i, monitor='val_loss', verbose=1, save_best_only=True)
        early = EarlyStopping(monitor="val_loss", mode="min", patience=3)
        tb = TensorBoard(log_dir='./logs/' + name +'/'+ '/fold_%d' % i, write_graph=True)
        callbacks_list = [checkpoint, early, tb]
        print("Fold: ", i)
        print("#" * 50)
        model = the_model(config)
        train_generator = DataGenerator(config, 'audio_train/', train_set.index,
                                        train_set.actual_label_idx, batch_size=64,
                                        preprocessing_fn=audio_norm)
        val_generator = DataGenerator(config, 'audio_train/', val_set.index,
                                      val_set.actual_label_idx, batch_size=64,
                                      preprocessing_fn=audio_norm)

        history = model.fit_generator(train_generator, callbacks=callbacks_list, validation_data=val_generator,
                                      epochs=config.max_epochs, use_multiprocessing=True, workers=num_cores, max_queue_size=20)

        model.load_weights(MODEL_PATH+'best_%d.h5' % i)

        # Save train predictions
        train_generator = DataGenerator(config, 'audio_train/', train.index, batch_size=128,
                                        preprocessing_fn=audio_norm)
        predictions = model.predict_generator(train_generator, use_multiprocessing=True,
                                              workers=num_cores, max_queue_size=20, verbose=1)
        np.save(MODEL_PATH+ "train_predictions_%d.npy" % i, predictions)

        # Save test predictions
        test_generator = DataGenerator(config, 'audio_train/', test.index, batch_size=128,
                                       preprocessing_fn=audio_norm)
        predictions = model.predict_generator(test_generator, use_multiprocessing=True,
                                              workers=num_cores, max_queue_size=20, verbose=1)
        np.save(MODEL_PATH + "test_predictions_%d.npy" % i, predictions)
        # Make a submission file
        top_3 = np.array(LABELS)[np.argsort(-predictions, axis=1)[:, :1]]
        predicted_labels = [' '.join(list(x)) for x in top_3]
        test['predicted_label'] = predicted_labels
        num_wrong = 0
        for entry in test.values:
            if (entry[2] != entry[0]):
                num_wrong += 1
        fold_acc.append((1 - (num_wrong / test.shape[0])) * 100)

    #     test[['pred_label']].to_csv(PREDICTION_FOLDER + "/predictions_%d.csv"%i)
    f = open(MODEL_PATH + 'fold_accuracies.csv', 'a+')
    for i in range(len(fold_acc)):
        val = "Fold_" + str(i) + ", " + str(fold_acc[i]) + '\n'
        f.write(val)
    val = "Avg_accuracy, " + str(sum(fold_acc) / len(fold_acc))
    f.write(val)
    f.close()
    f = open(MODEL_PATH+'fold_accuracies.csv', 'a+')
    for i in range(len(fold_acc)):
        val = "Fold_"+str(i)+", "+str(fold_acc[i])+'\n'
        f.write(val)
    val = "Avg_accuracy, "+str(sum(fold_acc)/len(fold_acc))
    f.write(val)
    f.close()

def main():
    num_cores = 11
    max_epochs,num_folds,lr, audio_duration = init_config()
    if max_epochs is None:
        max_epochs = 150
    if num_folds is None:
        num_folds = 2
    if lr is None:
        lr = .001
    if audio_duration is None:
        audio_duration = 2


    config = Config(sampling_rate=16000, audio_duration=audio_duration, n_folds=num_folds, learning_rate=lr)
    models_helper = Sound_Models()
    lst_of_networks = models_helper.get_models()
    network_names = models_helper.get_names()

    LABELS = ['Violin_or_fiddle', 'Saxophone', 'Gunshot_or_gunfire', 'Clarinet', 'Flute']

    train_df = pd.read_csv('train.csv')
    total = train_df.loc[train_df.manually_verified == 1, :]
    train_split = .7
    # total = manually_verified.copy()
    # test = test_df.copy()
    total = total.loc[total.label.isin(LABELS), :]
    total = total.sample(frac=1)
    train = total.iloc[0:int(len(total) * train_split)]
    test = total.iloc[int(len(total) * train_split):]
    label_idx = {label: i for i, label in enumerate(LABELS)}

    train.set_index('fname', inplace=True)
    test.set_index("fname", inplace=True)
    train["actual_label_idx"] = train.label.apply(lambda x: label_idx[x])

    lst_learning_rates = [lr, lr/10,lr*10]
    # lst_audio_length = [audio_duration+1, audio_duration-1, audio_duration+2]

    for ad in lst_learning_rates:
        for i in range(len(lst_of_networks)):
            print('--------------Starting training ', network_names[i]+'lr_'+str(ad))
            try:
                train_network(network_names[i]+'lr_'+str(ad),lst_of_networks[i],config, train, test, LABELS, num_cores)
                print('################################################################################\n'*20)
            except:
                print('#$'*20)
                print('ERROR with '+str(network_names[i])+'lr_'+str(ad))

if __name__ == '__main__':
    main()