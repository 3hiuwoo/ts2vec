import os
import numpy as np
import pandas as pd
import math
import random
from datetime import datetime
import pickle
from utils import pkl_load, pad_nan_to_target
from scipy.io.arff import loadarff
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import shuffle
from scipy.signal import butter, lfilter
from tqdm import tqdm
from itertools import repeat

def load_UCR(dataset):
    train_file = os.path.join('datasets/UCR', dataset, dataset + "_TRAIN.tsv")
    test_file = os.path.join('datasets/UCR', dataset, dataset + "_TEST.tsv")
    train_df = pd.read_csv(train_file, sep='\t', header=None)
    test_df = pd.read_csv(test_file, sep='\t', header=None)
    train_array = np.array(train_df)
    test_array = np.array(test_df)

    # Move the labels to {0, ..., L-1}
    labels = np.unique(train_array[:, 0])
    transform = {}
    for i, l in enumerate(labels):
        transform[l] = i

    train = train_array[:, 1:].astype(np.float64)
    train_labels = np.vectorize(transform.get)(train_array[:, 0])
    test = test_array[:, 1:].astype(np.float64)
    test_labels = np.vectorize(transform.get)(test_array[:, 0])

    # Normalization for non-normalized datasets
    # To keep the amplitude information, we do not normalize values over
    # individual time series, but on the whole dataset
    if dataset not in [
        'AllGestureWiimoteX',
        'AllGestureWiimoteY',
        'AllGestureWiimoteZ',
        'BME',
        'Chinatown',
        'Crop',
        'EOGHorizontalSignal',
        'EOGVerticalSignal',
        'Fungi',
        'GestureMidAirD1',
        'GestureMidAirD2',
        'GestureMidAirD3',
        'GesturePebbleZ1',
        'GesturePebbleZ2',
        'GunPointAgeSpan',
        'GunPointMaleVersusFemale',
        'GunPointOldVersusYoung',
        'HouseTwenty',
        'InsectEPGRegularTrain',
        'InsectEPGSmallTrain',
        'MelbournePedestrian',
        'PickupGestureWiimoteZ',
        'PigAirwayPressure',
        'PigArtPressure',
        'PigCVP',
        'PLAID',
        'PowerCons',
        'Rock',
        'SemgHandGenderCh2',
        'SemgHandMovementCh2',
        'SemgHandSubjectCh2',
        'ShakeGestureWiimoteZ',
        'SmoothSubspace',
        'UMD'
    ]:
        return train[..., np.newaxis], train_labels, test[..., np.newaxis], test_labels
    
    mean = np.nanmean(train)
    std = np.nanstd(train)
    train = (train - mean) / std
    test = (test - mean) / std
    return train[..., np.newaxis], train_labels, test[..., np.newaxis], test_labels


def load_UEA(dataset):
    train_data = loadarff(f'datasets/UEA/{dataset}/{dataset}_TRAIN.arff')[0]
    test_data = loadarff(f'datasets/UEA/{dataset}/{dataset}_TEST.arff')[0]
    
    def extract_data(data):
        res_data = []
        res_labels = []
        for t_data, t_label in data:
            t_data = np.array([ d.tolist() for d in t_data ])
            t_label = t_label.decode("utf-8")
            res_data.append(t_data)
            res_labels.append(t_label)
        return np.array(res_data).swapaxes(1, 2), np.array(res_labels)
    
    train_X, train_y = extract_data(train_data)
    test_X, test_y = extract_data(test_data)
    
    scaler = StandardScaler()
    scaler.fit(train_X.reshape(-1, train_X.shape[-1]))
    train_X = scaler.transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
    test_X = scaler.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)
    
    labels = np.unique(train_y)
    transform = { k : i for i, k in enumerate(labels)}
    train_y = np.vectorize(transform.get)(train_y)
    test_y = np.vectorize(transform.get)(test_y)
    return train_X, train_y, test_X, test_y
    
    
def load_forecast_npy(name, univar=False):
    data = np.load(f'datasets/{name}.npy')    
    if univar:
        data = data[: -1:]
        
    train_slice = slice(None, int(0.6 * len(data)))
    valid_slice = slice(int(0.6 * len(data)), int(0.8 * len(data)))
    test_slice = slice(int(0.8 * len(data)), None)
    
    scaler = StandardScaler().fit(data[train_slice])
    data = scaler.transform(data)
    data = np.expand_dims(data, 0)

    pred_lens = [24, 48, 96, 288, 672]
    return data, train_slice, valid_slice, test_slice, scaler, pred_lens, 0


def _get_time_features(dt):
    return np.stack([
        dt.minute.to_numpy(),
        dt.hour.to_numpy(),
        dt.dayofweek.to_numpy(),
        dt.day.to_numpy(),
        dt.dayofyear.to_numpy(),
        dt.month.to_numpy(),
        dt.weekofyear.to_numpy(),
    ], axis=1).astype(np.float)


def load_forecast_csv(name, univar=False):
    data = pd.read_csv(f'datasets/{name}.csv', index_col='date', parse_dates=True)
    dt_embed = _get_time_features(data.index)
    n_covariate_cols = dt_embed.shape[-1]
    
    if univar:
        if name in ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'):
            data = data[['OT']]
        elif name == 'electricity':
            data = data[['MT_001']]
        else:
            data = data.iloc[:, -1:]
        
    data = data.to_numpy()
    if name == 'ETTh1' or name == 'ETTh2':
        train_slice = slice(None, 12*30*24)
        valid_slice = slice(12*30*24, 16*30*24)
        test_slice = slice(16*30*24, 20*30*24)
    elif name == 'ETTm1' or name == 'ETTm2':
        train_slice = slice(None, 12*30*24*4)
        valid_slice = slice(12*30*24*4, 16*30*24*4)
        test_slice = slice(16*30*24*4, 20*30*24*4)
    else:
        train_slice = slice(None, int(0.6 * len(data)))
        valid_slice = slice(int(0.6 * len(data)), int(0.8 * len(data)))
        test_slice = slice(int(0.8 * len(data)), None)
    
    scaler = StandardScaler().fit(data[train_slice])
    data = scaler.transform(data)
    if name in ('electricity'):
        data = np.expand_dims(data.T, -1)  # Each variable is an instance rather than a feature
    else:
        data = np.expand_dims(data, 0)
    
    if n_covariate_cols > 0:
        dt_scaler = StandardScaler().fit(dt_embed[train_slice])
        dt_embed = np.expand_dims(dt_scaler.transform(dt_embed), 0)
        data = np.concatenate([np.repeat(dt_embed, data.shape[0], axis=0), data], axis=-1)
    
    if name in ('ETTh1', 'ETTh2', 'electricity'):
        pred_lens = [24, 48, 168, 336, 720]
    else:
        pred_lens = [24, 48, 96, 288, 672]
        
    return data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols


def load_anomaly(name):
    res = pkl_load(f'datasets/{name}.pkl')
    return res['all_train_data'], res['all_train_labels'], res['all_train_timestamps'], \
           res['all_test_data'],  res['all_test_labels'],  res['all_test_timestamps'], \
           res['delay']


def gen_ano_train_data(all_train_data):
    maxl = np.max([ len(all_train_data[k]) for k in all_train_data ])
    pretrain_data = []
    for k in all_train_data:
        train_data = pad_nan_to_target(all_train_data[k], maxl, axis=0)
        pretrain_data.append(train_data)
    pretrain_data = np.expand_dims(np.stack(pretrain_data), 2)
    return pretrain_data


def load_data(root='../MCP/dataset', name='chapman', length=None, overlap=0, norm=True, shuff=True):
    '''
    load and preprocess data
    '''
    data_path = os.path.join(root, name, 'feature')
    labels, train_ids, valid_ids, test_ids = load_label_split(root, name)
    
    filenames = []
    for fn in os.listdir(data_path):
        filenames.append(fn)
    filenames.sort()
    
    train_trials = []
    train_labels = []
    valid_trials = []
    valid_labels = []
    test_trials = []
    test_labels = []
    
    for i, fn in enumerate(tqdm(filenames, desc=f'=> Loading {name}')):
        label = labels[i]
        feature = np.load(os.path.join(data_path, fn))
        for trial in feature:
            if i+1 in train_ids:
                train_trials.append(trial)
                train_labels.append(label)
            elif i+1 in valid_ids:
                valid_trials.append(trial)
                valid_labels.append(label)
            elif i+1 in test_ids:
                test_trials.append(trial)
                test_labels.append(label)
                
    X_train = np.array(train_trials)
    X_val = np.array(valid_trials)
    X_test = np.array(test_trials)
    y_train = np.array(train_labels)
    y_val = np.array(valid_labels)
    y_test = np.array(test_labels)
    
    if shuff:
        X_train, y_train = shuffle(X_train, y_train)
        X_val, y_val = shuffle(X_val, y_val)
        X_test, y_test = shuffle(X_test, y_test)
    
    if norm:
        X_train = process_batch_ts(X_train, normalized=True, bandpass_filter=False)
        X_val = process_batch_ts(X_val, normalized=True, bandpass_filter=False)
        X_test = process_batch_ts(X_test, normalized=True, bandpass_filter=False)
      
    if length:
        # X_train, y_train = segment(X_train, y_train, split)
        # X_val, y_val = segment(X_val, y_val, split)
        # X_test, y_test = segment(X_test, y_test, split)
        
        X_train, y_train = split_data_label(X_train, y_train, sample_timestamps=length, overlapping=overlap)
        X_val, y_val = split_data_label(X_val, y_val, sample_timestamps=length, overlapping=overlap)
        X_test, y_test = split_data_label(X_test, y_test, sample_timestamps=length, overlapping=overlap)
        
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def load_label_split(root='dataset', name='chapman'):
    '''
    load labels for dataset and split information
    '''
    label_path = os.path.join(root, name, 'label', 'label.npy')
    labels = np.load(label_path)
    
    if name == 'chapman':
        pids_sb = list(labels[np.where(labels[:, 0]==0)][:, 1])
        pids_af = list(labels[np.where(labels[:, 0]==1)][:, 1])
        pids_gsvt = list(labels[np.where(labels[:, 0]==2)][:, 1])
        pids_sr = list(labels[np.where(labels[:, 0]==3)][:, 1])
        
        train_ids = pids_sb[:-500] + pids_af[:-500] + pids_gsvt[:-500] + pids_sr[:-500]
        val_ids = pids_sb[-500:-250] + pids_af[-500:-250] + pids_gsvt[-500:-250] + pids_sr[-500:-250]
        test_ids = pids_sb[-250:] + pids_af[-250:] + pids_gsvt[-250:] + pids_sr[-250:]
        
    elif name == 'ptb':
        pids_neg = list(labels[np.where(labels[:, 0]==0)][:, 1])
        pids_pos = list(labels[np.where(labels[:, 0]==1)][:, 1])
        
        train_ids = pids_neg[:-14] + pids_pos[:-42]  # specify patient ID for training, validation, and test set
        val_ids = pids_neg[-14:-7] + pids_pos[-42:-21]   # 28 patients, 7 healthy and 21 positive
        test_ids = pids_neg[-7:] + pids_pos[-21:]  # # 28 patients, 7 healthy and 21 positive
        
    elif name == 'ptbxl':
        pids_norm = list(labels[np.where(labels[:, 0]==0)][:, 1])
        pids_mi = list(labels[np.where(labels[:, 0]==1)][:, 1])
        pids_sttc = list(labels[np.where(labels[:, 0]==2)][:, 1])
        pids_cd = list(labels[np.where(labels[:, 0]==3)][:, 1])
        pids_hyp = list(labels[np.where(labels[:, 0]==3)][:, 1])
        
        train_ids = pids_norm[:-1200] + pids_mi[:-600] + pids_sttc[:-600] + pids_cd[:-400] + pids_hyp[:-200]
        val_ids = pids_norm[-1200:-600] + pids_mi[-600:-300] + pids_sttc[-600:-300] + pids_cd[-400:-200] + pids_hyp[-200:-100]
        test_ids = pids_norm[-600:] + pids_mi[-300:] + pids_sttc[-300:] + pids_cd[-200:] + pids_hyp[-100:]
        
    else:
        raise ValueError(f'Unknown dataset: {name}')
        
    return labels, train_ids, val_ids, test_ids


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    ''' see https://stackoverflow.com/questions/12093594/how-to-implement-band-pass-butterworth-filter-with-scipy-signal-butter

    '''
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data, axis=0)
    return y


def process_ts(ts, fs, normalized=True, bandpass_filter=False):
    ''' preprocess a time-series data

    Args:
        ts (numpy.ndarray): The input time-series in shape (timestamps, feature).
        fs (float): The sampling frequency for bandpass filtering.
        normalized (bool): Whether to normalize the time-series data.
        bandpass_filter (bool): Whether to filter the time-series data.

    Returns:
        ts (numpy.ndarray): The processed time-series.
    '''

    if bandpass_filter:
        ts = butter_bandpass_filter(ts, 0.5, 50, fs, 5)
    if normalized:
        scaler = StandardScaler()
        scaler.fit(ts)
        ts = scaler.transform(ts)
    return ts


def process_batch_ts(batch, fs=256, normalized=True, bandpass_filter=False):
    ''' preprocess a batch of time-series data

    Args:
        batch (numpy.ndarray): A batch of input time-series in shape (n_samples, timestamps, feature).

    Returns:
        A batch of processed time-series.
    '''

    bool_iterator_1 = repeat(fs, len(batch))
    bool_iterator_2 = repeat(normalized, len(batch))
    bool_iterator_3 = repeat(bandpass_filter, len(batch))
    return np.array(list(map(process_ts, batch, bool_iterator_1, bool_iterator_2, bool_iterator_3)))


def split_data_label(X_trial, y_trial, sample_timestamps, overlapping):
    ''' split a batch of time-series trials into samples and adding trial ids to the label array y

    Args:
        X_trial (numpy.ndarray): It should have a shape of (n_trials, trial_timestamps, features) B_trial x T_trial x C.
        y_trial (numpy.ndarray): It should have a shape of (n_trials, 2). The first column is the label and the second column is patient id.
        sample_timestamps (int): The length for sample-level data (T_sample).
        overlapping (float): How many overlapping for each sample-level data in a trial.

    Returns:
        X_sample (numpy.ndarray): It should have a shape of (n_samples, sample_timestamps, features) B_sample x T_sample x C. The B_sample = B x sample_num.
        y_sample (numpy.ndarray): It should have a shape of (n_samples, 3). The three columns are the label, patient id, and trial id.
    '''
    X_sample, trial_ids, sample_num = split_data(X_trial, sample_timestamps, overlapping)
    # all samples from same trial should have same label and patient id
    y_sample = np.repeat(y_trial, repeats=sample_num, axis=0)
    # append trial ids. Segments split from same trial should have same trial ids
    label_num = y_sample.shape[0]
    y_sample = np.hstack((y_sample.reshape((label_num, -1)), trial_ids.reshape((label_num, -1))))
    X_sample, y_sample = shuffle(X_sample, y_sample, random_state=42)
    return X_sample, y_sample


def split_data(X_trial, sample_timestamps=256, overlapping=0.5):
    ''' split a batch of trials into samples and mark their trial ids

    Args:
        See split_data_label() function

    Returns:
        X_sample (numpy.ndarray): (n_samples, sample_timestamps, feature).
        trial_ids (numpy.ndarray): (n_samples,)
        sample_num (int): one trial splits into sample_num of samples
    '''
    length = X_trial.shape[1]
    # check if sub_length and overlapping compatible
    if overlapping:
        assert (length - (1-overlapping)*sample_timestamps) % (sample_timestamps*overlapping) == 0
        sample_num = (length - (1 - overlapping) * sample_timestamps) / (sample_timestamps * overlapping)
    else:
        assert length % sample_timestamps == 0
        sample_num = length / sample_timestamps
    sample_feature_list = []
    trial_id_list = []
    trial_id = 1
    for trial in X_trial:
        counter = 0
        # ex. split one trial(5s, 1280 timestamps) into 9 half-overlapping samples (1s, 256 timestamps)
        while counter*sample_timestamps*(1-overlapping)+sample_timestamps <= trial.shape[0]:
            sample_feature = trial[int(counter*sample_timestamps*(1-overlapping)):int(counter*sample_timestamps*(1-overlapping)+sample_timestamps)]
            # print(f"{int(counter*length*(1-overlapping))}:{int(counter*length*(1-overlapping)+length)}")
            sample_feature_list.append(sample_feature)
            trial_id_list.append(trial_id)
            counter += 1
        trial_id += 1
    X_sample, trial_ids = np.array(sample_feature_list), np.array(trial_id_list)

    return X_sample, trial_ids, sample_num
