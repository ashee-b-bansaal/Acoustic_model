# python emo_test_cnn.py --dataset_path /data/asl_test -poi 250,600 -g 0 --target_height 330 --folder_name all_signs --batch 10 --test_sessions 0901 --exclude_sessions 0101,0201

import os
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
import cv2
import logging
import random

from torch.utils.data import DataLoader
#from .dataset import CNNDataset, DataBatches
from copy import deepcopy
from torch import Tensor
from math import ceil

import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

import argparse
import logging


parser = argparse.ArgumentParser(description='Conditions')
parser.add_argument('--dataset_path', default='', type=str, help='dataset')
parser.add_argument('-poi','--point_of_interest', default='0,600', type=str, help='point of interest')
parser.add_argument('-g','--gpu_num', default=0, type=int, help='gpus')
parser.add_argument('--target_height', default=80, type=int, help='target')
parser.add_argument('--epoch', default=100, type=int, help='epoch')
parser.add_argument('-rt','--retraining', default=False, type=bool, help='retraining')
parser.add_argument('--exclude_sessions', default='', type=str, help='exclude-session')
parser.add_argument('--test_sessions', default='', type=str, help='test-session')
parser.add_argument('--folder_name', default='', type=str, help='folder_name')
parser.add_argument('--batch', default=5, type=int, help='batch')
parser.add_argument('--resume', default='', type=str, help='Path to checkpoint to resume from')


args = parser.parse_args()

def ensure_folder_exists(folder_path):
    """Check if a folder exists, and create it if not."""
    if not os.path.exists(folder_path):  # Check if folder exists
        os.makedirs(folder_path)  # Create folder (including parent directories if needed)
        print(f"✅ Folder created: {folder_path}")
    else:
        print(f"📂 Folder already exists: {folder_path}")



# Example usage
retraining = args.retraining
num_epochs = args.epoch
target_height = args.target_height
gpu_set = args.gpu_num
dataset_folder = args.dataset_path
poi = args.point_of_interest
poi_list = poi.split(',')
exclude_sessions = args.exclude_sessions
test_sessions = args.test_sessions
folder_nm = args.folder_name
batch_size = args.batch


test_sessions = [i for i in test_sessions.split(',')]

print(test_sessions)
fusion = False
imu_1d = False
only_imu = False

input_channel_slice = [3] #use channel 4 for acoustic data
input_channel = len(input_channel_slice)


folder = dataset_folder.split('/dataset/')[0]+'_poi_%s_%s'%(poi_list[0],poi_list[1])+'_th_%s'%(target_height)+'ch%s'%input_channel + '_fusion_%s'%folder_nm

best_save_path = "./experiments/%s/"%(folder)
ensure_folder_exists(best_save_path)



# Set up logging configuration once at the beginning
def setup_logging(log_file_path):
    """Set up logging configuration once."""
    # Clear any existing handlers to avoid duplicates
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Configure logging
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),  # Print to console
            logging.FileHandler(log_file_path, mode='a')  # Log to file (append mode)
        ]
    )
    return logging.getLogger(__name__)

# Initialize logger
logger = setup_logging(best_save_path + "logfile.txt")

def print_and_log(message, log_file = best_save_path + "logfile.txt"):
    """Print message to console and log it to a .txt file."""
    # Print message to console and log it
    print(message)  # Print to console
    logger.info(message)  # Log the message

# Log script start
print_and_log("="*50)
print_and_log("Training script started")
print_and_log(f"Experiment folder: {best_save_path}")
print_and_log("="*50)

# lst = ["WH", "YN", "RQ", "CD", "NG", "RC", "TP", "AF", 
#        "Happy", "Sad", "Anger", "Fear", "Surprise", "Disgust", 
#        "MM", "CS", "TH", "INTENSE", "PUFF", "PS", "OO", "CHA"]

# lst = ["WH", "YN", "CD", "NG", "AF", 
#        "Happy", "Sad", "Anger", "Fear", "Surprise", "Disgust", 
#        "MM", "CS", "TH", "INTENSE", "PUFF", "PS", "OO", "CHA"]
    
lst = [
    'tomorrow',
    'not',
    'alone',
    'always',
    'only_just',
    'first',
    'lonely',
    'single',
    'dessert',
    'date(romantic_outing)',
    'business',
    'busy',
    'movie',
    'pet',
    'warning',
    'summon',
    'live',
    'address',
    'share',
    'bath',
    'music',
    'bread',
    'wood',
    'medium_average',
    'money',
    'buy',
    'shopping',
    'sell',
    'store',
    'chocolate',
    'church',
    'computer',
    'worried',
    'frustrated',
    'no',
    '30',
    'cleaning',
    'nice',
    'shool',
    'paper',
    'gym',
    'celebrate',
    'of_course',
    'gain',
    'normal',
    'tuesday',
    'bathroom',
    'science',
    'show',
    'bicycle',
    'dry',
    'ugly',
    'summer',
    'black',
    'gray',
    'doesn\'t_matter',
    'football',
    'wrestling',
    'roomate',
    'machine_factory',
    'lucky',
    'favorite',
    'pig',
    'dirty',
    'cereal',
    'metal',
    'sneakers',
    'sex',
    'rubber',
    'family',
    'important',
    'onion',
    'apple',
    'scared',
    'breath',
    'class',
    'late',
    'not_yet',
    'where',
    'red',
    'sweet',
    'please',
    'sorry',
    'your',
    'mine',
    'doctor',
    'again',
    'discuss',
    'sometime',
    'read',
    'dance',
    'stay',
    'today',
    'favorite',
    'birthday',
    'frog',
    'sunday',
    'wonderful',
    'sorrowful',
    'cry',
    'good',
    'bad',
    'mean(cruel)',
    'punish',
    'star',
    'socks',
    'weekend',
    'sick',
    'shave',
    'each_other',
    'sneakers_rubber'
]

label_dic =  {value: index for index, value in enumerate(lst)}
label_dic_reverse = {index: value for index, value in enumerate(lst)}
class_num = len(lst)
print(class_num)
print(label_dic)
print(label_dic_reverse)

dp = dataset_folder+'/dataset/'
tmp_dp = sorted(os.listdir(dp))
train_sessions = [i.split('_')[1] for i in tmp_dp if i.find('session') == 0]

data_path = dp + 'session_'
#train_sessions = ['0101','0201','0301','0401','0501', '0601', '0701','0801','0901']
#train_sessions.remove(test_sessions[0])

if len(exclude_sessions)>0:
    ex_tmp = [i for i in exclude_sessions.split(',')]
    for i in range(0, len(ex_tmp)):
        train_sessions.remove(ex_tmp[i])

# Clean up test_sessions list and remove 'session_' prefix if present
test_sessions = [s.replace('session_', '') for s in test_sessions if s.strip() != '']
# Remove test sessions from training sessions
train_sessions = [s for s in train_sessions if s not in test_sessions]

print_and_log('test_sessions: ' + ",".join(test_sessions))
print_and_log('train_sessions: ' + ",".join(train_sessions))

# label_dic = {'CS':0,
#  'INTENSE':1,
#  'MM':2,
#  'PS':3,
#  'PUFF':4,
#  'TH':5,
#  'angry':6,
#  'disgusted':7,
#  'happy':8,
#  'sad':9,
#  'surprised':10,
#  'terrified':11}

# label_dic_reverse = {0:'CS',
# 1:'INTENSE',
# 2: 'MM',
# 3: 'PS',
# 4: 'PUFF',
# 5: 'TH',
# 6: 'angry',
# 7: 'disgusted',
# 8: 'happy',
# 9: 'sad',
# 10: 'surprised',
# 11: 'terrified'}

def save_checkpoint(model, optimizer, epoch, best_acc=0.0, filename= best_save_path + "best_checkpoint.pth"):
    """Save model, optimizer, epoch number, and best accuracy."""
    checkpoint = {
        "epoch": epoch,  
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_accuracy": best_acc
    }
    torch.save(checkpoint, filename)
    print_and_log(f"✅ Checkpoint saved at epoch {epoch+1} with best accuracy: {best_acc:.2f}%")



print_and_log("This is a log message.")



def collate_various_size(batch):

    data_list_arr = [x[0][0] for x in batch]
    data_list_imu = [x[0][1] for x in batch]
    target = [x[1] for x in batch]
    data_max_size = max([x.shape[1] for x in data_list_arr])
    
    # check the windown size, for example, if windion size 10, the target size should be dividied by windon size. 
    #target_length = ceil(target_length / 16) * 16
    window_size = 10
    target_length = data_max_size 
    target_length = ceil(target_length / window_size) * window_size
   
    #data_list_imu = [x[0][1].reshape(1, x[0][1].shape[0], x[0][1].shape[1]) for x in batch]

    data_arr = np.zeros((len(batch), data_list_arr[0].shape[0], target_length, data_list_arr[0].shape[2]))
    data_imu = np.zeros((len(batch), data_list_imu[0].shape[0], target_length, data_list_imu[0].shape[2]))
    
    # horizontal shifting time axis. 
    for i in range(0, len(data_list_arr)):
        start_x = random.randint(0, target_length - data_list_arr[i].shape[1])
        data_arr[i, :, start_x: start_x + data_list_arr[i].shape[1], :] = data_list_arr[i]
        data_imu[i, :, start_x: start_x + data_list_imu[i].shape[1], :] = data_list_imu[i]

    # data1 = Tensor(data_arr)
    # data2 = Tensor(data_imu)
    # return (data1, data2), target
    data_arr = data_arr.swapaxes(2,3) # C, H (spatial height), W (temporal dimension, e.g., time steps)
    data_imu = data_imu.swapaxes(2,3)
        
    return (data_arr, data_imu), target

class DataSplitter:
    train_loader: DataLoader
    val_loader: DataLoader

    def __init__(self, train_data, test_data, BATCH_SIZE, WORKER_NUM):
        train_data = train_data
        val_data = test_data

        print_and_log('train length: ' + str(len(train_data)))
        print_and_log('test length: ' + str(len(val_data)))
        # convert to 'Dataloader'
        if len(train_data):
            train_dataset = CNNDataset(train_data, is_train=True)
            self.train_loader = DataLoader(
                train_dataset,
                # batch_size=BATCH_SIZE,
                # shuffle=shuffle,
                # drop_last=True,
                num_workers=WORKER_NUM,
                collate_fn=collate_various_size,
                batch_sampler=DataBatches(len(train_dataset), BATCH_SIZE)
            )
        else:
            self.train_loader = None
        test_dataset = CNNDataset(val_data, is_train=False)
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=WORKER_NUM,
            collate_fn=collate_various_size,
            # batch_sampler=DataBatches(len(test_dataset), BATCH_SIZE)
        )
        
class CNNDataset(torch.utils.data.Dataset):

    def __init__(self, data, is_train):
        self.data = data
        self.is_train = is_train
        
    def __getitem__(self, index):
        input_arr = self.data[index][0]
        # input_imu = self.data[index][3]
        input_imu = self.data[index][2] #now 2 bcuz removed echo_org

        output_arr = deepcopy(self.data[index][1]) #now 1 bcuz removed echo_org

        input_arr_copy = deepcopy(input_arr)
        input_imu_copy = deepcopy(input_imu)

        aug_arr = input_arr_copy
        aug_imu = input_imu_copy
        #print(aug_arr.shape, aug_imu.shape)

        if is_train:
            if (random.random() > 0.2):
                mask_width = random.randint(10, 20)
                rand_start = random.randint(0, aug_arr.shape[1] - mask_width)
                aug_arr[:, rand_start: rand_start + mask_width, :] = 0.0
                aug_imu[:, rand_start: rand_start + mask_width, :] = 0.0
            #print('mask')

        padded_input = aug_arr
        padded_imu = aug_imu

        if is_train:
            if random.random() > 0.2:
                noise_arr = np.random.random(padded_input.shape).astype(np.float32) * 0.1 + 0.95
                noise_imu = np.random.random(padded_imu.shape).astype(np.float32) * 0.1 + 0.95
                padded_input *= noise_arr
                padded_imu *= noise_imu
                #print('noise: ', noise_arr.shape, noise_imu.shape)

        padded_input_list = []
        
        for j in range(0, padded_input.shape[0]):
            padded_input_tmp = padded_input[j]
            #print(padded_input_tmp.shape)
            for c in range(padded_input_tmp.shape[0]):
                # instance-level norm
                mu, sigma = np.mean(padded_input_tmp[c]), np.std(padded_input_tmp[c])
                #print( mu, sigma)

                # avoid division by zero
                if sigma < 1e-8:
                    padded_input_tmp[c] = padded_input_tmp[c] - mu
                else:
                    padded_input_tmp[c] = (padded_input_tmp[c] - mu) / sigma

            padded_input_tmp = np.nan_to_num(padded_input_tmp, nan=0.0, posinf=0.0, neginf=0.0)
            padded_input_list.append(padded_input_tmp)
            #print(j, padded_input_tmp.shape)

        padded_input_fn = np.array(padded_input_list)
        padded_imu = np.nan_to_num(padded_imu, nan=0.0, posinf=0.0, neginf=0.0)

        # if poi
        padded_input_fn = padded_input_fn[:,:,int(poi_list[0]):int(poi_list[1])]
        poi_length = int(poi_list[1]) - int(poi_list[0])
        
        if is_train:
            target_height_start = random.randint(0, poi_length-target_height)
            target_height_end = target_height_start + target_height
            padded_input_fn = padded_input_fn[:,:,target_height_start:target_height_end]
        else:
            # test_dataset
            padded_input_fn = padded_input_fn[:,:,:target_height]

        return (padded_input_fn, padded_imu), output_arr

    def __len__(self):
        return len(self.data)
    
class DataBatches:

    def __init__(self, dataset_size, batch_size):
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.all_indices = self.batches()
        random.shuffle(self.all_indices)
        
    def __len__(self):
        return ceil(self.dataset_size / self.batch_size)

    def __iter__(self):
        for x in self.all_indices:
            yield x
        random.shuffle(self.all_indices)

    def batches(self):
        all_indices = []
        for i in range(0, self.dataset_size, self.batch_size):
            all_indices += [list(range(i, min(i + self.batch_size, self.dataset_size)))]
        return all_indices



def upsample_imu_data(time, imu_data, target_num_samples):
    """
    Upsample IMU data to a target number of samples.

    Parameters:
    - time: 1D array, timestamps of the original IMU data.
    - imu_data: 2D array, IMU data (e.g., acceleration, angular velocity).
    - target_num_samples: desired number of samples after upsampling.

    Returns:
    - upsampled_time: 1D array, timestamps of the upsampled data.
    - upsampled_imu_data: 2D array, upsampled IMU data.
    """
    # Ensure time values are strictly increasing and remove duplicates
    unique_time, unique_idx = np.unique(time, return_index=True)
    sorted_idx = np.argsort(unique_time)
    unique_time = unique_time[sorted_idx]
    unique_idx = unique_idx[sorted_idx]

    # Sort imu_data based on unique_time
    sorted_imu_data = imu_data[unique_idx]

    # Create an interpolation function for each dimension of the IMU data
    interp_functions = [CubicSpline(unique_time, sorted_imu_data[:, i]) for i in range(sorted_imu_data.shape[1])]
    #interp_functions = [CubicSpline(unique_time, sorted_imu_data[:, i]) for i in range(sorted_imu_data.shape[1])]

    # Create upsampled time array
    upsampled_time = np.linspace(unique_time[0], unique_time[-1], target_num_samples)

    # Interpolate IMU data at upsampled time points
    upsampled_imu_data = np.column_stack([f(upsampled_time) for f in interp_functions])

    return upsampled_time, upsampled_imu_data

def normalize_imu_data(upsampled_imu_data):
    """
    Normalize upsampled IMU data.

    Parameters:
    - upsampled_imu_data: 2D array, upsampled IMU data.

    Returns:
    - normalized_imu_data: 2D array, normalized IMU data.
    - means: 1D array, means of each axis before normalization.
    - stds: 1D array, standard deviations of each axis before normalization.
    """
    means = np.mean(upsampled_imu_data, axis=0)
    stds = np.std(upsampled_imu_data, axis=0)

    normalized_imu_data = (upsampled_imu_data - means) / stds

    return normalized_imu_data, means, stds

def plot_profiles(profiles, max_val=None, min_val=None):
    max_h = 0       # red
    min_h = 120     # blue
    if not max_val:
        max_val = np.max(profiles)
    if not min_val:
        min_val = np.min(profiles)
    print(max_val, min_val)
    heat_map_val = np.clip(profiles, min_val, max_val)
    heat_map = np.zeros(
        (heat_map_val.shape[0], heat_map_val.shape[1], 3), dtype=np.uint8)
    # print(heat_map_val.shape)
    heat_map[:, :, 0] = heat_map_val / \
        (max_val + 1e-6) * (max_h - min_h) + min_h
    heat_map[:, :, 1] = np.ones(heat_map_val.shape) * 255
    heat_map[:, :, 2] = np.ones(heat_map_val.shape) * 255
    heat_map = cv2.cvtColor(heat_map, cv2.COLOR_HSV2BGR)
    return heat_map


def plot_profiles_split_channels(profiles, n_channels, maxval=None, minval=None):
    channel_width = profiles.shape[0] // n_channels

    profiles_img = np.zeros(
        ((channel_width + 5) * n_channels, profiles.shape[1], 3))

    for n in range(n_channels):
        channel_profiles = profiles[n * channel_width: (n + 1) * channel_width]
        profiles_img[n * (channel_width + 5): (n + 1) * (channel_width + 5) - 5,
                     :, :] = plot_profiles(channel_profiles, maxval, minval)

    return profiles_img

def vis(input):
    img_input = input.copy()
    diff_profiles_img = plot_profiles_split_channels(img_input.T, 1, 50000000, -50000000)
    #profiles11 = plot_profiles(profiles1, 20000000, -20000000)
    acous_npy_img = cv2.cvtColor(np.float32(diff_profiles_img), cv2.COLOR_BGR2RGB)
    plt.imshow(acous_npy_img.astype(np.uint16), aspect = 'auto')



def read_from_folder(session_num, data_path, is_train=False):
    file_path = data_path + '%s'%str(session_num)
    file_echo_org = file_path +  "/" + 'acoustic/non_diff'
    file_echo_diff = file_path +  "/" + 'acoustic/diff'
    file_imus = file_path +  "/"  + 'imu'
    file_gnds = file_path +  "/" + 'gnd_truth.txt'
    file_echo_org_list = sorted([f for f in os.listdir(file_echo_org)])
    file_echo_diff_list = sorted([f for f in os.listdir(file_echo_diff)])
    file_imus_list = sorted([f for f in os.listdir(file_imus)])

    with open(file_gnds, 'r', encoding='utf-8') as f:
        gt = f.read()

    gt = gt.split("\n")[:-1]

    loaded_gt = []
    data_pairs = []
    n_bad = 0
########################################################################################
    bad_signal_remove_length = 5
########################################################################################    
    for i in range(0, len(file_echo_diff_list)):
        # ground truth
        # gnd = int(file_echo_diff_list[i].split('.')[0].split('_')[2])
        # truth = gt[gnd].split(';')[3]
        # loaded_gt += [gt[gnd].split(';')]
        file = file_echo_diff_list[i]
        #last letter before the .npy
        truth = file.split('_')[-1].split('.')[0]
        if truth in ['sneakers', 'rubber']:
            truth = 'sneakers_rubber'

        # load imu
        imu_path = file_imus + "/" + file_imus_list[i]
        if imu_path.endswith('.npy'):
            File_data = np.load(imu_path)
        else:
            # Try to load as text, fallback to latin1 if needed
            try:
                File_data = np.loadtxt(imu_path, dtype=float, delimiter=" ")
            except Exception as e:
                print_and_log(f"Failed to load {imu_path} as text: {e}")
                continue  # Skip this file if it can't be loaded as text

        # Now File_data should be a numeric array
        all_imu = File_data[:, :3]
        all_imu_time = File_data[:, 3:]
        all_imu_time = np.array([i[0] for i in all_imu_time])
       
        # load echo_diff
        profiles = np.load(file_echo_diff+"/"+file_echo_diff_list[i])
        profile_data_piece = profiles.copy()
        profile_data_piece = profile_data_piece.swapaxes(1, 2) # 

        # load echo_org (original echo data)
        # profiles_org = np.load(file_echo_org+"/"+file_echo_org_list[i])
        # profile_data_piece_org = profiles_org.copy()
        # profile_data_piece_org = profile_data_piece_org.swapaxes(1, 2) # 
        
        # upsampling imu data based on echo profile
        psampled_time, upsampled_imu_data = upsample_imu_data(all_imu_time, all_imu, profile_data_piece.shape[1])
        normalized_imu_data, means, stds = normalize_imu_data(upsampled_imu_data)
        normalized_imu_data.shape = 1, normalized_imu_data.shape[0], normalized_imu_data.shape[1]

        if profile_data_piece.shape[1] > 50: # check the data quality 
            #print(truth)
            if truth in lst:
            #print("final:",  i,truth, profile_data_piece.shape, normalized_imu_data.shape)
                data_pairs += [(profile_data_piece[:,:-bad_signal_remove_length,:], 
                                # profile_data_piece_org[:,:-bad_signal_remove_length-1,:], 
                                truth, 
                                normalized_imu_data[:,:-bad_signal_remove_length,:])]
        else:
            n_bad +=1

    if n_bad:
        print('     %d bad data pieces' % n_bad)

    if is_train:
        data_pairs

    return data_pairs, loaded_gt

# def print_and_log(content, end='\n'):
#     print(content, end=end)
#     logging.info(content)



train_data = []
test_data = []
is_train = True
is_concate = False
if is_train:
    print_and_log('Loading training data...')
    for p in train_sessions:
        print_and_log(' Loading from %s' % p)
        this_train_data, _ = read_from_folder(p, data_path, is_train=True)
        train_data += this_train_data


if is_train & is_concate:
    print('is_concate')
    itr = 10
    for num in range(0, itr):
        n = random.sample(range(2, 3), 1)[0]
        # manuplate the sequecial 
        index = random.sample(range(0, len(train_data)-1), n)
        tmp = []
        for j in range(0, 4):
            if j !=2:
                tmp.append(np.concatenate([train_data[i][j] for i in index], axis=1))
            else:
                tmp.append(" ".join([train_data[i][2] for i in index]))

        tmp_tuple = tuple(tmp)
        train_data += [tmp_tuple]

print_and_log('Loading testing data...')
test_loaded_gt = []
last_index_offset = 0   # avoid confusing the sliding window truths from multiple files
for p in test_sessions:
    # data_file = os.path.join(data_config['root_folder'], p, data_config['data_file'])
    # static_file = os.path.join(data_config['root_folder'], p, data_config['static_file'])
    # truth_file = os.path.join(data_config['root_folder'], p, data_config['truth_file'])
    # config_file = os.path.join(data_config['root_folder'], p, data_config['config_file'])
    print_and_log(' Loading from %s' % p)
    this_test_data, this_loaded_gt = read_from_folder(p, data_path, is_train=False)
    test_data += this_test_data
    test_loaded_gt += this_loaded_gt


label = []
for i in this_loaded_gt:
    label.append(i[3])




def save_cm_figure(true_label,predict_label, best_save_path, acc, lst): 
    true_labels= [label_dic_reverse[i] for i in  true_label]
    #predicted_labels = df["Predicted Label"].tolist()
    predicted_labels= [label_dic_reverse[i] for i in predict_label]
    # Get unique class names and sort them (ensures correct label order)
    unique_classes = sorted(set(true_labels) | set(predicted_labels))
    # Compute confusion matrix with string labels
    cm = confusion_matrix(true_labels, predicted_labels, labels=unique_classes)
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    # Plot confusion matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", linewidths=0.5)
    # Keep the label order in figure
    plt.xticks(ticks=np.arange(len(lst)) + 0.5, labels=lst, rotation=90)
    plt.yticks(ticks=np.arange(len(lst)) + 0.5, labels=lst, rotation=0)

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix - Best Accuracy : %.3f"%acc + " %")
    plt.xticks(rotation=45)  # Rotate class labels for better visibility
    plt.yticks(rotation=0)
    plt.savefig(best_save_path+"confusion_matrix.png", dpi=300, bbox_inches="tight")  # Saves as a high-quality PNG


rand_indices = list(range(len(train_data)))
random.shuffle(rand_indices)
train_data_w_lengths = [(x[0].shape[1], rand_indices[i], x) for i, x in enumerate(train_data)]
train_data_w_lengths.sort()
train_data = [x[2] for x in train_data_w_lengths]

#########################################################################################################################################################################
#########################################################################################################################################################################
#########################################################################################################################################################################
#########################################################################################################################################################################
#########################################################################################################################################################################


import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F


# Define a basic Residual Block
class BasicBlock(nn.Module):
    expansion = 1  # Output channels same as input
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity  # Residual connection
        out = F.relu(out)
        return out

# Define ResNet Architecture
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):  # Default CIFAR-10 classification
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Define ResNet layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
       
        x = self.fc(x)
        return x

# Instantiate ResNet-18 Model
def resnet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


class DualResNetClassifier(nn.Module):
    def __init__(self, num_classes=class_num):
        super(DualResNetClassifier, self).__init__()

        # ResNet18 for First Input Modality
        self.resnet1 = models.resnet18(num_classes=class_num)
        self.resnet1.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)  # Change input channels to 1
        self.resnet1.fc = nn.Identity()  # Remove final FC layer

        # ResNet18 for Second Input Modality
        self.resnet2 = models.resnet18(num_classes=class_num)
        self.resnet2.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Change input channels to 4
        self.resnet2.fc = nn.Identity()  # Remove final FC layer

        # Batch Normalization for Normalizing Feature Vectors
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(512)

        # Fully Connected Layer for Classification
        self.fc = nn.Linear(512 + 512, num_classes)  # Combine ResNet1 (512) + ResNet2 (512)

    def normalize(self, x):
        """Normalize the feature map to the range [0, 1]"""
        min_val = torch.min(x, dim=-1, keepdim=True)[0]
        max_val = torch.max(x, dim=-1, keepdim=True)[0]
        return (x - min_val) / (max_val - min_val)
    
    def forward(self, x1, x2):
        # Extract Features
        feat1 = self.resnet1(x1)  # (B, 512, H', W')
        feat2 = self.resnet2(x2)  # (B, 512, H', W')

        # Flatten Features
        feat1 = feat1.view(feat1.size(0), -1)  # (B, 512)
        feat2 = feat2.view(feat2.size(0), -1)  # (B, 512)

        # Normalize Features
        feat1 = self.bn1(feat1)  # Batch normalization
        feat2 = self.bn2(feat2)  # Batch normalization

        # Alternative: L2 Normalization (Uncomment if needed)
        # feat1 = F.normalize(feat1, p=2, dim=1)  # L2 Normalization
        # feat2 = F.normalize(feat2, p=2, dim=1)

        feat1 = self.normalize(feat1)  # (B, 512)
        feat2 = self.normalize(feat2)  # (B, 512)

        # Concatenate Features
        fused_features = torch.cat([feat1, feat2], dim=1)  # (B, 1024)

        fused_features = self.normalize(fused_features)   # (B, 1024)

        # Fully Connected Layer for Classification
        logits = self.fc(fused_features)  # (B, num_classes)

        return logits

class Imu1dImage2dModel(nn.Module):
    def __init__(self, num_classes=class_num):
        super(Imu1dImage2dModel, self).__init__()

        # 1D CNN for IMU data

        self.imu_cnn = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # Pool to 1 to reduce to [batch, 128, 1]
        )

        # ResNet18 for Second Input Modality
        self.resnet2 = models.resnet18(num_classes=class_num)
        self.resnet2.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Change input channels to 4
        self.resnet2.fc = nn.Identity()  # Remove final FC layer

        # Batch Normalization for Normalizing Feature Vectors
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(512)

        # Fully Connected Layer for Classification
        self.fc = nn.Linear(128 + 512, num_classes)  # Combine ResNet1 (512) + ResNet2 (512)

    def normalize(self, x):
        """Normalize the feature map to the range [0, 1]"""
        min_val = torch.min(x, dim=-1, keepdim=True)[0]
        max_val = torch.max(x, dim=-1, keepdim=True)[0]
        return (x - min_val) / (max_val - min_val)
    
    def forward(self, x1, x2):
        # Extract Features
        B, C, H, W =  x1.shape # Input 1: 1D CNN for sequential IMU data (shape: [B, C, T]).
        x1 = x1.reshape(B, H, W) # Height 3 - Channel, W - Time
        feat1 = self.imu_cnn(x1)
        feat1 = feat1.view(feat1.size(0), -1)   # Flatten to [batch, 128]

        feat2 = self.resnet2(x2)  # (B, 512, H', W')
        #print("AAA", feat1.shape, feat2.shape)

        # Flatten Features
        feat1 = feat1.view(feat1.size(0), -1)  # (B, 512)
        feat2 = feat2.view(feat2.size(0), -1)  # (B, 512)
        #print("BBB", feat1.shape, feat2.shape)

        # Normalize Features
        feat1 = self.bn1(feat1)  # Batch normalization
        feat2 = self.bn2(feat2)  # Batch normalization

        # Alternative: L2 Normalization (Uncomment if needed)
        # feat1 = F.normalize(feat1, p=2, dim=1)  # L2 Normalization
        # feat2 = F.normalize(feat2, p=2, dim=1)

        feat1 = self.normalize(feat1)  # (B, 128)
        feat2 = self.normalize(feat2)  # (B, 512)

        # Concatenate Features
        fused_features = torch.cat([feat1, feat2], dim=1)  # (B, 640)

        fused_features = self.normalize(fused_features)   # (B, 640)

        # Fully Connected Layer for Classification
        logits = self.fc(fused_features)  # (B, num_classes)

        return logits


class Imu1dModel(nn.Module):
    def __init__(self, num_classes=class_num):
        super(Imu1dModel, self).__init__()

        # 1D CNN for IMU data

        self.imu_cnn = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # Pool to 1 to reduce to [batch, 128, 1]
        )

        # Batch Normalization for Normalizing Feature Vectors
        self.bn1 = nn.BatchNorm1d(128)

        # Fully Connected Layer for Classification
        self.fc = nn.Linear(128, num_classes)  # Combine ResNet1 (512) + ResNet2 (512)

    def normalize(self, x):
        """Normalize the feature map to the range [0, 1]"""
        min_val = torch.min(x, dim=-1, keepdim=True)[0]
        max_val = torch.max(x, dim=-1, keepdim=True)[0]
        return (x - min_val) / (max_val - min_val)
    
    def forward(self, x1, x2):
        # Extract Features
        B, C, H, W =  x1.shape # Input 1: 1D CNN for sequential IMU data (shape: [B, C, T]).
        x1 = x1.reshape(B, H, W) # Height 3 - Channel, W - Time
        feat1 = self.imu_cnn(x1)
        feat1 = feat1.view(feat1.size(0), -1)   # Flatten to [batch, 128]

        # Flatten Features
        feat1 = feat1.view(feat1.size(0), -1)  # (B, 512)
        #print("BBB", feat1.shape, feat2.shape)

        # Normalize Features
        feat1 = self.bn1(feat1)  # Batch normalization

        # Alternative: L2 Normalization (Uncomment if needed)
        # feat1 = F.normalize(feat1, p=2, dim=1)  # L2 Normalization
        # feat2 = F.normalize(feat2, p=2, dim=1)

        fused_features = self.normalize(feat1)  # (B, 128)

        # # Concatenate Features
        # fused_features = torch.cat([feat1, feat2], dim=1)  # (B, 640)

        # fused_features = self.normalize(fused_features)   # (B, 640)

        # Fully Connected Layer for Classification
        logits = self.fc(fused_features)  # (B, num_classes)

        return logits
    
#########################################################################################################################################################################
#########################################################################################################################################################################
#########################################################################################################################################################################
#########################################################################################################################################################################
#########################################################################################################################################################################

device = torch.device("cuda:%d"%gpu_set if torch.cuda.is_available() else "cpu")

if fusion == True:
    if retraining == False:
        if imu_1d == True:
            model = Imu1dImage2dModel(num_classes = class_num)
            model.resnet2.conv1 = nn.Conv2d(input_channel, 64, kernel_size=3, stride=1, padding=1, bias=False)  # Change input channels to 1
            print_and_log(model)
            model.to(device)

        else:
            model = DualResNetClassifier(num_classes = class_num)
            model.resnet1.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)  # Change input channels to 1
            model.resnet2.conv1 = nn.Conv2d(input_channel, 64, kernel_size=3, stride=1, padding=1, bias=False)  # Change input channels to 1
            print_and_log(model)
            model.to(device)

    else:
        if imu_1d == True:
            model = Imu1dImage2dModel(num_classes = class_num)
            model.resnet2.conv1 = nn.Conv2d(input_channel, 64, kernel_size=3, stride=1, padding=1, bias=False)  # Change input channels to 1
            model.to(device)
        else:
            model = DualResNetClassifier(num_classes = class_num)
            model.resnet1.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)  # Change input channels to 1
            model.resnet2.conv1 = nn.Conv2d(input_channel, 64, kernel_size=3, stride=1, padding=1, bias=False)  # Change input channels to 1
            model.to(device)

        print_and_log('Model loading...')
        #model.load_state_dict(torch.load(best_save_path+"best_model.pth"))
        print_and_log(model)
        checkpoint_path = best_save_path + "checkpoint.pth"
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)  # Ensure loading on the correct device
            model.load_state_dict(checkpoint["model_state"])
            start_epoch = checkpoint["epoch"] + 1  # Resume from next epoch

            # Define loss function and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            optimizer.load_state_dict(checkpoint["optimizer_state"])  # Load optimizer state

            # Move model and optimizer to the correct device
            model.to(device)
            for param_group in optimizer.param_groups:
                param_group["lr"] = 0.001  # Ensure learning rate is set correctly

            print_and_log(f"✅ Resuming training from epoch {start_epoch} on {device}")
        except FileNotFoundError:
            print_and_log("⚠ No checkpoint found. Training from scratch.")
            start_epoch = 0  # Start from the beginning

else:
    if imu_1d == True:
        if retraining == False:
            # Instantiate model
            model = Imu1dModel(num_classes=class_num)
            #model.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
            print_and_log(model)
            model.to(device)
        else:
            model = Imu1dModel(num_classes=class_num)
            #model.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
            print_and_log('Model loading...')
            #model.load_state_dict(torch.load(best_save_path+"best_model.pth"))
            print_and_log(model)
            model.to(device)

            checkpoint_path = best_save_path + "checkpoint.pth"
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device)  # Ensure loading on the correct device
                model.load_state_dict(checkpoint["model_state"])
                start_epoch = checkpoint["epoch"] + 1  # Resume from next epoch

                # Define loss function and optimizer
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=0.001)
                optimizer.load_state_dict(checkpoint["optimizer_state"])  # Load optimizer state

                # Move model and optimizer to the correct device
                model.to(device)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = 0.001  # Ensure learning rate is set correctly

                print_and_log(f"✅ Resuming training from epoch {start_epoch} on {device}")
            except FileNotFoundError:
                print_and_log("⚠ No checkpoint found. Training from scratch.")
                start_epoch = 0  # Start from the beginning

    else:

        if retraining == False:
            # Instantiate model
            model = resnet18(num_classes=class_num)
            model.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
            print_and_log(model)
            model.to(device)

        else:
            model = resnet18(num_classes=class_num)
            model.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
            print_and_log('Model loading...')
            #model.load_state_dict(torch.load(best_save_path+"best_model.pth"))
            print_and_log(model)
            checkpoint_path = best_save_path + "checkpoint.pth"
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device)  # Ensure loading on the correct device
                model.load_state_dict(checkpoint["model_state"])
                start_epoch = checkpoint["epoch"] + 1  # Resume from next epoch

                # Define loss function and optimizer
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=0.001)
                optimizer.load_state_dict(checkpoint["optimizer_state"])  # Load optimizer state

                # Move model and optimizer to the correct device
                model.to(device)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = 0.001  # Ensure learning rate is set correctly

                print_and_log(f"✅ Resuming training from epoch {start_epoch} on {device}")
            except FileNotFoundError:
                print_and_log("⚠ No checkpoint found. Training from scratch.")
                start_epoch = 0  # Start from the beginning



criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Handle resume functionality
if args.resume:
    print_and_log(f"Resuming from checkpoint: {args.resume}")
    checkpoint = torch.load(args.resume, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    start_epoch = checkpoint['epoch'] + 1
    
    # Load previous best accuracy from checkpoint
    best_val_acc = checkpoint.get('best_accuracy', 0.0)
    
    print_and_log(f"Resumed from epoch {start_epoch}")
    print_and_log(f"Training will continue from epoch {start_epoch} to {num_epochs}")
    print_and_log(f"📊 Loaded previous best accuracy from checkpoint: {best_val_acc:.2f}%")
else:
    start_epoch = 0
    best_val_acc = 0.0
    print_and_log(f"Starting training from epoch 0 to {num_epochs}")

#device = "cpu"

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'




data_splitter = DataSplitter(train_data, test_data, batch_size, 0)
# for param in model.parameters():
#     param.requires_grad = True  # Ensure that all layers require gradients
train_loader = data_splitter.train_loader
test_loader = data_splitter.test_loader
for i, (input_arr_raw, target) in enumerate(train_loader):
    input_arr = input_arr_raw[0][:,input_channel_slice,:,:]
    input_imu = input_arr_raw[1][:,:,:,:]
    print_and_log('train input shape: acoustic' + str(input_arr.shape) + ' imu' + str(input_imu.shape))
    break

for i, (input_arr_raw, target) in enumerate(test_loader):
    input_arr = input_arr_raw[0][:,input_channel_slice,:,:]
    input_imu = input_arr_raw[1][:,:,:,:]
    print_and_log('test input shape: acoustic' + str(input_arr.shape) + ' imu' + str(input_imu.shape))
    break


train_losses = []
val_losses = []
for epoch in range(start_epoch, num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    model.train()
    running_loss = 0.0

    for i, (input_arr_raw, target) in enumerate(train_loader):

        optimizer.zero_grad()

        input_arr = input_arr_raw[0][:,input_channel_slice,:,:]
        input_imu = input_arr_raw[1][:,:,:,:]

        input_arr = Tensor(input_arr).to(device)
        input_imu = Tensor(input_imu).to(device)
        labels = torch.tensor([label_dic[x] for x in target], dtype=torch.long).to(device)
        
        if fusion == True:
                outputs = model(input_imu, input_arr)
        else:
            if imu_1d == True:
                outputs = model(input_imu)
            else:
                outputs = model(input_arr)
        #print(outputs.shape, labels, outputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    print_and_log(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100 * correct/total:.2f}%")

    if epoch % 3 == 0:
        model.eval()
        test_correct = 0
        test_total = 0

        predictions = []
        true_labels = []
        #best_save_path = "/data3/hyunchul/asl/Headset_silentspeech1/dl_model/results_emo/"
        with torch.no_grad():
            for i, (input_arr_raw, target) in enumerate(test_loader):

                input_arr = input_arr_raw[0][:,input_channel_slice,:,:]
                input_imu = input_arr_raw[1][:,:,:,:]

                input_arr = Tensor(input_arr).to(device)
                input_imu = Tensor(input_imu).to(device)
                labels = torch.tensor([label_dic[x] for x in target], dtype=torch.long).to(device)
            
                if fusion == True:
                    outputs = model(input_imu, input_arr)
                else:
                    if imu_1d == True:
                        outputs = model(input_imu)
                    else:
                        outputs = model(input_arr)
                    
                _, predicted = torch.max(outputs, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
                
                predictions.extend(predicted.cpu().numpy())  
                true_labels.extend(labels.cpu().numpy())  

            test_acc = 100 * test_correct / test_total
            #print(val_correct,val_total)
                
            if test_acc > best_val_acc:
                # Store predictions and actual labels

                best_val_acc = test_acc
                df = pd.DataFrame({"True Label": true_labels, "Predicted Label": predictions})
                df.to_csv(best_save_path+"test_results.csv", index=False)
                torch.save(model.state_dict(), best_save_path+"best_model.pth")
                save_checkpoint(model, optimizer, epoch, best_acc=best_val_acc)
                print_and_log(f"🔥 Best model saved with Test Accuracy: {best_val_acc:.2f}%")
                save_cm_figure(df["True Label"],df["Predicted Label"], best_save_path, best_val_acc, lst)

            print_and_log(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100 * correct/total:.2f}%, Test Accuracy: {test_acc:.2f}%, , Best Accuracy: {best_val_acc:.2f}%")

# calculate train loss and validation loss
    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    if epoch % 3 == 0:
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for i, (input_arr_raw, target) in enumerate(test_loader):
                input_arr = input_arr_raw[0][:,input_channel_slice,:,:]
                input_imu = input_arr_raw[1][:,:,:,:]
                if not isinstance(input_arr, torch.Tensor):
                    input_arr = Tensor(input_arr).to(device) #torch.tensor(input_arr, dtype=torch.float32).to(device)
                else:
                    input_arr = input_arr.to(device)
                if not isinstance(input_imu, torch.Tensor):
                    input_imu = Tensor(input_imu).to(device) #torch.tensor(input_imu, dtype=torch.float32).to(device)
                else:
                    input_imu = input_imu.to(device)
                labels = torch.tensor([label_dic[x] for x in target], dtype=torch.long).to(device)
                if fusion == True:
                    outputs = model(input_imu, input_arr)
                else:
                    if imu_1d == True:
                        outputs = model(input_imu)
                    else:
                        outputs = model(input_arr)
                val_loss = criterion(outputs, labels)
                val_running_loss += val_loss.item()
            avg_val_loss = val_running_loss / len(test_loader)
            val_losses.append(avg_val_loss)


# plot loss curves

plt.figure()
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss Curves')
plt.savefig(best_save_path + "loss_curve.png", dpi=300, bbox_inches="tight")
plt.close()

# Save final model regardless of performance
torch.save(model.state_dict(), best_save_path+"final_model.pth")
save_checkpoint(model, optimizer, num_epochs-1, best_acc=best_val_acc, filename= best_save_path+"final_checkpoint.pth")
print_and_log(f"✅ Final model saved at epoch {num_epochs}")

# Log training completion
print_and_log("="*50)
print_and_log("Training completed successfully!")
print_and_log(f"Best validation accuracy: {best_val_acc:.2f}%")
print_and_log(f"Final model and checkpoint saved in: {best_save_path}")
print_and_log("="*50)
