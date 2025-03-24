# These are all the parameters. 
import argparse

# parameters (training)
parser = argparse.ArgumentParser()
parser.add_argument('-data', '--train_dataset', default='MDD12', help='choose the training dataset')
args = parser.parse_args()

# dataset for training
train_dataset = args.train_dataset

# dataset for testing
test_dataset = train_dataset

# check validity
ds_range = ['Dxy5','MZ10', 'MZ4', 'MDD12']

assert train_dataset in ds_range
assert test_dataset in ds_range

device_num = 0

# train/test data path
train_path = []
if train_dataset != 'all':
    train_path.append('data/{}/train_set.json'.format(train_dataset))
else:
    for ds in ds_range[1:]:
        train_path.append('data/{}/train_set.json'.format(ds))

test_path = []
if test_dataset != 'all':
    test_path.append('data/{}/test_set.json'.format(test_dataset))
else:
    for ds in ds_range[1:]:
        test_path.append('data/{}/test_set.json'.format(ds))

best_pt_path = 'saved/{}/best_pt_exe_model.pt'.format(train_dataset)
last_pt_path = 'saved/{}/last_pt_exe_model.pt'.format(train_dataset)

# global settings
suffix = {'0': '-Negative', '1': '-Positive', '2': '-Negative'}
min_sx_freq = None
max_voc_size = None
keep_unk = True
digits = 4

# model hyperparameter setting

num_attrs = 5
num_executor = 2


# group 3: transformer encoder
enc_emb_dim = 256 if train_dataset == 'Dxy5' else 512
enc_dim_feedforward = 2 * enc_emb_dim

enc_num_heads = 8 if train_dataset == 'Dxy5' or train_dataset == 'MDD12' else 4
enc_num_layers = 2 if train_dataset == 'Dxy5' or train_dataset == 'MDD12' else 1
enc_dropout = 0.25 # {'Dxy5': 0.75, 'MZ10': 0.6, 'MZ4': 0.25, 'MDD12': 0.25}.get(train_dataset)


# training
num_workers = 0
pt_train_epochs = 60
pt_learning_rate = 3e-4 
train_bsz = 128
test_bsz = 128

alpha = 0.2
exp_name = 'hai_all'
num_repeats = 5
verbose = True
checkpoint_state = 'm_acc'

# human
human_acc = 0.95
human_weak_acc = 0.5
num_dx_s = {'Dxy5': 4, 'MZ10': 8, 'MZ4': 3, 'MDD12': 10}.get(train_dataset)
num_dx = {'Dxy5': 5, 'MZ10': 10, 'MZ4': 4, 'MDD12': 12}.get(train_dataset)
# utility
correct = 1
error = 4
lambda_a = 0.25
lambda_r = 0.5
# gamma
k_ul = {'Dxy5': 0.62, 'MZ10': 0.07, 'MZ4': 0.93, 'MDD12': 1.0}.get(train_dataset) # if human_weak_acc == 0.5
print_per = False