
import sys, os
ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
sys.path.append(ROOT)
    
import numpy as np 
import torch 


import utils.lib_io as lib_io
import utils.lib_commons as lib_commons
import utils.lib_datasets as lib_datasets
import utils.lib_augment as lib_augment
import utils.lib_ml as lib_ml
import utils.lib_rnn as lib_rnn


args = lib_rnn.set_default_args()

args.learning_rate = 0.001
args.num_epochs = 25
args.learning_rate_decay_interval = 5 
args.learning_rate_decay_rate = 0.5 
args.do_data_augment = True
args.train_eval_test_ratio=[0.8, 0.1, 0.1]
args.data_folder = "data"
args.classes_txt = "classes.names" 
args.load_weights_from = None

files_name, files_label = lib_datasets.AudioDataset.load_filenames_and_labels(
    args.data_folder, args.classes_txt)

if args.do_data_augment:
    Aug = lib_augment.Augmenter
    aug = Aug([        
        Aug.Shift(rate=0.2, keep_size=False), 
        Aug.PadZeros(time=(0, 0.3)),
        Aug.Amplify(rate=(0.2, 1.5)),
        Aug.Noise(noise_folder="data/noises/", 
                        prob_noise=0.7, intensity=(0, 0.7)),
    ], prob_to_aug=0.8)
else:
    aug = None

tr_X, tr_Y, ev_X, ev_Y, te_X, te_Y = lib_ml.split_train_eval_test(
    X=files_name, Y=files_label, ratios=args.train_eval_test_ratio, dtype='list')
train_dataset = lib_datasets.AudioDataset(files_name=tr_X, files_label=tr_Y, transform=aug)
eval_dataset = lib_datasets.AudioDataset(files_name=ev_X, files_label=ev_Y, transform=None)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset, batch_size=args.batch_size, shuffle=True)


model = lib_rnn.create_RNN_model(args, load_weights_from=args.load_weights_from)
lib_rnn.train_model(model, args, train_loader, eval_loader)
