"""
Source: https://nbviewer.jupyter.org/github/kaushaltrivedi/bert-toxic-comments-multilabel/blob/master/toxic-bert-multilabel-classification.ipynb
"""

import sys
import os
import logging
import torch
import random
import pickle
import numpy as np

from pathlib import Path
from bert_model import BertForMultiLabelSequenceClassification, accuracy_thresh, warmup_linear
from pytorch_pretrained_bert.optimization import BertAdam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from torch.nn import BCEWithLogitsLoss

# -------------------------------------------------------------------------------------------------
# SETTINGS
# -------------------------------------------------------------------------------------------------
FOLD_ID = int(sys.argv[1])
print('STARTING FOLD ID = [{}]'.format(FOLD_ID))


DATA_PATH = Path('./preprocessed_data')
os.makedirs(DATA_PATH, exist_ok=True)

PATH = Path('./preprocessed_data/tmp')
os.makedirs(PATH, exist_ok=True)

BERT_PRETRAINED_PATH = 'finetune_{}_uncased_512_1e'.format(FOLD_ID)
PYTORCH_PRETRAINED_BERT_CACHE = Path('trained_models') / 'fold_{}/'.format(FOLD_ID)
os.makedirs(str(PYTORCH_PRETRAINED_BERT_CACHE), exist_ok=True)

args = {
    "full_data_dir": DATA_PATH,
    "data_dir": PATH,
    "bert_model": BERT_PRETRAINED_PATH,
    "batch_size": 14,
    "learning_rate": 3e-5,
    "num_train_epochs": 2,
    "seed": 42,
    "num_train_epochs_stop": 1.5,  # Work-around for finishing training in-between epochs.
    "warmup_type": 'warmup_linear'
}


# -------------------------------------------------------------------------------------------------
# TRAIN
# -------------------------------------------------------------------------------------------------
def main():
    print(args)
    # --------------------------------------
    # GPU Parameters
    # --------------------------------------
    device = torch.device("cuda:0")

    # --------------------------------------
    # Random Seeds
    # --------------------------------------
    random.seed(args['seed'])
    np.random.seed(args['seed'])
    torch.manual_seed(args['seed'])

    # --------------------------------------
    # Set-up Model
    # --------------------------------------
    model = BertForMultiLabelSequenceClassification.from_pretrained(args['bert_model'], num_labels=1)
    model.to(device)

    # --------------------------------------
    # Set-up Loss Function
    # --------------------------------------
    loss_func = BCEWithLogitsLoss()

    # --------------------------------------
    # Load Data
    # --------------------------------------
    train_features = pickle.load(open('preprocessed_data/train_features_{}.pkl'.format(FOLD_ID), 'rb'))
    val_features = pickle.load(open('preprocessed_data/val_features_{}.pkl'.format(FOLD_ID), 'rb'))

    train_dataloader = make_dataloader(train_features, args['batch_size'])
    val_dataloader = make_dataloader(val_features, args['batch_size'])

    # --------------------------------------
    # Set-up Training Data
    # --------------------------------------
    print('---------------------------------')
    print("Running Training")
    print('---------------------------------')
    print("Number of Examples = {}".format(len(train_features)))
    print("Batch Size = {}".format(args['batch_size']))

    # --------------------------------------
    # Set-up Optimizer
    # --------------------------------------
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    t_total = int(len(train_features) / args['batch_size'] * args['num_train_epochs'])

    optimizer = BertAdam(optimizer_grouped_parameters, lr=args['learning_rate'], warmup=0.1, t_total=t_total,
                         schedule=args['warmup_type'])

    # --------------------------------------
    # Train
    # --------------------------------------
    print('---------------------------------')
    print('Training Classifier')
    print('---------------------------------')
    model.freeze_bert_encoder()
    training_loop(model, device, loss_func, optimizer, train_dataloader, val_dataloader, t_total,
                  num_epochs=1, num_epochs_stop=1e9, save_best_models=False)

    print('---------------------------------')
    print('Training Entire Model')
    print('---------------------------------')
    model.unfreeze_bert_encoder()
    training_loop(model, device, loss_func, optimizer, train_dataloader, val_dataloader, t_total,
                  num_epochs=args['num_train_epochs'], save_best_models=True)


def make_dataloader(features, batch_size=64):
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)
    data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)

    return dataloader


def training_loop(model, device, loss_func, optimizer, train_dataloader, val_dataloader, t_total,
                  num_classes=1, num_epochs=args['num_train_epochs'], num_epochs_stop=args['num_train_epochs_stop'],
                  save_best_models=True):
    global_step = 0
    eval_interval = int((t_total / num_epochs * 0.2))  # Report eval results every X% of an epoch.
    stop_step = int(num_epochs_stop * (t_total / num_epochs))  # When to stop training early.
    best_acc, best_loss = 0.0, float('inf')

    model.train()
    for i_ in range(int(num_epochs)):
        tr_loss, tr_acc = 0, 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(train_dataloader):

            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            logits = model(input_ids, segment_ids, input_mask)
            train_acc = accuracy_thresh(logits, label_ids)
            loss = loss_func(logits.view(-1, num_classes), label_ids.view(-1, num_classes))
            loss.backward()

            tr_loss += loss.item()
            tr_acc += train_acc.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1

            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            # Display the training statistics every X% of an epoch.
            if global_step > 1 and global_step % eval_interval == 0:
                print('\n..... Step [{}] Results .....'.format(global_step))
                print('Train Loss: {}'.format(tr_loss / nb_tr_steps))
                print('Train Acc: {}'.format(tr_acc / nb_tr_steps))
                val_loss, val_acc = val_loop(model, device, loss_func, val_dataloader)

                # Periodically save the best model.
                if save_best_models:
                    if val_loss < best_loss:
                        print('Saving model for best loss.')
                        best_loss = val_loss
                        model_to_save = model.module if hasattr(model, 'module') else model
                        output_model_file = os.path.join(PYTORCH_PRETRAINED_BERT_CACHE, "bert_best_loss_{}.bin".format(FOLD_ID))
                        torch.save(model_to_save.state_dict(), output_model_file)
                    if val_acc > best_acc:
                        print('Saving model for best accuracy.')
                        best_acc = val_acc
                        model_to_save = model.module if hasattr(model, 'module') else model
                        output_model_file = os.path.join(PYTORCH_PRETRAINED_BERT_CACHE, "bert_best_acc_{}.bin".format(FOLD_ID))
                        torch.save(model_to_save.state_dict(), output_model_file)

            # Stop training early.
            if global_step >= stop_step:
                print('STOPPING TRAINING AT STEP {}'.format(global_step))
                print('Train Loss: {}'.format(tr_loss / nb_tr_steps))
                print('Train Acc: {}'.format(tr_acc / nb_tr_steps))
                val_loop(model, device, loss_func, val_dataloader)
                return

        print('***** Epoch [{}], Step [{}] Finished *****'.format(i_, global_step))


def val_loop(model, device, loss_func, val_dataloader, num_classes=1):

    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    for input_ids, input_mask, segment_ids, label_ids in val_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask)
            loss = loss_func(logits.view(-1, num_classes), label_ids.view(-1, num_classes))

        val_acc = accuracy_thresh(logits, label_ids)

        eval_loss += loss.item()
        eval_accuracy += val_acc.item()

        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_steps

    print('Val Loss: {}'.format(eval_loss))
    print('Val Acc: {}'.format(eval_accuracy))

    return eval_loss, eval_accuracy


if __name__ == '__main__':
    main()
