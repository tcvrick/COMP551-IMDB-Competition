import argparse
import os
import random
import pickle
import torch
import numpy as np
import pandas as pd
import torch.nn as nn

from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

from datasets import imdb
from model_pytorch import DoubleHeadModel, load_openai_pretrained_model
from opt import OpenAIAdam
from text_utils import TextEncoder
from utils import (encode_dataset, iter_data,
                   ResultLogger, make_path)

from pathlib import Path
from loss import MultipleChoiceLossCompute


def transform_input(X1):
    n_batch = len(X1)
    xmb = np.zeros((n_batch, 1, n_ctx, 2), dtype=np.int32)
    mmb = np.zeros((n_batch, 1, n_ctx), dtype=np.float32)
    start = encoder['_start_']
    for i, x1 in enumerate(X1):
        x12 = [start] + x1[:max_len] + [clf_token]
        l12 = len(x12)
        xmb[i, 0, :l12, 0] = x12
        mmb[i, 0, :l12] = 1
    # Position information that is added to the input embeddings in the TransformerModel
    xmb[:, :, :, 1] = np.arange(n_vocab + n_special, n_vocab + n_special + n_ctx)
    return xmb, mmb


def iter_apply(Xs, Ms, Ys):
    logits = []

    # Compute BCE Loss
    bce = torch.nn.BCEWithLogitsLoss()
    bce_losses = []
    with torch.no_grad():
        dh_model.eval()
        for xmb, mmb, ymb in iter_data(Xs, Ms, Ys, n_batch=n_batch_train, truncate=False, verbose=False):
            XMB = torch.tensor(xmb, dtype=torch.long).to(device)
            YMB = torch.tensor(ymb, dtype=torch.long).to(device)
            MMB = torch.tensor(mmb).to(device)
            _, clf_logits = dh_model(XMB)

            bce_loss = bce(clf_logits.softmax(1)[:, 1], YMB.float()).item()
            bce_losses.append(bce_loss)
            logits.append(clf_logits.cpu().detach().numpy())
        logits = np.concatenate(logits, 0)
        bce_losses = sum(bce_losses) / len(bce_losses)

    return logits, bce_losses


def iter_predict(Xs, Ms):
    logits = []
    with torch.no_grad():
        dh_model.eval()
        for xmb, mmb in iter_data(Xs, Ms, n_batch=n_batch_train, truncate=False, verbose=False):
            XMB = torch.tensor(xmb, dtype=torch.long).to(device)
            _, clf_logits = dh_model(XMB)
            logits.append(clf_logits.cpu().detach().numpy())
    logits = np.concatenate(logits, 0)
    return logits


def log_val_performance(save_dir, desc):
    global best_acc, best_loss
    print("Logging")
    tr_logits, tr_cost = iter_apply(trX[:n_valid], trM[:n_valid], trY[:n_valid])
    va_logits, va_cost = iter_apply(vaX, vaM, vaY)
    tr_acc = accuracy_score(trY[:n_valid], np.argmax(tr_logits, 1)) * 100.
    va_acc = accuracy_score(vaY, np.argmax(va_logits, 1)) * 100.
    logger.log(n_epochs=n_epochs, n_updates=n_updates, tr_cost=tr_cost, va_cost=va_cost, tr_acc=tr_acc, va_acc=va_acc)
    print('\n#Epoch, #Update, Train Loss, Val Loss, Train Acc, Val Acc')
    print('%d %d %.3f %.3f %.2f %.2f' % (n_epochs, n_updates, tr_cost, va_cost, tr_acc, va_acc))

    if va_acc > best_acc:
        print('Saving model because of best accuracy.')
        best_acc = va_acc
        path = os.path.join(save_dir, desc, 'best_params_acc.pt')
        torch.save(dh_model.state_dict(), make_path(path))

    if va_cost < best_loss:
        print('Saving model because of best loss.')
        best_loss = va_cost
        path = os.path.join(save_dir, desc, 'best_params_loss.pt')
        torch.save(dh_model.state_dict(), make_path(path))


def run_epoch():
    for xmb, mmb, ymb in iter_data(*shuffle(trX, trM, trYt, random_state=np.random),
                                   n_batch=n_batch_train, truncate=True, verbose=False):
        global n_updates
        dh_model.train()
        XMB = torch.tensor(xmb, dtype=torch.long).to(device)
        YMB = torch.tensor(ymb, dtype=torch.long).to(device)
        MMB = torch.tensor(mmb).to(device)
        lm_logits, clf_logits = dh_model(XMB)
        compute_loss_fct(XMB, YMB, MMB, clf_logits, lm_logits)
        n_updates += 1
        if n_updates % 500 == 0:
            log_val_performance(save_dir, desc)


if __name__ == '__main__':
    # --------------------------------------------------
    # Arguments
    # --------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--desc', type=str, help="Description", default='imdb')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--log_dir', type=str, default='log/')
    parser.add_argument('--save_dir', type=str, default='save/')
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--submission_dir', type=str, default='submission/')
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--analysis', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_iter', type=int, default=3)
    parser.add_argument('--n_batch', type=int, default=8)
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument('--lr', type=float, default=6.25e-5)
    parser.add_argument('--lr_warmup', type=float, default=0.002)
    parser.add_argument('--n_ctx', type=int, default=512)
    parser.add_argument('--n_embd', type=int, default=768)
    parser.add_argument('--n_head', type=int, default=12)
    parser.add_argument('--n_layer', type=int, default=12)
    parser.add_argument('--embd_pdrop', type=float, default=0.1)
    parser.add_argument('--attn_pdrop', type=float, default=0.1)
    parser.add_argument('--resid_pdrop', type=float, default=0.1)
    parser.add_argument('--clf_pdrop', type=float, default=0.1)
    parser.add_argument('--l2', type=float, default=0.01)
    parser.add_argument('--vector_l2', action='store_true')
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--afn', type=str, default='gelu')
    parser.add_argument('--lr_schedule', type=str, default='warmup_linear')
    parser.add_argument('--encoder_path', type=str, default='model/encoder_bpe_40000.json')
    parser.add_argument('--bpe_path', type=str, default='model/vocab_40000.bpe')
    parser.add_argument('--n_transfer', type=int, default=12)
    parser.add_argument('--lm_coef', type=float, default=0.5)
    parser.add_argument('--b1', type=float, default=0.9)
    parser.add_argument('--b2', type=float, default=0.999)
    parser.add_argument('--e', type=float, default=1e-8)
    parser.add_argument('--n_valid', type=int, default=374)
    parser.add_argument('--topic', type=str, default=None)
    parser.add_argument('--make_predictions', action='store_true')
    parser.add_argument('--fold_id', type=int, required=True)
    args = parser.parse_args()
    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # --------------------------------------------------
    # Constants
    # --------------------------------------------------
    fold_id = args.fold_id
    split_size = 5000

    submit = args.submit
    dataset = args.dataset
    n_ctx = args.n_ctx
    save_dir = args.save_dir
    desc = args.desc + '_{}_{}'.format(fold_id, split_size)
    data_dir = args.data_dir
    log_dir = args.log_dir
    submission_dir = args.submission_dir + '_{}_{}'.format(fold_id, split_size)
    topic = args.topic

    if (Path(save_dir) / desc).is_dir() and args.make_predictions is False:
        raise FileExistsError('There is an existing model trained for this fold already!!')

    # --------------------------------------------------
    # Set-up
    # --------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("device", device, "n_gpu", n_gpu)

    logger = ResultLogger(path=os.path.join(log_dir, '{}.jsonl'.format(desc)), **args.__dict__)
    text_encoder = TextEncoder(args.encoder_path, args.bpe_path)
    encoder = text_encoder.encoder
    n_vocab = len(text_encoder.encoder)

    # --------------------------------------------------
    # Encode or load data
    # --------------------------------------------------
    split_size = 5000
    print('=================================================')
    print('FOLD ID = [{}], SPLIT SIZE = [{}]'.format(fold_id, split_size))
    print('=================================================')
    encoded_data_path = 'encoded_data_{}_{}.pkl'.format(fold_id, split_size)
    if not Path(encoded_data_path).exists():
        print("Encoding dataset... {}".format(encoded_data_path))
        ((trX, trY), (vaX, vaY), (teX,)) = encode_dataset(*imdb(fold_id=fold_id, split_size=split_size),
                                                          encoder=text_encoder)
        pickle.dump(((trX, trY), (vaX, vaY), (teX,)), open(encoded_data_path, 'wb'))
    else:
        print("Loading encoded dataset from cache... {}".format(encoded_data_path))
        ((trX, trY), (vaX, vaY), (teX,)) = pickle.load(open(encoded_data_path, 'rb'))

    # --------------------------------------------------
    # More data processing...
    # --------------------------------------------------
    encoder['_start_'] = len(encoder)
    encoder['_classify_'] = len(encoder)
    clf_token = encoder['_classify_']
    n_special = 2
    max_len = n_ctx - 2

    vocab = n_vocab + n_special + n_ctx
    trX, trM = transform_input(trX)
    vaX, vaM = transform_input(vaX)
    teX, teM = transform_input(teX)

    n_train = len(trY)
    n_valid = len(vaY)
    n_batch_train = args.n_batch * max(n_gpu, 1)
    n_updates_total = (n_train // n_batch_train) * args.n_iter

    # --------------------------------------------------
    # Create model
    # --------------------------------------------------
    dh_model = DoubleHeadModel(args, clf_token, ('classification', 2), vocab, n_ctx)

    criterion = nn.CrossEntropyLoss(reduction='none')
    model_opt = OpenAIAdam(dh_model.parameters(),
                           lr=args.lr,
                           schedule=args.lr_schedule,
                           warmup=args.lr_warmup,
                           t_total=n_updates_total,
                           b1=args.b1,
                           b2=args.b2,
                           e=args.e,
                           l2=args.l2,
                           vector_l2=args.vector_l2,
                           max_grad_norm=args.max_grad_norm)
    compute_loss_fct = MultipleChoiceLossCompute(criterion,
                                                 criterion,
                                                 args.lm_coef,
                                                 model_opt)
    load_openai_pretrained_model(dh_model.transformer, n_ctx=n_ctx, n_special=n_special)
    dh_model.to(device)

    n_updates = 0
    n_epochs = 0
    trYt = trY

    best_acc = 0
    best_loss = 1e12

    # --------------------------------------------------
    # Train
    # --------------------------------------------------
    if args.make_predictions is False:
        for i in range(args.n_iter):
            print("running epoch", i)
            run_epoch()
            n_epochs += 1
    else:
        path = os.path.join(save_dir, desc, 'best_params_acc.pt')
        dh_model.load_state_dict(torch.load(path))

        # -----------------------------------------
        # Val Set
        # -----------------------------------------
        print('Predicting on validation set...')
        logits = iter_predict(vaX, vaM)

        predicted_probabilities = torch.tensor(logits).sigmoid().numpy().squeeze()[:, 1]
        predicted_class = (predicted_probabilities > 0.5).astype(np.int32).squeeze()

        data = {'Id': list(range(len(vaX))),
                'Category': predicted_class}
        df = pd.DataFrame(data=data)
        df.to_csv('val_gpt_pred_class_{}.csv'.format(fold_id), header=True, index=False)

        data = {'Id': list(range(len(vaX))),
                'Probability': predicted_probabilities}
        df = pd.DataFrame(data=data)
        df.to_pickle('val_gpt_pred_prob_{}.pkl'.format(fold_id))

        # -----------------------------------------
        # Test Set
        # -----------------------------------------
        print('Predicting on test set...')
        logits = iter_predict(teX, teM)

        predicted_probabilities = torch.tensor(logits).sigmoid().numpy().squeeze()[:, 1]
        predicted_class = (predicted_probabilities > 0.5).astype(np.int32).squeeze()

        data = {'Id': list(range(len(teX))),
                'Category': predicted_class}
        df = pd.DataFrame(data=data)
        df.to_csv('test_gpt_pred_class_{}.csv'.format(fold_id), header=True, index=False)

        data = {'Id': list(range(len(teX))),
                'Probability': predicted_probabilities}
        df = pd.DataFrame(data=data)
        df.to_pickle('test_gpt_pred_prob_{}.pkl'.format(fold_id))
