import re
import html
import os
import shutil
from pathlib import Path
from fastai.text import *
from fastai.train import SaveModelCallback
re1 = re.compile(r'  +')


def main():
    for i in range(5):
        print('Fold: [{}]'.format(i))
        train_fold(i)


def train_fold(fold_id: int = 0, split_size: int = 5000):
    # -----------------------------------------
    # Create the dataloaders and cache results
    # -----------------------------------------
    df = pd.read_pickle('df_train.pkl')
    df = df.reindex(columns=['sentiment', 'text'])
    df['text'] = df['text'].apply(fixup)

    # Split the data into k-folds.
    df_val = df[split_size * fold_id:split_size * (fold_id + 1)]
    df_train = pd.concat((df[0:split_size * fold_id],
                          df[split_size * (fold_id + 1):]))

    # Sanity check to make sure there are no common elements between the two splits.
    if set(df_train.index).intersection(set(df_val.index)):
        raise ValueError('There are common training examples in the training and validation splits!')

    bs = 128
    bptt = 70
    print('Preprocessing data...')
    md = TextLMDataBunch.from_df('.', train_df=df_train, valid_df=df_val, bs=bs, bptt=bptt,
                                 min_freq=10)
    md.save('lm{fold}/tmp/'.format(fold=fold_id))

    # -----------------------------------------
    # Load the dataloaders from cache
    # -----------------------------------------
    md = TextLMDataBunch.load('lm{fold}'.format(fold=fold_id), bs=bs)
    learn = language_model_learner(md, pretrained_model=URLs.WT103_1, drop_mult=1)

    # -----------------------------------------
    # Hyperparameters
    # -----------------------------------------
    opt_fn = partial(optim.Adam, betas=(0.7, 0.99))
    opt_wrapper = OptimWrapper.create(opt_fn, 0.01, learn.layer_groups, wd=0.0, true_wd=True, bn_wd=True)
    learn.opt = opt_wrapper

    # -----------------------------------------
    # Discr. Learning Rate
    # -----------------------------------------
    init_lr = 4e-3
    lrs = np.array([init_lr / 2.6 ** 3, init_lr / 2.6 ** 2, init_lr / 2.6, init_lr])

    # -----------------------------------------
    # Train
    # -----------------------------------------
    print('Training...')
    learn.unfreeze()
    learn.fit(30, lrs, callbacks=[SaveModelCallback(learn)])

    # -----------------------------------------
    # Save the best encoder
    # -----------------------------------------
    learn.load('bestmodel')
    learn.save_encoder('best_encoder')


# https://github.com/prajjwal1/language-modelling/blob/master/ULMfit.py
def fixup(x):
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>', 'u_n').replace(' @.@ ', '.').replace(
        ' @-@ ', '-').replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(x))


if __name__ == '__main__':
    main()
