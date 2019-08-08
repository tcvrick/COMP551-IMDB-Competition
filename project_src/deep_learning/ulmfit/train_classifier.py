import dill
import spacy
import pandas as pd
import shutil, os
from pathlib import Path

from fastai.text import *
from fastai.text.models import get_language_model
from fastai.train import OptimWrapper, SaveModelCallback
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

    # Load test set for making predictions after training.
    df_test = pd.read_pickle('df_test.pkl')
    df_test = df_test.reindex(columns=['review_id', 'text'])
    df_test['text'] = df_test['text'].apply(fixup)

    # Sanity check to make sure there are no common elements between the two splits.
    if set(df_train.index).intersection(set(df_val.index)):
        raise ValueError('There are common training examples in the training and validation splits!')

    # Load pretrained LM and get its vocabulary.
    bs = 64
    data_lm = TextLMDataBunch.load('lm{fold}'.format(fold=fold_id))

    # Create and cache the dataloaders.
    data_clas = TextClasDataBunch.from_df('.',
                                          train_df=df_train,
                                          valid_df=df_val,
                                          test_df=df_test, vocab=data_lm.train_ds.vocab, bs=bs)
    data_clas.save('cls{fold}/tmp'.format(fold=fold_id))

    # -----------------------------------------
    # Copy the best encoder from the LM
    # -----------------------------------------
    os.makedirs('cls{fold}/models/'.format(fold=fold_id), exist_ok=True)
    shutil.copy('lm{fold}/models/best_encoder.pth'.format(fold=fold_id), 'cls{fold}/models/'.format(fold=fold_id))

    # -----------------------------------------
    # Load the dataloaders from cache
    # -----------------------------------------
    data_clas = TextClasDataBunch.load('cls{fold}'.format(fold=fold_id), bs=bs)
    learn = text_classifier_learner(data_clas, drop_mult=0.75)
    learn.load_encoder('best_encoder')

    # -----------------------------------------
    # Hyperparameters
    # -----------------------------------------
    opt_fn = partial(optim.Adam, betas=(0.7, 0.99))
    opt_wrapper = OptimWrapper.create(opt_fn, 0.01, learn.layer_groups, wd=0, true_wd=True, bn_wd=True)
    learn.opt = opt_wrapper

    # -----------------------------------------
    # Discr. Learning Rate
    # -----------------------------------------
    init_lr = 1e-2
    lrs = np.array([init_lr / 2.6 ** 4, init_lr / 2.6 ** 3, init_lr / 2.6 ** 2, init_lr / 2.6, init_lr])

    # -----------------------------------------
    # Train
    # -----------------------------------------
    print('Training...')
    learn.freeze_to(-1)
    learn.fit(1, lrs)
    learn.freeze_to(-2)
    learn.fit(1, lrs)
    learn.freeze_to(-3)
    learn.fit(1, lrs)
    learn.freeze_to(-4)
    learn.fit(1, lrs)
    learn.unfreeze()

    init_lr = 1e-2
    lrs = np.array([init_lr / 2.6 ** 4, init_lr / 2.6 ** 3, init_lr / 2.6 ** 2, init_lr / 2.6, init_lr])
    learn.fit(12, lrs, callbacks=[SaveModelCallback(learn, every='epoch')])

    # -----------------------------------------
    # Predictions
    # -----------------------------------------
    # learn.load('bestmodel_5')
    #
    # from fastai.basic_data import DatasetType
    # test_preds = learn.get_preds(DatasetType.Test, ordered=True)
    #
    # df = pd.DataFrame(test_preds[0].argmax(dim=1), columns=['Category'])
    # df['Id'] = range(len(df))
    # df = df.reindex(columns=['Id', 'Category'])
    # df.to_csv('submission_{fold}.csv'.format(fold=fold_id), header=True, index=False)
    #
    # probabilites = test_preds[0][:, 1]
    # data = {'Id': list(range(len(df))),
    #         'Probability': probabilites}
    # df = pd.DataFrame(data=data)
    # df.to_pickle('predicted_prob_{fold}.pkl'.format(fold=fold_id))


# https://github.com/prajjwal1/language-modelling/blob/master/ULMfit.py
def fixup(x):
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>', 'u_n').replace(' @.@ ', '.').replace(
        ' @-@ ', '-').replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(x))


if __name__ == '__main__':
    main()
