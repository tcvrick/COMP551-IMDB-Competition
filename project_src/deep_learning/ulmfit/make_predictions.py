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
    best_accurracy = [6, 8, 4, 9, 4]
    best_loss = [3, 8, 3, 2, 4]
    for fold_id in range(5):
        print('Fold: [{}]'.format(fold_id))
        predict_fold(classifier_model_id=best_accurracy[fold_id], fold_id=fold_id)


def predict_fold(classifier_model_id, fold_id: int = 0):
    # -----------------------------------------
    # Load the dataloaders from cache
    # -----------------------------------------
    bs = 64
    data_clas = TextClasDataBunch.load('cls{fold}'.format(fold=fold_id), bs=bs)
    learn = text_classifier_learner(data_clas, drop_mult=0.75)

    # -----------------------------------------
    # Predictions
    # -----------------------------------------
    learn.load('bestmodel_{}'.format(classifier_model_id))
    from fastai.basic_data import DatasetType

    # -----------------------------------------
    # Val
    # -----------------------------------------
    val_preds = learn.get_preds(DatasetType.Valid, ordered=True)

    df = pd.DataFrame(val_preds[0].argmax(dim=1), columns=['Category'])
    df['Id'] = range(len(df))
    df = df.reindex(columns=['Id', 'Category'])
    df.to_csv('val_ulmfit_pred_class_{fold}.csv'.format(fold=fold_id), header=True, index=False)

    probabilites = val_preds[0][:, 1]
    data = {'Id': list(range(len(df))),
            'Probability': probabilites}
    df = pd.DataFrame(data=data)
    df.to_pickle('val_ulmfit_pred_prob_{fold}.pkl'.format(fold=fold_id))

    # -----------------------------------------
    # Test
    # -----------------------------------------
    test_preds = learn.get_preds(DatasetType.Test, ordered=True)

    df = pd.DataFrame(test_preds[0].argmax(dim=1), columns=['Category'])
    df['Id'] = range(len(df))
    df = df.reindex(columns=['Id', 'Category'])
    df.to_csv('test_ulmfit_pred_class_{fold}.csv'.format(fold=fold_id), header=True, index=False)

    probabilites = test_preds[0][:, 1]
    data = {'Id': list(range(len(df))),
            'Probability': probabilites}
    df = pd.DataFrame(data=data)
    df.to_pickle('test_ulmfit_pred_prob_{fold}.pkl'.format(fold=fold_id))


# https://github.com/prajjwal1/language-modelling/blob/master/ULMfit.py
def fixup(x):
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>', 'u_n').replace(' @.@ ', '.').replace(
        ' @-@ ', '-').replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(x))


if __name__ == '__main__':
    main()
