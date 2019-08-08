import torch
import numpy as np
import pandas as pd
from torch.nn import BCELoss

from pathlib import Path
from collections import namedtuple
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from scipy.stats import mode

LevelOneModel = namedtuple('LevelOneModel', field_names=['file_paths'])


def main():
    ulmfit = LevelOneModel(file_paths=['inputs/val_ulmfit_pred_prob_0.pkl',
                                       'inputs/val_ulmfit_pred_prob_1.pkl',
                                       'inputs/val_ulmfit_pred_prob_2.pkl',
                                       'inputs/val_ulmfit_pred_prob_3.pkl',
                                       'inputs/val_ulmfit_pred_prob_4.pkl'])

    gpt = LevelOneModel(file_paths=['inputs/val_gpt_pred_prob_0.pkl',
                                    'inputs/val_gpt_pred_prob_1.pkl',
                                    'inputs/val_gpt_pred_prob_2.pkl',
                                    'inputs/val_gpt_pred_prob_3.pkl',
                                    'inputs/val_gpt_pred_prob_4.pkl'])

    bert = LevelOneModel(file_paths=['inputs/val_bert_pred_prob_0.pkl',
                                     'inputs/val_bert_pred_prob_1.pkl',
                                     'inputs/val_bert_pred_prob_2.pkl',
                                     'inputs/val_bert_pred_prob_3.pkl',
                                     'inputs/val_bert_pred_prob_4.pkl'])

    df_train = pd.read_pickle('df_train.pkl')
    df_test = pd.read_pickle('df_test.pkl')

    # Load stuff.
    ulmfit_df = load_level_one_model(ulmfit.file_paths)
    gpt_df = load_level_one_model(gpt.file_paths)
    bert_df = load_level_one_model(bert.file_paths)

    # Get CV metrics.
    print('ULMFiT')
    print(get_cv_acc(ulmfit_df.Probability.values, df_train.sentiment.values))
    print(get_cv_bce(ulmfit_df.Probability.values, df_train.sentiment.values))
    print(get_cv_metric_report(ulmfit_df.Probability.values, df_train.sentiment.values))
    print('')

    print('GPT')
    print(get_cv_acc(gpt_df.Probability.values, df_train.sentiment.values))
    print(get_cv_bce(gpt_df.Probability.values, df_train.sentiment.values))
    print(get_cv_metric_report(gpt_df.Probability.values, df_train.sentiment.values))
    print('')

    print('BERT')
    print(get_cv_acc(bert_df.Probability.values, df_train.sentiment.values))
    print(get_cv_bce(bert_df.Probability.values, df_train.sentiment.values))
    print(get_cv_metric_report(bert_df.Probability.values, df_train.sentiment.values))
    print('')

    # Make New Dataset
    models_df = [ulmfit_df, gpt_df, bert_df]
    models_df = [df.rename(index=str, columns={"Probability": i})
                 for i, df in enumerate(models_df)]

    level2_df = pd.concat(models_df, axis=1)
    level2_df = level2_df.drop('Id', axis=1)
    level2_df['sentiment'] = df_train.sentiment.values
    level2_df = level2_df.reset_index()

    # Train LR
    # ----------------------------------------------------
    # Load model and dataset
    # ----------------------------------------------------
    models = [LogisticRegression(solver='lbfgs'), RandomForestClassifier(max_depth=3,
                                                                         n_estimators=10),
              GaussianNB(),
              SVC(gamma='scale')]
    # model = LogisticRegression()
    # model = XGBClassifier()
    # model = RandomForestClassifier(max_depth=3)
    # model = SVC(gamma='scale')

    trained_folds = []
    fold_accuracies = []
    for i in range(5):
        print('===================================================')
        print('TRAINING FOLD {}'.format(i))
        print('===================================================')
        fold_df_train, fold_df_val = make_split_df(level2_df, fold_id=i)

        # ----------------------------------------------------
        # Fit the models
        # ----------------------------------------------------
        x = np.vstack([fold_df_train[i].values for i in range(len(models_df))]).T
        y = fold_df_train.sentiment

        trained_models = []
        for model in models:
            print('Fitting model [{}]...'.format(model.__class__.__name__))
            model.fit(x, y)
            trained_models.append(model)

        ensemble_model = VotingClassifier(estimators=[((str(i)), model) for i, model in enumerate(trained_models)]).fit(
            x,
            y)

        trained_models.append(ensemble_model)
        trained_folds.append(ensemble_model)

        # ----------------------------------------------------
        # Evaluation
        # ----------------------------------------------------
        x_val = np.vstack([fold_df_val[i].values for i in range(len(models_df))]).T
        y_val = fold_df_val.sentiment
        fold_accuracy = []
        for model in trained_models:
            print('----------------------------------------')
            print('Evaluating model [{}]...'.format(model.__class__.__name__))
            print('----------------------------------------')

            # Training MSE
            train_score = model.score(x, y)
            print('Training Acc: [{:.3f} %]'.format(train_score * 100))

            # Validation MSE
            val_score = model.score(x_val, y_val)
            fold_accuracy.append(val_score)
            print('Validation Acc: [{:.3f} %]'.format(val_score * 100))

        fold_accuracies.append(fold_accuracy)

    # ----------------------------------------------------
    # Summarize K-Folds Results
    # ----------------------------------------------------
    for fold_acc in np.array(fold_accuracies):
        print(fold_acc.mean())

    fold_accuracies = np.array(fold_accuracies)[-1, :]
    print("Mean K-Fold Validation Accuracies:")
    print(fold_accuracies.mean())

    # ----------------------------------------------------
    # Test Set
    # ----------------------------------------------------
    make_test_preds(trained_folds)


def make_test_preds(trained_folds):
    ulmfit = LevelOneModel(file_paths=['inputs_test/test_ulmfit_pred_prob_0.pkl',
                                       'inputs_test/test_ulmfit_pred_prob_1.pkl',
                                       'inputs_test/test_ulmfit_pred_prob_2.pkl',
                                       'inputs_test/test_ulmfit_pred_prob_3.pkl',
                                       'inputs_test/test_ulmfit_pred_prob_4.pkl'])

    gpt = LevelOneModel(file_paths=['inputs_test/test_gpt_pred_prob_0.pkl',
                                    'inputs_test/test_gpt_pred_prob_1.pkl',
                                    'inputs_test/test_gpt_pred_prob_2.pkl',
                                    'inputs_test/test_gpt_pred_prob_3.pkl',
                                    'inputs_test/test_gpt_pred_prob_4.pkl'])

    bert = LevelOneModel(file_paths=['inputs_test/test_bert_pred_prob_0.pkl',
                                     'inputs_test/test_bert_pred_prob_1.pkl',
                                     'inputs_test/test_bert_pred_prob_2.pkl',
                                     'inputs_test/test_bert_pred_prob_3.pkl',
                                     'inputs_test/test_bert_pred_prob_4.pkl'])

    # Load stuff.
    ulmfit_df = load_level_one_model(ulmfit.file_paths)
    gpt_df = load_level_one_model(gpt.file_paths)
    bert_df = load_level_one_model(bert.file_paths)

    models_df = [ulmfit_df, gpt_df, bert_df]
    models_df = [np.mean(np.vstack(np.split(df.Probability.values, 5)).T, axis=1) for df in models_df]
    x = np.vstack([prob for prob in models_df]).T

    results = []
    for model in trained_folds:
        results.append(model.predict(x))
    results = np.vstack(results).T
    vote_results = mode(results, axis=1)[0].squeeze()

    # SAVE
    data = {'Id': list(range(25000)),
            'Category': vote_results}
    df = pd.DataFrame(data=data)
    df.to_csv('STACKED_ENSEMBLE.csv', header=True, index=False)


def load_level_one_model(file_paths):
    result = []
    for fp in file_paths:
        result.append(pd.read_pickle(fp))
    return pd.concat(result)


def get_cv_acc(y_pred_prob: np.ndarray, y_true: np.ndarray):
    return sum((y_pred_prob > 0.5).astype(int) == y_true) / len(y_true)


def get_cv_bce(y_pred_prob: np.ndarray, y_true: np.ndarray):
    loss = BCELoss()
    return loss(torch.tensor(y_pred_prob).float(), torch.tensor(y_true).float()).item()


def get_cv_metric_report(y_pred_prob: np.ndarray, y_true: np.ndarray):
    return classification_report(y_true, (y_pred_prob > 0.5).astype(int))


def make_split_df(df, fold_id, split_size=5000):
    df = df.copy()

    # Split the data into k-folds.
    df_val = df[split_size * fold_id:split_size * (fold_id + 1)]
    df_train = pd.concat((df[0:split_size * fold_id],
                          df[split_size * (fold_id + 1):]))

    # Sanity check to make sure there are no common elements between the two splits.
    if set(df_train.index).intersection(set(df_val.index)):
        raise ValueError('There are common training examples in the training and validation splits!')

    return df_train, df_val


if __name__ == '__main__':
    main()
