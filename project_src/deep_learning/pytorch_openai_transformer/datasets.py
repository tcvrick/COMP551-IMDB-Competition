import re
import html
import pandas as pd
re1 = re.compile(r'  +')


def imdb(fold_id: int, split_size: int):
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

    df_test = pd.read_pickle('df_test.pkl')
    df_test = df_test.reindex(columns=['review_id', 'text'])
    df_test['text'] = df_test['text'].apply(fixup)

    return (df_train.text.values, df_train.sentiment.values), (df_val.text.values, df_val.sentiment.values),\
           (df_test.text.values,)


# https://github.com/prajjwal1/language-modelling/blob/master/ULMfit.py
def fixup(x):
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>', 'u_n').replace(' @.@ ', '.').replace(
        ' @-@ ', '-').replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(x))
