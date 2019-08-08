import os
import pandas as pd
from typing import Iterable
from collections import namedtuple, OrderedDict
from pathlib import Path


def main():
    file_name = ''.join(os.path.basename(__file__).split('.py')[:-1])
    print('===================================================')
    print('Executing script: [{}]'.format(file_name))
    print('===================================================')

    # -----------------------------
    # Load the training dataset
    # -----------------------------
    # Search for text files.
    pos_reviews = Path('../data/train/').glob('pos/*.txt')
    neg_reviews = Path('../data/train/').glob('neg/*.txt')
    test_reviews = Path('../data/test/').glob('*.txt')

    # Read from the files.
    pos_review_data = load_training_review_data(pos_reviews, sentiment=1)
    neg_review_data = load_training_review_data(neg_reviews, sentiment=0)

    # Convert to a DataFrame object.
    training_review_data = {**pos_review_data, **neg_review_data}
    df_train = pd.DataFrame.from_dict(training_review_data, orient='index').sort_values('review_id')

    # -----------------------------
    # Load the testing dataset
    # -----------------------------
    test_review_data = load_testing_review_data(test_reviews)
    df_test = pd.DataFrame.from_dict(test_review_data, orient='index').sort_values('review_id')

    # -----------------------------
    # Shuffle and save to file
    # -----------------------------
    df_train = df_train.sample(frac=1, random_state=169169961)

    # -----------------------------
    # Drop some unnecessary columns
    # -----------------------------
    df_train = df_train.reindex(columns=['review_id', 'sentiment', 'text'])
    df_test = df_test.reindex(columns=['review_id', 'text'])

    out_path = '../data/preprocessed_data/df_train.pkl'
    print('Saving preprocessed data to: [{}]'.format(out_path))
    df_train.to_pickle(out_path)

    out_path = '../data/preprocessed_data/df_test.pkl'
    print('Saving preprocessed data to: [{}]'.format(out_path))
    df_test.to_pickle(out_path)
    return


def load_training_review_data(review_paths: Iterable, sentiment: int) -> dict:
    loaded_data = OrderedDict()
    ReviewData = namedtuple('ReviewData', ['review_id', 'sentiment', 'score', 'filepath', 'text'])

    for path in review_paths:
        # 1001_7 -> 1001, 7
        review_id, score = path.stem.split('_')

        # The positive and negative values have conflicting enumerations, e.g. review_id=1000 exists
        # in both the pos and neg dataset. Work around this by adding 12.5k to all positive review ids.
        review_id = int(int(review_id) + 12500 * sentiment)
        score = int(score)

        if loaded_data.get(review_id) is not None:
            raise ValueError('Review ID already exists!')

        with open(str(path), 'r+', encoding="utf8") as f:
            text = f.read()
        loaded_data[review_id] = ReviewData(review_id=review_id, sentiment=sentiment, score=score,
                                            filepath=path.resolve(),
                                            text=text)
    return loaded_data


def load_testing_review_data(review_paths: Iterable) -> dict:
    loaded_data = OrderedDict()
    ReviewData = namedtuple('ReviewData', ['review_id', 'filepath', 'text'])

    for path in review_paths:
        review_id = path.stem
        if loaded_data.get(review_id) is not None:
            raise ValueError('Review ID already exists!')

        with open(str(path), 'r+', encoding="utf8") as f:
            text = f.read()
        loaded_data[review_id] = ReviewData(review_id=int(review_id), filepath=path.resolve(),
                                            text=text)
    return loaded_data


if __name__ == '__main__':
    main()
