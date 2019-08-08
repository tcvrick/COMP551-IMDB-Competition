"""
Source: https://nbviewer.jupyter.org/github/kaushaltrivedi/bert-toxic-comments-multilabel/blob/master/toxic-bert-multilabel-classification.ipynb
"""

import pickle
import os

from pathlib import Path
from pytorch_pretrained_bert.tokenization import BertTokenizer
from bert_model import MultiLabelTextProcessor, convert_examples_to_features

# -------------------------------------------------------------------------------------------------
# SETTINGS
# -------------------------------------------------------------------------------------------------
PATH = Path('./preprocessed_data')
os.makedirs(str(PATH), exist_ok=True)

args = {
    "bert_model": 'bert-base-uncased',
    "data_dir": PATH,
    "max_seq_length": 512,
    "do_lower_case": True,
}


def main():
    for fold_id in range(5):
        # --------------------------------------
        # Set-up Tokenizer
        # --------------------------------------
        processor = MultiLabelTextProcessor(args['data_dir'], fold_id=fold_id)
        tokenizer = BertTokenizer.from_pretrained(args['bert_model'], do_lower_case=args['do_lower_case'])

        # --------------------------------------
        # Preprocess Data
        # --------------------------------------
        # Load examples from DF.
        train_examples = processor.get_train_examples('')
        val_examples = processor.get_dev_examples('')
        test_examples = processor.get_test_examples('', '')

        # Tokenize.
        print('Processing training features...')
        train_features = convert_examples_to_features(train_examples, args['max_seq_length'], tokenizer)
        pickle.dump(train_features, open('./preprocessed_data/train_features_{}.pkl'.format(fold_id), 'wb'))

        print('Processing validation features...')
        val_features = convert_examples_to_features(val_examples, args['max_seq_length'], tokenizer)
        pickle.dump(val_features, open('./preprocessed_data/val_features_{}.pkl'.format(fold_id), 'wb'))

        print('Processing test features...')
        test_features = convert_examples_to_features(test_examples, args['max_seq_length'], tokenizer)
        pickle.dump(test_features, open('./preprocessed_data/test_features.pkl', 'wb'))


if __name__ == '__main__':
    main()