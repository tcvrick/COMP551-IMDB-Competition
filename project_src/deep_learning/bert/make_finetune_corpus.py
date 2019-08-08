import spacy
import pickle
import os
from pathlib import Path
from tqdm import tqdm
from bert_model import MultiLabelTextProcessor
PATH = Path('./preprocessed_data')
os.makedirs(str(PATH), exist_ok=True)

nlp = spacy.load('en', disable=['parser', 'tagger', 'ner', 'textcat'])
nlp.add_pipe(nlp.create_pipe('sentencizer'))

for fold_id in range(5):
    processor = MultiLabelTextProcessor(PATH, fold_id=fold_id)
    train_features = processor.df_train

    result = []
    for i, row in tqdm(enumerate(train_features.itertuples()), desc='Processing text...', total=len(train_features)):
        doc = nlp(row.text.replace('\n', '').strip())
        sentences = [x.text.strip() for x in doc.sents if len(x.text.strip()) > 0]
        sentences = '\n'.join(sentences)

        if len(sentences) > 0:
            result.append(sentences)

    with open('./preprocessed_data/imdb_train_corpus_{}.txt'.format(fold_id), 'w+', encoding='utf-8') as f:
        f.write('\n\n'.join(result))
