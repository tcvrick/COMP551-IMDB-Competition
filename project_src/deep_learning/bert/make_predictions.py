import torch
import pickle
import numpy as np
import pandas as pd

from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from bert_model import BertForMultiLabelSequenceClassification


def main():
    for i in range(5):
        make_predictions(i)


def make_predictions(fold_id):
    # --------------------------------------
    # Set-up
    # --------------------------------------
    batch_size = 64
    bert_model = 'finetune_{}_uncased_512_1e'.format(fold_id)
    model_state_dict_path = 'trained_models/fold_{fold}/bert_best_acc_{fold}.bin'.format(fold=fold_id)
    device = 'cuda:0'

    model_state_dict = torch.load(model_state_dict_path)
    model = BertForMultiLabelSequenceClassification.from_pretrained(bert_model, num_labels=1,
                                                                    state_dict=model_state_dict)
    model.to(device)

    # --------------------------------------
    # VAL
    # --------------------------------------
    val_features = pickle.load(open('preprocessed_data/val_features_{}.pkl'.format(fold_id), 'rb'))
    val_dataloader = make_dataloader(val_features, batch_size)
    # --------------------------------------
    # Evaluate
    # --------------------------------------
    model.eval()

    all_logits = []
    for input_ids, input_mask, segment_ids, label_ids in tqdm(val_dataloader, desc='Evaluating val set...'):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask)

        all_logits.append(logits)

    predicted_probabilities = torch.cat(all_logits).sigmoid().cpu().detach().numpy().squeeze()
    predicted_class = (predicted_probabilities > 0.5).astype(np.int32).squeeze()

    # --------------------------------------
    # Save
    # --------------------------------------
    data = {'Id': list(range(len(val_features))),
            'Category': predicted_class}
    df = pd.DataFrame(data=data)
    df.to_csv('val_bert_pred_class_{}.csv'.format(fold_id), header=True, index=False)

    data = {'Id': list(range(len(val_features))),
            'Probability': predicted_probabilities}
    df = pd.DataFrame(data=data)
    df.to_pickle('val_bert_pred_prob_{}.pkl'.format(fold_id))

    # --------------------------------------
    # TEST
    # --------------------------------------
    test_features = pickle.load(open('preprocessed_data/test_features.pkl', 'rb'))
    test_dataloader = make_dataloader(test_features, batch_size)

    # --------------------------------------
    # Evaluate
    # --------------------------------------
    model.eval()

    all_logits = []
    for input_ids, input_mask, segment_ids, label_ids in tqdm(test_dataloader, desc='Evaluating test set...'):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask)

        all_logits.append(logits)

    predicted_probabilities = torch.cat(all_logits).sigmoid().cpu().detach().numpy().squeeze()
    predicted_class = (predicted_probabilities > 0.5).astype(np.int32).squeeze()

    # --------------------------------------
    # Save
    # --------------------------------------
    data = {'Id': list(range(len(test_features))),
            'Category': predicted_class}
    df = pd.DataFrame(data=data)
    df.to_csv('test_bert_pred_class_{}.csv'.format(fold_id), header=True, index=False)

    data = {'Id': list(range(len(test_features))),
            'Probability': predicted_probabilities}
    df = pd.DataFrame(data=data)
    df.to_pickle('test_bert_pred_prob_{}.pkl'.format(fold_id))


def make_dataloader(features, batch_size=64):
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)
    data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)

    return dataloader


if __name__ == '__main__':
    main()
