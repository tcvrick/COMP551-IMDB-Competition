import pandas as pd
import numpy as np


def main():
    submissions = ['bert_predicted_prob.pkl', 'ed5_epoch5_predicted_prob.pkl', 'gpt_predicted_prob.pkl']
    submissions = [pd.read_pickle(x) for x in submissions]

    probabilities = np.vstack([df.Probability for df in submissions]).T
    averaged_probabilities = probabilities.mean(axis=1).squeeze()
    predictions = (averaged_probabilities > 0.5).astype(np.int32)

    data = {'Id': list(range(25000)),
            'Category': predictions}
    df = pd.DataFrame(data=data)
    df.to_csv('ensembled_predictions.csv', header=True, index=False)


if __name__ == '__main__':
    main()
