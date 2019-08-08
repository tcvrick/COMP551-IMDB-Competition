# COMP551 Project 2: IMDB Sentiment Classification Competition

This project was completed as part of the W2019 COMP551 course and involved implementing a text sentiment classifier
for an in-class Kaggle competition. 

This repo contains the software used to produce the first-place submission of the competition and achieves 95.53% and 95.18% accuracy
on the public and private leaderboards, respectively.


## Installation Instructions

- If possible, use a Windows 10 64-bit environment with 32GB+ RAM. Linux is mostly untested, but should work.

- Install pre-requisites depending on which part of the project you want to run:
    - venv (scikit-learn, PyTorch 1.0.1, etc.) - Python 3.6
        ```
        pip install -r requirements.txt
        ```
    - FastAI (only required for ULMFiT) - Python 3.6
        ```
        conda create --name <envname> --file requirements_fastai.yml
        ```
- Downloading and installing the dataset 
    - Download the dataset from Kaggle: https://www.kaggle.com/c/comp-551-imbd-sentiment-classification
    - Move the contents of 'train' and 'test' into ```'project_src/data/'```.
    - Run the ```'utils/n1_convert_dataset_to_dataframe.py'``` utility which should load the dataset and convert it into
    a pickled dataframe object saved in ```'project_src/data/preprocessed_data```.
    
- Make sure that you are using an IDE (e.g., PyCharm or Visual Studio Code) and that the project root (i.e. the folder that contains ```project_src```) is in your 
```$PATH```, otherwise the module imports probably will fail. Some IDE's will do this for you automatically.
    
  
## Running the deep learning models

### Recommended System Pre-requisites
- Windows 10 64-bit
- 64 GB RAM
- NVIDIA GPU w/ 16 GB+ VRAM
- CUDA 10

We recommend using a cloud-computing service with a Tesla V100 or greater.

### ULMFiT
#### Setup
- Create and activate the FastAI environment (see above).
#### Training
- Navigate to ```deep_learning/ulmfit```.
- Run ```train_lm.py``` to fine-tune a language model.
- Run ```train_classifier.py``` to train a classifier.
- Run ```make_predictions.py``` to create the predicted probabilities and classes.


### OpenAI GPT

#### Setup
- Make sure ```'pytorch_openai_transformer'``` is in the current shell's PATH or PYTHONPATH so that the modules will import properly.
- Install the spacy en module (```python -m spacy download en```)

#### Training
- Navigate to ```deep_learning/pytorch_openai_transformer```.
- Either run ```train_folds.bat``` or ```python train_imdb.py --fold_id 0```

#### Evaluation
- Run ```python train_imdb.py --fold_id 0 --make_predictions```. The output
should consist of the predicted probabilties and classes.

### BERT

#### Training
##### Fine-tuning a LM (Optional)
- Install "en" for Spacy (https://spacy.io/usage/models)
- Create the training corpus using "make_finetune_corpus.py"
- Run "run_lm_finetune.py", e.g.
``` bash
python3 run_lm_finetune.py --train_file preprocessed_data/imdb_train_corpus_0.txt --bert_model bert-base-uncased --output_dir finetune_0_uncased_512_1e --max_seq_length 512 --on_memory --do_lower_case --do_train --train_batch_size 12 --num_train_epochs 1
```

- Use the resulting trained LM for the next step.

##### Training a classifier
- Create the pre-processed data using ```preprocess_data.py```, ensuring that you select the correct max sequence length.
- Modify ```run_experiment.py``` so that ```BERT_PRETRAINED_PATH``` points to either ```bert-base-uncased``` or wherever you saved the fine-tuned LM (if applicable).
Furthermore, if you pretrained a LM, you will need to obtain the vocab and config files to pair with it (i.e., placed in the same directory). They can be found at ```https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz``` and ```https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt```.
- Run the script ```run_experiment.py 0```, where the first argument specifies the fold ID. If 
- Create the submission file using ```make_predictions.py```. The output
should consist of the predicted probabilties and classes.


### Stacked Ensemble
- Collect the 15 validation prediction, and 15 test prediction files from the 3 models and place them into 
```deep_learning/stacked_ensemble/inputs``` and ```deep_learning/stacked_ensemble/inputs_test```.
- Run ```deep_learning/stacked_ensemble/train_stacker.py```.