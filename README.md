# EECS-487-Project

back_trarnslate_main.ipynb: code for running data_augementation.py

data_augementation.py: does back translation

main.ipynb: code for running 2 transformer models and 1 LSTM model

main.py: functions for running the baseline model, searching hyperparameters for LSTM, and overfitting LSTM

data_processor.py: make a Dataset for LSTM

model.py: builds LSTM model

naive_bayes.py: builds naive bayes model

pickle_to_csv.py: converts pickle files to csv files. Reuters and Wikipedia data were originally pickle files

transformer.py: make a Dataset for BERT models, builds BERT model with 2 FC layers

preds.csv: predictions outputted by BERT model

train.py: contains functions for training LSTM model and getting LSTM model performance

other files are for book keeping






train.csv, dev.csv, gold-test-27446.csv: dataset containing all data, including translated texts

