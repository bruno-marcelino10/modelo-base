# Imports Base
import pandas as pd
import numpy as np

from model import Modelo

# Dados
dados = pd.read_csv("titanic.csv")
dados.drop(["Name", "PassengerId","Ticket", "Cabin"], axis=1, inplace=True)
train_data = dados.drop("Survived", axis=1)
test_data = dados["Survived"]

# Modelagem
from model import Modelo
from sklearn.ensemble import RandomForestClassifier

params = {
  "preprocess?": False, 
  "shuffle?": True,
  "train_size": 0.7,
  "n_folds": 2,
  "cv_scoring": "roc_auc",
  "model_params": {
    'criterion': ["gini", 'entropy'],
    'min_samples_leaf': [1, 3, 5],
    'min_samples_split': [2, 4]
    #'max_depth': [1, 5, None], # Retirar
    #'n_estimators': [25, 50, 100], # Retirar
  }
}

modelo = Modelo(
  RandomForestClassifier(), 
  train=train_data,
  test=test_data,
  **params
  )