import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import precision_score, roc_auc_score, accuracy_score

class Modelo:
  
  def __init__(self, model: object, train: pd.DataFrame, test: pd.DataFrame, **params) -> None:
    self.train_data = train
    self.test_data = test
    self.params = params
    self.initial_model = model

  def execute(self):
    self.sample_data()

    if self.params["preprocess?"]:
      self.preprocessing()

    if self.params["selectors?"]:
      self.fit_selectors()
    else:
      self.fit_selectors(other_selectors=False)

    self.train_model()
    print("Processo Finalizado")

  def sample_data(self):
    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
      self.train_data, self.test_data,
      random_state=42, 
      shuffle=self.params["shuffle?"],
      train_size=self.params["train_size"]
    )

  def preprocessing(self):
    pass

  def fit_selectors(self, other_selectors: bool = True):
    self.selectors_used = ["all"]
    self.__pca = PCA()
    self.__kpca = KernelPCA()
    self.__lda = LinearDiscriminantAnalysis()

    self.pre_samples = {
      "all": {
        "X_train": self.X_train,
        "X_test": self.X_test,
      }
    }
    
    if other_selectors:
      self.selectors_used += ["PCA", "KPCA", "LDA"]
      self.pre_samples.update({
        "PCA": {
            "X_train": self.__pca.fit_transform(self.X_train),
            "X_test": self.__pca.transform(self.X_test)
          },
          "KPCA": { 
            "X_train": self.__kpca.fit_transform(self.X_train),
            "X_test": self.__kpca.transform(self.X_test)
          },
          "LDA": {
            "X_train": self.__lda.fit_transform(self.X_train, self.y_train),
            "X_test": self.__lda.transform(self.X_test)
          }
        })

  def train_model(self):
    self.fitted_models, self.results = {}, {}

    for selector in self.selectors_used:
      fitted_model = GridSearchCV(
        self.initial_model,
        self.params["model_params"],
        n_jobs=-1, 
        cv=self.params["n_folds"], 
        scoring=self.params["cv_scoring"]
        ).fit(
          X=self.pre_samples[selector]["X_train"],
          y=self.y_train
          )
      self.fitted_models[selector] = fitted_model
      
      predicted_values = fitted_model.predict(self.pre_samples[selector]["X_test"])
      
      self.results[selector] = {
        "precision": precision_score(self.y_test, predicted_values, zero_division=1),
        "accuracy": accuracy_score(self.y_test, predicted_values),
        "auc": roc_auc_score(self.y_test, predicted_values),
      }
