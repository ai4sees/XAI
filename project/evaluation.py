from abc import abstractmethod

import numpy as np
from model import train_model
from sklearn.model_selection import train_test_split


#add model as a class attribute or method parameter?
class evaluation():
    def __init__(self, df, y = None):
        self.df = df
        self.y = y


    @abstractmethod
    def train_with_feat_transformation(self, epochs = 100, batch_size = 32,
                                       device = 'cpu', test_size = 0.2, shuffle = True, random_state = None):

        pass



    @abstractmethod
    def train_with_feat_aug(self, epochs = 100, batch_size = 32,
                            device = 'cpu', test_size = 0.2, shuffle = True, random_state = None):
        pass




    @abstractmethod
    def get_perturbation_score(self, Perturbation_Analysis, windows=False):
        pass
            







