import numpy as np
import matplotlib.pyplot as plt
import torch




#add model as a class attribute or method parameter?
class evaluation():
    def __init__(self, feat_cont, df, model):
        self.feat_cont = feat_cont
        self.model = model
        self.df = df



    def train_with_feat_Trans(self):
        new_data = self.df*self.feat_cont
        if model==""

    def train_with_feat_aug(self):
        new_data = []
        for i, j in zip(self.df, self.feat_cont):
            new_data.append(np.hstack((i, j)))

        new_data = np.asarray(new_data)

    def get_perturbation_score(self):



    def visual_analysis(self):



