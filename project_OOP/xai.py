import numpy as np
import pandas as pd
import shap
import dill
from sklearn import metrics, linear_model
from sklearn.model_selection import train_test_split
from tsmule.xai.lime import LimeTS
from tsmule.sampling.segment import WindowSegmentation, MatrixProfileSegmentation, SAXSegmentation
from tsmule.sampling.perturb import Perturbation
from tsmule.xai.evaluation import PerturbationAnalysis
import torch
from train import RNNModel, TransformerRegressor



class xai():
    def __init__(self, model, back_data, df, kernel = linear_model.Lasso(alpha=0.01)):
        self.model = model
        self.back_data = back_data
        self.df = df
        self.kernel = kernel


    def predict_fn(self):
        if len(self.df.shape) == 2:
            prediction = self.model.predict(self.df[np.newaxis, :, :]).ravel()
        else:
            prediction = self.model.predict(self.df).ravel()
        return prediction


    def get_shap_values(self):
            e = shap.DeepExplainer(self.model, self.back_data)
            shap_values = e.shap_values(self.df)
            print("Shape of shap values: ", shap_values.shape)
            shap_values = np.array(shap_values).squeeze()
            return



    def get_lime_values(self, seg_method, par = 4, win_length = 24, n_samples= 24):
        per = Perturbation()
        if seg_method == "uniform segmentation":
            uniform_seg = WindowSegmentation(partitions=par, win_length=win_length)
            uniform_lime = LimeTS(kernel=self.kernel, segmenter=uniform_seg, sampler=per, n_samples=n_samples)
            lime_values_uni = [uniform_lime.explain(self.df[i], self.predict_fn, segmentation_method='uniform')
                               for i in range(len(self.df))]

            print("LIME Values Shape: ", np.array(lime_values_uni).shape)
            return lime_values_uni

        if seg_method == "exponential segmentation":
            # segment object, WindowSegmentation has stationery and exponentials segmentation techniques
            exp_seg = WindowSegmentation(partitions=par, win_length=win_length)
            exp_lime = LimeTS(kernel=self.kernel, segmenter=exp_seg, sampler=per, n_samples=n_samples)
            # explainer for LimeTS
            lime_values_exp = [exp_lime.explain(self.df[i], self.predict_fn, segmentation_method='exponential')
                               for i in range(len(self.df))]

            print("LIME Values Shape: ", np.array(lime_values_exp).shape)
            return lime_values_exp


        if seg_method == "sax segmentation":
            # create segment object for SAX Transformation
            seg_sax = SAXSegmentation(partitions=par, win_length=win_length)

            lime_sax = LimeTS(kernel=self.kernel, segmenter=seg_sax, sampler=per, n_samples=n_samples)
            lime_values_sax = [lime_sax.explain(self.df[i], self.predict_fn) for i in range(len(self.df))]

            print("LIME Values Shape: ", np.array(lime_values_sax).shape)
            return lime_values_sax

        else:
            print("Invalid segmentation technique")


