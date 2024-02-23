import numpy as np
import shap
import torch
from sklearn import metrics, linear_model
from sklearn.model_selection import train_test_split
from tsmule.xai.lime import LimeTS
from tsmule.sampling.segment import WindowSegmentation, MatrixProfileSegmentation, SAXSegmentation
from tsmule.sampling.perturb import Perturbation
from evaluation import Evaluation
from model import TrainModel
from perturb.perturbation import PerturbationAnalysis


class Xai(Evaluation):
    def __init__(self, model, back_data, df, y, kernel=linear_model.Lasso(alpha=0.01)):
        super().__init__(df, y=None)
        self.model = model
        self.back_data = back_data
        self.df = df
        self.y = y
        self.kernel = kernel


    def predict_fn(self, X):
      if not torch.is_tensor(X):
        X=torch.tensor(X, dtype=torch.float)
        if len(X.shape) == 2:
            prediction = self.model(X[np.newaxis, :, :]).ravel()
        else:
            prediction = self.model(X).ravel()
        return prediction.detach().numpy()

    @staticmethod
    def eval_fn(y_pred, y):
      y_pred = torch.from_numpy(y_pred).float()
      y = torch.from_numpy(y).float()
      mse_= torch.mean(torch.square(y_pred - y))
      return mse_

    def get_shap_values(self):
        back_data = torch.from_numpy(self.back_data).float()
        df=torch.from_numpy(self.df).float()
        e = shap.DeepExplainer(self.model, back_data)
        shap_values = e.shap_values(df)
        print("Shape of shap values: ", shap_values.shape)
        shap_values = np.array(shap_values).squeeze()
        return shap_values

    def get_lime_values(self, seg_method, par=4, win_length=24, n_samples=24):
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

    def train_with_feat_transformation(self, feat_cont, epochs=100, batch_size=32,
                                       device='cpu', test_size=0.2, shuffle=True, random_state=None):

        new_data = self.df * feat_cont
        x_train, x_test, y_train, y_test = train_test_split(new_data,
                      self.y, test_size=test_size, shuffle=shuffle,
                           random_state=random_state)
        train = TrainModel(self.model, x_train, y_train, x_test, y_test, epochs, batch_size, device)

        trained_model, train_loss, val_loss = train()

        return trained_model, train_loss, val_loss

    def train_with_feat_aug(self, feat_cont, epochs=100, batch_size=32,
                            device='cpu', test_size=0.2, shuffle=True, random_state=None):
        new_data = []
        for i, j in zip(self.df, feat_cont):
            new_data.append(np.hstack((i, j)))

        new_data = np.asarray(new_data)

        x_train, x_test, y_train, y_test = train_test_split(new_data, self.y,
               test_size=test_size, shuffle=shuffle,
                   random_state=random_state)
        train = TrainModel(self.model, x_train, y_train, x_test, y_test, epochs, batch_size, device)

        trained_model, train_loss, val_loss = train()

        return trained_model, train_loss, val_loss

    def get_perturbation_score(self, feat_cont, windows=False):
        per = PerturbationAnalysis()
        scores = per.analysis_relevance(self.df, self.y, feat_cont,
                                        predict_fn=self.predict_fn, eval_fn=self.eval_fn, replace_method='zeros', percentile=90,
                                        delta=0.0)
        orig_score = scores['original']
        pert_score = scores['percentile']
        rand_score = scores['random']
        pert_c = np.abs((orig_score - pert_score) / orig_score)
        rand_c = np.abs((orig_score - rand_score) / orig_score)
        score = pert_c / rand_c

        if windows:
            scores = per.analysis_relevance_windows(self.df, self.y, feat_cont,
                 predict_fn=self.predict_fn, eval_fn=self.eval_fn, replace_method='zeros',
                       percentile=90, delta=0.0)
            orig_score = scores['original_windows']
            pert_score = scores['percentile_windows']
            rand_score = scores['random_windows']
            pert_c = np.abs((orig_score - pert_score) / orig_score)
            rand_c = np.abs((orig_score - rand_score) / orig_score)
            score_win = pert_c / rand_c
            return score, score_win

        else:
            return score
