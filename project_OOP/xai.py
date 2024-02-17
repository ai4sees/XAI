import numpy as np
import shap
from sklearn import metrics, linear_model
from sklearn.model_selection import train_test_split
from tsmule.xai.lime import LimeTS
from tsmule.sampling.segment import WindowSegmentation, MatrixProfileSegmentation, SAXSegmentation
from tsmule.sampling.perturb import Perturbation
from evaluation import evaluation
from model import train_model


class xai(evaluation):
    def __init__(self, model, back_data, df, kernel = linear_model.Lasso(alpha=0.01)):
        super().__init__(df, y = None)
        self.model = model
        self.back_data = back_data
        self.df = df
        self.kernel = kernel
        self.feat_cont = None

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
            self.feat_cont = shap_values
            return self.feat_cont



    def get_lime_values(self, seg_method, par = 4, win_length = 24, n_samples= 24):
        per = Perturbation()
        if seg_method == "uniform segmentation":
            uniform_seg = WindowSegmentation(partitions=par, win_length=win_length)
            uniform_lime = LimeTS(kernel=self.kernel, segmenter=uniform_seg, sampler=per, n_samples=n_samples)
            lime_values_uni = [uniform_lime.explain(self.df[i], self.predict_fn, segmentation_method='uniform')
                               for i in range(len(self.df))]

            print("LIME Values Shape: ", np.array(lime_values_uni).shape)
            self.feat_cont = lime_values_uni
            return self.feat_cont

        if seg_method == "exponential segmentation":
            # segment object, WindowSegmentation has stationery and exponentials segmentation techniques
            exp_seg = WindowSegmentation(partitions=par, win_length=win_length)
            exp_lime = LimeTS(kernel=self.kernel, segmenter=exp_seg, sampler=per, n_samples=n_samples)
            # explainer for LimeTS
            lime_values_exp = [exp_lime.explain(self.df[i], self.predict_fn, segmentation_method='exponential')
                               for i in range(len(self.df))]

            print("LIME Values Shape: ", np.array(lime_values_exp).shape)
            self.feat_cont = lime_values_exp
            return self.feat_cont


        if seg_method == "sax segmentation":
            # create segment object for SAX Transformation
            seg_sax = SAXSegmentation(partitions=par, win_length=win_length)

            lime_sax = LimeTS(kernel=self.kernel, segmenter=seg_sax, sampler=per, n_samples=n_samples)
            lime_values_sax = [lime_sax.explain(self.df[i], self.predict_fn) for i in range(len(self.df))]

            print("LIME Values Shape: ", np.array(lime_values_sax).shape)
            self.feat_cont = lime_values_sax
            return self.feat_cont

        else:
            print("Invalid segmentation technique")




    def train_with_feat_transformation(self, epochs=100, batch_size=32,
                                       device='cpu', test_size=0.2, shuffle=True, random_state=None):

        new_data = self.df*self.feat_cont
        x_train, y_train, x_test, y_test = train_test_split(new_data, self.y,
                             test_size = test_size, shuffle = shuffle, random_state = random_state)
        train = train_model(self.model, x_train, y_train, x_test, y_test, epochs, batch_size, device)

        trained_model, train_loss, val_loss = train()

        return trained_model, train_loss, val_loss





    def train_with_feat_aug(self, epochs = 100, batch_size = 32,
                            device = 'cpu', test_size = 0.2, shuffle = True, random_state = None):
        new_data = []
        for i, j in zip(self.df, self.feat_cont):
            new_data.append(np.hstack((i, j)))

        new_data = np.asarray(new_data)

        x_train, y_train, x_test, y_test = train_test_split(new_data, self.y,
                                                            test_size=test_size, shuffle=shuffle,
                                                            random_state=random_state)
        train = train_model(self.model, x_train, y_train, x_test, y_test, epochs, batch_size, device)

        trained_model, train_loss, val_loss = train()

        return trained_model, train_loss, val_loss






    def get_perturbation_score(self, Perturbation_Analysis, windows=False):
        scores = Perturbation_Analysis.analysis_relevance(self.df, self.y, self.feat_cont,
                                                          predict_fn=self.pred_fn, replace_method='zeros', percentile=90,
                                                          delta=0.0)
        orig_score = scores['original']
        pert_score = scores['percentile']
        rand_score = scores['random']
        pert_c = np.abs((orig_score - pert_score) / orig_score)
        rand_c = np.abs((orig_score - rand_score) / orig_score)
        score = pert_c / rand_c

        if windows:
            scores = Perturbation_Analysis.analysis_relevance_windows(self.df, self.y, self.feat_cont,
                                                                      predict_fn=self.pred_fn, replace_method='zeros',
                                                                      percentile=90,
                                                                      delta=0.0)
            orig_score = scores['original_windows']
            pert_score = scores['percentile_windows']
            rand_score = scores['random_windows']
            pert_c = np.abs((orig_score - pert_score) / orig_score)
            rand_c = np.abs((orig_score - rand_score) / orig_score)
            score_win = pert_c / rand_c
            return score, score_win

        else:
            return score
