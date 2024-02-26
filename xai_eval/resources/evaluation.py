from abc import abstractmethod


# add model as a class attribute or method parameter?
class Evaluation:
    def __init__(self, df, y=None):
        self.df = df
        self.y = y

    @abstractmethod
    def train_with_feat_transformation(self, feat_cont, epochs=100, batch_size=32,
                                       device='cpu', test_size=0.2, shuffle=True, random_state=None):
        pass

    @abstractmethod
    def train_with_feat_aug(self, feat_cont, epochs=100, batch_size=32,
                            device='cpu', test_size=0.2, shuffle=True, random_state=None):
        pass

    @abstractmethod
    def get_perturbation_score(self, feat_cont, windows=False):
        pass
