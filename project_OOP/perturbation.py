from functools import partial
import numpy as np
import replace as repl



class PerturbationBase:
    def mask_percentile(x, percentile=90):
        """Create a mask based on percentile for the relevance.

        Args:
            x (ndarray): Relevance or coefficient.
            percentile (float, optional): Percentile of relevance to mask. Defaults to 90.

        Returns:
            ndarray: an ndarray mask of 0s and 1s. Same shape with x.
        """
        # 1/on/keep and 0/off/disabled
        # n_steps, features = x

        # normalized relevance
        amin = partial(np.min, axis=0)
        amax = partial(np.max, axis=0)
        relevance_norm = (x - amin(x)) / (amax(x) - (amin(x)))

        # get points > percentile 90, which are being perturbed
        p90 = np.percentile(relevance_norm, percentile, axis=0)
        m = (relevance_norm > p90)

        # reverse to have 1 = on, 0 = off
        m = 1 - m

        return m


    def _randomize(m, delta = 0.0):
        m = np.array(m)
        n_steps, features = m.shape
        n_offs = (m == 0).sum(axis = 0)
        n_offs = (np.ceil(n_offs*(1+delta))).astype(int)
        n_ons = (n_steps - n_offs).astype(int)


        random_mask = []

        # Get mask based on relevance
        for i in range(features):
            t = np.concatenate([np.zeros(n_offs[i]), np.ones(n_ons[i])])
            random_mask.append(t)
        random_mask = np.stack(random_mask, axis=1).astype(int)

        assert m.shape == random_mask.shape

        #inplace shuffle for each feature
        _ = np.apply_along_axis(np.random.shuffle, axis=0, arr=random_mask)
        return random_mask







    def mask_randomize(self, x, percentile = 90, delta = 0.0):

        m = self.mask_percentile(x, percentile)
        m = self._randomize(m ,delta)
        return m









    def _perturb(x, m, replace_method = 'zeros'):
        replace_fn = getattr(repl, replace_method)
        r = replace_fn(x, m)
        assert x.shape == m.shape == r.shape
        z = x * m + r * (1-m)
        return z





    def perturb(self, X, R, replace_method="zeros", percentile=90, shuffle=False, delta=0.0):

        for x, r in zip(X, R):
            assert x.shape == r.shape, \
                f"Conflict in shape, instance x with shape {x.shape} while relevance r: {r.shape}"

            # Get mask based on relevance
            if shuffle:
                m = self.mask_randomize(r, percentile, delta)
            else:
                m = self.mask_percentile(r, percentile)
            yield self._perturb(x, m, replace_method=replace_method)




    def perturbation(self, X, relevance, replace_method = 'zeros', percentile = 90, shuffle = False, delta = 0.0):

        for x, r in zip(X, relevance):
            assert x.shape == r.shape, (
                f"Conflict in shape, instance x with shape {x.shape} while relevance r: {r.shape}"
            )

            # Get mask based on relevance
            if shuffle:
                m = self.mask_randomize(r, percentile, delta)
            else:
                m = self.mask_percentile(r, percentile)
            yield self._perturb(x, m, replace_method = replace_method)





class PerturbationAnalysis(PerturbationBase):


    def __init__(self,) -> None:
        """Construct analysis class for perturbation method."""
        super().__init__()

        self.insights = dict()


    def add_insight(self, k, v):
        """Store the result to the insights dict.

        Args:
            k (str): name of the insight/evalutation
            v (float): value/score of the evaluation.
        """
        self.insights.update({k: v})

    def analysis_relevance(self, X, y, relevance, predict_fn,
                           eval_fn, replace_method = 'zeros', percentile = 90,
                           delta = 0.0):

        X_percentile = self.perturb(X, relevance, replace_method = replace_method,
                                percentile = percentile)

        X_percentile = np.array(list(X_percentile))
        print(X_percentile.shape)

        X_random = self.perturb(X, relevance,
                                replace_method=replace_method,
                                percentile=percentile,
                                shuffle=True,
                                delta=delta
                                )
        X_random = np.array(list(X_random))


        # Score for original
        y_pred = predict_fn(X).ravel()
        score = eval_fn(y_pred, y)
        self.add_insight('original', score)

        # Score for Percentile
        y_pred = predict_fn(X_percentile).ravel()
        score = eval_fn(y_pred, y)
        self.add_insight('percentile', score)

        # Score for random
        y_pred = predict_fn(X_random).ravel()
        score = eval_fn(y_pred, y)
        self.add_insight('random', score)

        return self.insights