import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ 
    select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ 
        select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # implement model selection based on BIC scores
        models = list()
        n = sum(self.lengths)
        for num_states in range(self.min_n_components, (self.max_n_components + 1)):
            try:
                hmm_model = self.base_model(num_states)
                LL = hmm_model.score(self.X, self.lengths)
                free_params = (num_states**2) + (2 * num_states * n) - 1
                BIC = (-2 * LL) + (free_params * np.log(n))
                models.append((BIC, hmm_model))
            except Exception as e:  # if it is a fail, record highest error
                models.append((float("-inf"), hmm_model))

        # finding out the best in the model storage
        # this is a minimization problem
        best_score, best_model = min(models, key=lambda x: x[0])

        return best_model


class SelectorDIC(ModelSelector):
    """
    select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    
    """

    def select(self):
        """
        
        :return: GaussianHMM object 
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # implement model selection based on Discriminative Information Criterion

        models = list()
        for num_states in range(self.min_n_components, self.max_n_components + 1):
            try:
                hmm_model = self.base_model(num_states)
                LL = hmm_model.score(self.X, self.lengths)

                adv_scores = list()
                for word in self.words:
                    if word != self.this_word:
                        adv_X, adv_len = self.hwords[word]
                        adv_scores.append(hmm_model.score(adv_X, adv_len))

                models.append((LL - np.mean(adv_scores), hmm_model))

            except Exception as e:  # if it is a fail, record highest error
                models.append((float("-inf"), hmm_model))

        # finding out the best in the model storage
        # this is a maximization problem
        best_score, best_model = max(models, key=lambda x: x[0])
        return best_model


class SelectorCV(ModelSelector):
    """
    select best model based on average log Likelihood of cross-validation folds
    """

    def select(self):
        """
        
        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # implement model selection using CV
        models = list()
        for num_states in range(self.min_n_components, (self.max_n_components + 1)):

            cv_results = list()

            if len(self.sequences) == 1:
                hmm_model = self.base_model(num_states)
                try:
                    cv_results.append(hmm_model.score(self.X, self.lengths))
                except Exception as e:  # if it is a fail, pass by
                    cv_results.append(float("-inf"))
            else:
                # Setting the split method
                try:
                    split_method = KFold(n_splits=min(3, len(self.sequences)), shuffle=False, random_state=self.random_state)
                    for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                        # recombining train and test sequences according to split method
                        X_train, train_len = combine_sequences(cv_train_idx, self.sequences)
                        X_test, test_len = combine_sequences(cv_test_idx, self.sequences)

                        # setting X and lenghts
                        self.X = X_train
                        self.lengths = train_len

                        # trying to model the splitted data
                        hmm_model = self.base_model(num_states)
                        cv_results.append(hmm_model.score(X_test, test_len))

                except Exception as e:  # if it is a fail, pass by
                    pass

            if len(cv_results) > 0: # if no cv results are available, num_states is not testable
                # adding mean cv log likelyhood to the model storage
                models.append((np.mean(cv_results), hmm_model))

        # finding out the best in the model storage
        # this is a maximization problem
        best_score, best_model = max(models, key=lambda x: x[0])

        return best_model



