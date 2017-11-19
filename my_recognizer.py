import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # implement the recognizer
    probabilities = list()
    guesses = list()

    test_size = len(test_set.get_all_Xlengths())
    for i in range(test_size):
        features, length = test_set.get_item_Xlengths(i)
        LL = dict()
        for word, model in models.items():
            try:
                score = model.score(features, length)
                LL[word] = score
            except Exception as e: # if it is a fail, record highest error
                LL[word] = float("-inf")
        probabilities.append(LL)
        guesses.append(max(LL, key=LL.get))

    return probabilities, guesses
