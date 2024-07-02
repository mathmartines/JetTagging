"""
Implementation of the Full Tagger interface.

It describes how binary taggers can be combined to performed mult-class tagging.
"""

from abc import ABC, abstractmethod

import numpy as np


class Tagger(ABC):
    """Defines the interface for a generic tagger."""
    @abstractmethod
    def predict_proba(self, input_data):
        """
        Predicts the probability of the input data to belong to each possible class.
        The output is a array with shape [samples, classes] with the probabilities for each sample
        to belong to a given class.
        """
        raise NotImplementedError("predict_proba is not implemented")


class BinaryTagger(Tagger, ABC):
    """A binary classifier must hold only one tagger."""
    def __init__(self, classifier):
        # binary tagger to use
        self._classifier = classifier


class ScikitLearnBinaryTagger(BinaryTagger):
    """Implementation of the tagger interface for models trainned with the scikit-learn library"""
    def predict_proba(self, input_data):
        # sklearn models provides a method to predict the probability
        # the first column is for class 0 and the second for class 1
        return self._classifier.predict_proba(input_data)[:, ::-1]


class KerasBinaryTaggerSoftMax(BinaryTagger):
    """Implementation of the tagger interface for keras models with output given by the softmax function."""
    def predict_proba(self, input_data):
        # for keras binary classifiers with two output nodes with the softmax function, we just
        # need to call the predict method
        return self._classifier.predict(input_data)


class FullTagger(Tagger):
    """
    Implements the tagger interface for a multi-class classifier.
    in our case, one classifier has to be a top-tagger, while the other has to be a quark-gluon tagger
    """

    def __init__(self, top_tagger: BinaryTagger, quark_gluon_tagger: BinaryTagger):
        self._top_tagger = top_tagger
        self._quark_gluon_tagger = quark_gluon_tagger

    def predict_proba(self, input_data):
        """
        Returns the probabilities in the following form:
        [probability for top, probability for quark, probability for gluon].
        """
        # first, evaluating the prediction of probabilities of being top and non-top jet
        top_probabilities = self._top_tagger.predict_proba(input_data)
        # probability of being quarks or gluons
        quark_probabilities = self._quark_gluon_tagger.predict_proba(input_data)
        # we have to account the probability of it being not a top jet
        for index_col in range(quark_probabilities.shape[-1]):
            quark_probabilities[:, index_col] = quark_probabilities[:, index_col] * top_probabilities[:, 1]
        # returning all the probabilities
        return np.hstack([top_probabilities[:, :1], quark_probabilities])
