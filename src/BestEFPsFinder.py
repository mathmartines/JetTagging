from sklearn.tree import DecisionTreeClassifier


class BestEFPsFinder:
    """
    Class responsible for finding the best EFPs for a given tree.

    It uses the backward elimination algorithm. The algorithm find first the N - 1 features that leads
    to the best accuracy, then it finds the N - 2 features, and so on.

    The search stops when the current tree accuracy is less than some percentage threshold of the
    full tree accuracy. The accuracy is evaluated on the validation set.
    """

    def __init__(self, tree: DecisionTreeClassifier, X_train, y_train, X_val, y_val):
        """
        Constructor for the BestEFPsFinder.

        :param tree: Decision Tree which we want to find the best set of features
        :param X_train: Training data
        :param y_train: Labels for the training data
        :param X_val: Validation dataset
        :param y_val: Labels for the validation dataset
        """
        self._tree = tree
        self._X_train = X_train
        self._y_train = y_train
        self._X_val = X_val
        self._y_val = y_val
        # stores the model accuracy
        self._model_accuracy = None
        # stores the features indices
        self._features_set = list(range(X_train.shape[1]))
        self._best_features_set = list(range(X_train.shape[1]))

    def backward_elimination(self, accuracy_fraction_limit: float = 0.95):
        """Runs the backward elimination algorithm."""
        # first we need to evaluate the models accuracy
        if self._model_accuracy is None:
            self._model_accuracy = self._evaluate_accuracy(features_indices=self._features_set)

        # stores the best accuracy
        best_accuracy, best_features_set = 0, None
        # trainning the model with N - 1 features and keeping the set of features with the best score
        for features_set in self._generate_features_combinations():
            current_accuracy = self._evaluate_accuracy(features_indices=features_set)
            # keeping track of the set of features that leads to the best score
            if current_accuracy > best_accuracy:
                best_accuracy, best_features_set = current_accuracy, features_set

        print(f"Best feature set is", best_features_set)
        print(f"Accuracy is {best_accuracy:.3f} ({best_accuracy/self._model_accuracy:.2f}% of model's accuracy "
              f"{self._model_accuracy:.3f})")
        # base condition
        # Either we reached the lower accuracy limit we wanted or the feature set has only one feature
        if best_accuracy < accuracy_fraction_limit * self._model_accuracy or len(best_features_set) == 1:
            return best_features_set

        # making the recursive call
        self._best_features_set = best_features_set
        return self.backward_elimination(accuracy_fraction_limit=accuracy_fraction_limit)

    def _evaluate_accuracy(self, features_indices: list) -> float:
        """Evaluates the model accuracy for a given set of features."""
        self._tree.fit(self._X_train[:, features_indices], self._y_train)
        # evaluate the accuracy of the model using the validation set
        return self._tree.score(self._X_val[:, features_indices], self._y_val)

    def _generate_features_combinations(self):
        """Generates the all the N - 1 features combinations."""
        for index in range(len(self._best_features_set)):
            yield self._best_features_set[:index] + self._best_features_set[index + 1:]
