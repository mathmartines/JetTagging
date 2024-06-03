from src.EFPDataGenerator import EFPDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from src.BestEFPsFinder import BestEFPsFinder


if __name__ == "__main__":
    # loading the data for the Decision Tree
    gen = EFPDataGenerator(
        top="../../../Data/t_jets_efp_d5.npy",
        gluon="../../../Data/g_jets_efp_d5.npy",
        quark="../../../Data/q_jets_efp_d5.npy"
    )
    X, y = gen.create_data(class_definitions=[('top',), ('gluon', 'quark')])
    # Dividing the data into trainning, validation, and test
    # diving the set into trainning, validation, and test
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state=42)

    # defining the model
    dt_model = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=70, random_state=0)
    efps_finder = BestEFPsFinder(
        tree=dt_model, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val
    )

    print("Evaluating the best set of features")
    print(efps_finder.backward_elimination(accuracy_fraction_limit=0.99))
