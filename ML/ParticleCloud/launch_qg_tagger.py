import pandas as pd
from tensorflow import keras
from src.ParticleCloud import EdgeConvolutionLayer, ChannelWiseGlobalAvaragePooling
from src.Preprocessing.JetPreprocessing import JetProcessingParticleCloud
from src.Particle import ParticleType
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, confusion_matrix, roc_curve, auc


if __name__ == '__main__':
    # loading all the data
    data_gluon = pd.read_csv('../../Data/g_jets.csv', header=None, sep=' ')
    data_quark = pd.read_csv('../../Data/q_jets.csv', header=None, sep=' ')

    # joinning data frames
    all_jets = pd.concat(
        [data_quark, data_gluon], axis=0)
    all_jets.reset_index(drop=True, inplace=True)
    # defining the dictionary with the order of jets in the full data frame
    jets_order = {
        ParticleType.LightQuark: (0, len(data_quark) - 1),
        ParticleType.Gluon: (len(data_quark), len(all_jets) - 1),
    }

    jet_preprocessing = JetProcessingParticleCloud()
    X = jet_preprocessing.transform(X=all_jets.to_numpy(), y=jets_order)
    y = jet_preprocessing.jet_labels

    # Dividing the data into trainning, validation, and test
    # diving the set into trainning, validation, and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)

    # first we need the MLPs
    # just including one layer
    mlp_jets_first = keras.Sequential([
        keras.layers.InputLayer(shape=[6]),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
    ])

    mlp_jets_second = keras.Sequential([
        # always remember that the input is 2x the output of the previous mlp
        keras.layers.InputLayer(shape=[64]),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
    ])

    jet_tag_model = keras.Sequential([
        keras.layers.InputLayer(shape=(30, 4)),
        EdgeConvolutionLayer(mlp=mlp_jets_first, k_neighbors=4, final_index_coord=2, max_number_particles=30),
        EdgeConvolutionLayer(mlp=mlp_jets_second, k_neighbors=4, final_index_coord=8, max_number_particles=30),
        ChannelWiseGlobalAvaragePooling(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(rate=0.05),
        keras.layers.Dense(2, activation='softmax')
    ])

    jet_tag_model.compile(optimizer='adam', loss='crossentropy', metrics=['accuracy', 'recall', 'precision'])
    early_stopping = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    jet_tag_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val),
                      callbacks=[early_stopping])

    # perfoming the prediction
    y_train_pred = jet_tag_model.predict(X_train)
    y_test_pred = jet_tag_model.predict(X_test)

    print("Trainnig set:")
    print(f"Recall for Quark tagging: {recall_score(y_train[:, 0], y_train_pred[:, 0] > 0.5):.2f}")
    print(f"Precision for Quark tagging: {precision_score(y_train[:, 0], y_train_pred[:, 0] > 0.5):.2f}")
    print("Confusion Matrix")
    print(confusion_matrix(y_train[:, 0], y_train_pred[:, 0] > 0.5, labels=[0, 1]))

    print("Test set:")
    print(f"Recall for Quark tagging: {recall_score(y_test[:, 0], y_test_pred[:, 0] > 0.5):.2f}")
    print(f"Precision for Quark tagging: {precision_score(y_test[:, 0], y_test_pred[:, 0] > 0.5):.2f}")
    print("Confusion Matrix")
    print(confusion_matrix(y_test[:, 0], y_test_pred[:, 0] > 0.5, labels=[0, 1]))

    # roc curve for top tagging
    fpr, tpr, _ = roc_curve(y_train[:, 0], y_train_pred[:, 0])
    print(f"AUC: {auc(fpr, tpr):.2f}")

    # roc curve for top tagging
    fpr, tpr, _ = roc_curve(y_test[:, 0], y_test_pred[:, 0])
    print(f"AUC Test: {auc(fpr, tpr):.2f}")
