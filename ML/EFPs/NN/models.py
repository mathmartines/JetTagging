# models
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling1D, Conv1D, BatchNormalization, InputLayer


#Função para construir o modelo a partir do
# paper: https://arxiv.org/pdf/1712.07124
# descrição na Pag.34
def model_efp(input_shape):
    model = Sequential()
    
    # Camadas densas
    model.add(Dense(100, input_shape=input_shape, activation='relu'))
    model.add(Dropout(0.4))
    
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.3))
    
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.2))
    
    model.add(Dense(1, activation='sigmoid'))
    
    return model



# Função para construir o modelo (a)
# paper: https://journals.aps.org/prd/pdf/10.1103/PhysRevD.101.056019
# Fig.2 (a)
def build_model_a_wBN(input_shape):
  
    model = Sequential()
    
    # Primeira camada Conv1D para simular EdgeConv
    model.add(Conv1D(64, 16, activation='relu', input_shape=input_shape))
    model.add(Conv1D(64, 1, activation='relu'))
    model.add(Conv1D(64, 1, activation='relu'))
    model.add(BatchNormalization())
    
    # Segunda camada Conv1D para simular EdgeConv
    model.add(Conv1D(128, 16, activation='relu'))
    model.add(Conv1D(128, 1, activation='relu'))
    model.add(Conv1D(128, 1, activation='relu'))
    model.add(BatchNormalization())
    
    # Terceira camada Conv1D para simular EdgeConv
    model.add(Conv1D(256, 16, activation='relu'))
    model.add(Conv1D(256, 1, activation='relu'))
    model.add(Conv1D(256, 1, activation='relu'))
    model.add(BatchNormalization())
    
    # Global Average Pooling
    model.add(GlobalAveragePooling1D())
    
    # Camadas densas
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid'))
    
    return model

def build_model_a(input_shape):
  
    model = Sequential()
    
    # Primeira camada Conv1D para simular EdgeConv
    model.add(InputLayer(shape=input_shape))
    model.add(Conv1D(64, 16, activation='relu'))
    model.add(Conv1D(64, 1, activation='relu'))
    model.add(Conv1D(64, 1, activation='relu'))
    
    # Segunda camada Conv1D para simular EdgeConv
    model.add(Conv1D(128, 16, activation='relu'))
    model.add(Conv1D(128, 1, activation='relu'))
    model.add(Conv1D(128, 1, activation='relu'))
    
    # Terceira camada Conv1D para simular EdgeConv
    model.add(Conv1D(256, 16, activation='relu'))
    model.add(Conv1D(256, 1, activation='relu'))
    model.add(Conv1D(256, 1, activation='relu'))
    
    # Global Average Pooling
    model.add(GlobalAveragePooling1D())
    
    # Camadas densas
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid'))
    
    return model


#Função para construir o modelo (b)
# paper: https://journals.aps.org/prd/pdf/10.1103/PhysRevD.101.056019
# Fig.2 (b)
def build_model_b(input_shape):
    model = Sequential()
    
    # Primeira camada Conv1D para simular EdgeConv
    model.add(Conv1D(32, 7, activation='relu', input_shape=input_shape))
    model.add(Conv1D(32, 1, activation='relu'))
    model.add(Conv1D(32, 1, activation='relu'))
    
    # Segunda camada Conv1D para simular EdgeConv
    model.add(Conv1D(64, 7, activation='relu'))
    model.add(Conv1D(64, 1, activation='relu'))
    model.add(Conv1D(64, 1, activation='relu'))
    
    # Global Average Pooling
    model.add(GlobalAveragePooling1D())
    
    # Camadas densas
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(2, activation='softmax'))
    
    return model


# modelo do paper: https://arxiv.org/pdf/1704.02124
# descrição página 6
def build_model_DNN(input_shape):

    model = Sequential()
    
    # Primeira camada de entrada
    model.add(Dense(300, activation='relu', input_shape=input_shape))
    model.add(Dense(102, activation='relu'))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(6, activation='relu'))
    
    # Camada de saída
    model.add(Dense(1, activation='sigmoid'))
    

    return model
