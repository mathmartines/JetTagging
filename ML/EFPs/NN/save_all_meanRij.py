import os
import numpy as np
import pandas as pd

# Função deltaRij
def deltaRij(eta_i, phi_i, eta_j, phi_j):
    deta = eta_i - eta_j
    dphi = phi_i - phi_j
    dphi = np.where(dphi > np.pi, dphi - 2*np.pi, dphi)
    dphi = np.where(dphi < -np.pi, dphi + 2*np.pi, dphi)
    return np.sqrt(deta**2 + dphi**2)

# Função para calcular mean DeltaRij
def calculate_mean_deltaRij(jet_data):
    mean_deltaRij = []
    
    for i in range(len(jet_data)):
        # Transforma os dados de um jato em um array 2D com 4 colunas
        jato = jet_data.iloc[i].values.reshape(-1, 4)
        
        deltaRij_sum = 0
        count_pairs = 0
        
        # Calcula DeltaRij para todos os pares de constituintes no jato
        for k in range(len(jato)):
            for l in range(k + 1, len(jato)):
                deltaRij_sum += deltaRij(jato[k, 0], jato[k, 1], jato[l, 0], jato[l, 1])
                count_pairs += 1
        
        mean_deltaRij.append(deltaRij_sum / count_pairs)
    
    return mean_deltaRij

# Função para processar mean DeltaRij para os diferentes tipos de jet
def process_mean_deltaRij_top_tagging(data_orig_quark, data_orig_gluon, data_orig_top):
    print("Loading data...")
    
    # Leitura dos dados
    df_quark = pd.read_csv(data_orig_quark, header=None, sep='\s+')
    df_gluon = pd.read_csv(data_orig_gluon, header=None, sep='\s+')
    df_top = pd.read_csv(data_orig_top, header=None, sep='\s+')
    
    print("Calculating mean DeltaRij for quark jets...")
    # Cálculo do mean_deltaRij para quark jets
    mean_deltaRij_quark = calculate_mean_deltaRij(df_quark)
    
    print("Calculating mean DeltaRij for gluon jets...")
    # Cálculo do mean_deltaRij para gluon jets
    mean_deltaRij_gluon = calculate_mean_deltaRij(df_gluon)
    
    print("Calculating mean DeltaRij for top jets...")
    # Cálculo do mean_deltaRij para top jets
    mean_deltaRij_top = calculate_mean_deltaRij(df_top)
    
    return mean_deltaRij_quark, mean_deltaRij_gluon, mean_deltaRij_top

# Definindo os arquivos de dados
root_data_folder = '/eos/home-i03/m/mdemelop/SWAN_projects/ML/Seminario/Data/'

data_orig_quark = f'{root_data_folder}Originals/q_jets.csv'
data_orig_gluon = f'{root_data_folder}Originals/g_jets.csv'
data_orig_top = f'{root_data_folder}Originals/t_jets.csv'

data_train_quark = f'{root_data_folder}Trainning/q_jets.csv'
data_train_gluon = f'{root_data_folder}Trainning/g_jets.csv'
data_train_top = f'{root_data_folder}Trainning/t_jets.csv'

data_test_quark = f'{root_data_folder}Test/q_jets.csv'
data_test_gluon = f'{root_data_folder}Test/g_jets.csv'
data_test_top = f'{root_data_folder}Test/t_jets.csv'

data_val_quark = f'{root_data_folder}Validation/q_jets.csv'
data_val_gluon = f'{root_data_folder}Validation/g_jets.csv'
data_val_top = f'{root_data_folder}Validation/t_jets.csv'

# Chamando a função para os dados originais totais
mean_deltaRij_quark_orig, mean_deltaRij_gluon_orig, mean_deltaRij_top_orig = process_mean_deltaRij_top_tagging(
    data_orig_quark=data_orig_quark,
    data_orig_gluon=data_orig_gluon,
    data_orig_top=data_orig_top
)

# Chamando a função para os dados de treinamento
mean_deltaRij_quark_train, mean_deltaRij_gluon_train, mean_deltaRij_top_train = process_mean_deltaRij_top_tagging(
    data_orig_quark=data_train_quark,
    data_orig_gluon=data_train_gluon,
    data_orig_top=data_train_top
)

# Chamando a função para os dados de teste
mean_deltaRij_quark_test, mean_deltaRij_gluon_test, mean_deltaRij_top_test = process_mean_deltaRij_top_tagging(
    data_orig_quark=data_test_quark,
    data_orig_gluon=data_test_gluon,
    data_orig_top=data_test_top
)

# Chamando a função para os dados de validação
mean_deltaRij_quark_val, mean_deltaRij_gluon_val, mean_deltaRij_top_val = process_mean_deltaRij_top_tagging(
    data_orig_quark=data_val_quark,
    data_orig_gluon=data_val_gluon,
    data_orig_top=data_val_top
)

# Diretórios de destino
originals_dir = '/eos/home-i03/m/mdemelop/SWAN_projects/ML/Seminario/Data/Originals'
training_dir = '/eos/home-i03/m/mdemelop/SWAN_projects/ML/Seminario/Data/Trainning'
test_dir = '/eos/home-i03/m/mdemelop/SWAN_projects/ML/Seminario/Data/Test'
validation_dir = '/eos/home-i03/m/mdemelop/SWAN_projects/ML/Seminario/Data/Validation'

# Salvando os resultados para cada tipo de jet separadamente
np.save(os.path.join(originals_dir, 'mean_deltaRij_originals_q.npy'), mean_deltaRij_quark_orig)
np.save(os.path.join(originals_dir, 'mean_deltaRij_originals_g.npy'), mean_deltaRij_gluon_orig)
np.save(os.path.join(originals_dir, 'mean_deltaRij_originals_t.npy'), mean_deltaRij_top_orig)

np.save(os.path.join(training_dir, 'mean_deltaRij_training_q.npy'), mean_deltaRij_quark_train)
np.save(os.path.join(training_dir, 'mean_deltaRij_training_g.npy'), mean_deltaRij_gluon_train)
np.save(os.path.join(training_dir, 'mean_deltaRij_training_t.npy'), mean_deltaRij_top_train)

np.save(os.path.join(test_dir, 'mean_deltaRij_test_q.npy'), mean_deltaRij_quark_test)
np.save(os.path.join(test_dir, 'mean_deltaRij_test_g.npy'), mean_deltaRij_gluon_test)
np.save(os.path.join(test_dir, 'mean_deltaRij_test_t.npy'), mean_deltaRij_top_test)

np.save(os.path.join(validation_dir, 'mean_deltaRij_validation_q.npy'), mean_deltaRij_quark_val)
np.save(os.path.join(validation_dir, 'mean_deltaRij_validation_g.npy'), mean_deltaRij_gluon_val)
np.save(os.path.join(validation_dir, 'mean_deltaRij_validation_t.npy'), mean_deltaRij_top_val)
