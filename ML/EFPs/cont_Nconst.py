def calculate_num_constituents(jet_data):
    num_constituents = []
    
    for i in range(len(jet_data)):
        # Reshape jet data into a 2D array with 4 columns
        jato = jet_data.iloc[i].values.reshape(-1, 4)
        
        # Count the number of valid constituents (mask == 1)
        num_valid_constituents = sum(jato[:, 3] == 1)
        
        num_constituents.append(num_valid_constituents)
    
    return num_constituents

def process_num_constituents(data_quark, data_gluon, data_top):
    print("Loading data...")
    
    # Read data from CSV files
    df_quark = pd.read_csv(data_quark, header=None, sep=',')
    df_gluon = pd.read_csv(data_gluon, header=None, sep=',')
    df_top = pd.read_csv(data_top, header=None, sep=',')
    
    print("Calculating number of constituents for quark jets...")
    # Calculate number of constituents for quark jets
    num_constituents_quark = calculate_num_constituents(df_quark)
    
    print("Calculating number of constituents for gluon jets...")
    # Calculate number of constituents for gluon jets
    num_constituents_gluon = calculate_num_constituents(df_gluon)
    
    print("Calculating number of constituents for top jets...")
    # Calculate number of constituents for top jets
    num_constituents_top = calculate_num_constituents(df_top)
    
    return num_constituents_quark, num_constituents_gluon, num_constituents_top

# Directories for data storage
training_dir = '/eos/home-i03/m/mdemelop/SWAN_projects/ML/Seminario/Data/Trainning'
test_dir = '/eos/home-i03/m/mdemelop/SWAN_projects/ML/Seminario/Data/Test'
validation_dir = '/eos/home-i03/m/mdemelop/SWAN_projects/ML/Seminario/Data/Validation'

root_data_folder = '/eos/home-i03/m/mdemelop/SWAN_projects/ML/Seminario/Data/'

# Define data file paths
data_train_quark = f'{root_data_folder}Trainning/q_jets.csv'
data_train_gluon = f'{root_data_folder}Trainning/g_jets.csv'
data_train_top = f'{root_data_folder}Trainning/t_jets.csv'

data_test_quark = f'{root_data_folder}Test/q_jets.csv'
data_test_gluon = f'{root_data_folder}Test/g_jets.csv'
data_test_top = f'{root_data_folder}Test/t_jets.csv'

data_val_quark = f'{root_data_folder}Validation/q_jets.csv'
data_val_gluon = f'{root_data_folder}Validation/g_jets.csv'
data_val_top = f'{root_data_folder}Validation/t_jets.csv'

# Calling the function for training data
num_constituents_quark_train, num_constituents_gluon_train, num_constituents_top_train = process_num_constituents(
    data_quark=data_train_quark,
    data_gluon=data_train_gluon,
    data_top=data_train_top
)

# Calling the function for test data
num_constituents_quark_test, num_constituents_gluon_test, num_constituents_top_test = process_num_constituents(
    data_quark=data_test_quark,
    data_gluon=data_test_gluon,
    data_top=data_test_top
)

# Calling the function for validation data
num_constituents_quark_val, num_constituents_gluon_val, num_constituents_top_val = process_num_constituents(
    data_quark=data_val_quark,
    data_gluon=data_val_gluon,
    data_top=data_val_top
)

# Saving the results for each jet type separately
print('Saving trainning results')
np.save(os.path.join(training_dir, 'num_constituents_q.npy'), num_constituents_quark_train)
np.save(os.path.join(training_dir, 'num_constituents_g.npy'), num_constituents_gluon_train)
np.save(os.path.join(training_dir, 'num_constituents_t.npy'), num_constituents_top_train)
print('Saving testing results')
np.save(os.path.join(test_dir, 'num_constituents_q.npy'), num_constituents_quark_test)
np.save(os.path.join(test_dir, 'num_constituents_g.npy'), num_constituents_gluon_test)
np.save(os.path.join(test_dir, 'num_constituents_t.npy'), num_constituents_top_test)
print('Saving validation results')
np.save(os.path.join(validation_dir, 'num_constituents_q.npy'), num_constituents_quark_val)
np.save(os.path.join(validation_dir, 'num_constituents_g.npy'), num_constituents_gluon_val)
np.save(os.path.join(validation_dir, 'num_constituents_t.npy'), num_constituents_top_val)
