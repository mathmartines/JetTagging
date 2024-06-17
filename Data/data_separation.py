"""Breaks the dataset into Trainning, Validation and Testing sets"""
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    file_name = "t_jets"
    pd_data = pd.read_csv(f"{file_name}.csv", header=None, sep=" ")

    # Separating the data into train, validation and test
    # 15% for Testing, 85% for trainning
    data_train, data_test = train_test_split(pd_data, test_size=0.15, random_state=42)
    # Of the trainning set, let us keep 15% for validation
    data_train, data_valid = train_test_split(data_train, test_size=0.15, random_state=42)

    # saving the files
    data_train.to_csv(f"Trainning/{file_name}.csv", index=False, header=False)
    data_valid.to_csv(f"Validation/{file_name}.csv", index=False, header=False)
    data_test.to_csv(f"Test/{file_name}.csv", index=False, header=False)




