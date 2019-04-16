#import important modules
import pandas as pd
from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")

# function to Load train dataset
def load_train_data(file_name):

    df = pd.read_csv(file_name)

    return df

# Function to Load validation dataset
def load_valid_data(file_name):

    df = pd.read_csv(file_name)

    return df


# function to load test data and prepara it to test in the model
def load_test_data(file_name):

    df = pd.read_csv(file_name)

    # drop some columns
    df.drop(['student_id', 'school_id', 'total_toilets', 'establishment_year', 'total_students'],
            axis=1, inplace=True)

    # convert these strings into integer keys
    le = preprocessing.LabelEncoder()
    df['gender'] = le.fit_transform(df['gender'])
    df['guardian'] = le.fit_transform(df['guardian'])
    df['internet'] = le.fit_transform(df['internet'])
    df['caste'] = le.fit_transform(df['caste'])

    return df


# function to load the test labels
def load_test_labels(labels_name):

    df = pd.read_csv(labels_name)

    # convert these strings into integer keys
    le = preprocessing.LabelEncoder()
    df['continue_drop'] = le.fit_transform(df['continue_drop'])

    return df


:
        
