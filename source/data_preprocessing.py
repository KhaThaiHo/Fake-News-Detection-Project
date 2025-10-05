import pandas as pd
import regex as re

def clean_text(text):
    if not isinstance(text, str):
        return text
    # Delete \n, \t and replace multiple spaces with a single space
    text = re.sub(r'[\n\t\r]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def preprocessing_data(df):
    """
    Preprocess your data (eg. Drop null datapoints or fill missing data)
    :param df: pandas DataFrame
    :return: pandas DataFrame
    """
    # Todo: preprocess data
    df.dropna(inplace = True)
    if 'Unnamed: 0' in df.columns:
        df.drop('Unnamed: 0', axis = 1, inplace = True)
    
    df['text'] = df['text'].apply(clean_text)
    df.drop_duplicates(subset=['text'], keep='first', inplace=True)
    
    return df


def main():
    PATH1 = "DataSet_Misinfo_TRUE.csv"
    df_true = pd.read_csv(PATH1)
    df_true['target'] = 1

    PATH2 = "DataSet_Misinfo_FAKE.csv"
    df_fake = pd.read_csv(PATH2)
    df_fake['target'] = 0

    df = pd.concat([df_true, df_fake], ignore_index=True)