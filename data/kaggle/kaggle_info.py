import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

pd.set_option('display.max_rows', 10)

# create subsets from kaggle_all_data_info.csv

"""
done already by filter because not in genres.index ?
# drop all rows without genre or style
kaggle_dataset = kaggle_dataset[not kaggle_dataset['genre'].isna()]
kaggle_dataset = kaggle_dataset[not kaggle_dataset['style'].isna()]
"""

#"""
# filter dataset by small genres and small styles

kaggle_dataset = pd.read_csv('kaggle_train_info (nfs_data).csv')

genres = kaggle_dataset.groupby('genre').count()['filename'].sort_values()

genres = genres[genres >= 100] # >= 100 + miniature genre because test set would contain only 1

kaggle_dataset = kaggle_dataset[kaggle_dataset['genre'].isin(genres.index)]

styles = kaggle_dataset.groupby('style').count()['filename'].sort_values()
styles = styles[styles >= 100]

kaggle_dataset = kaggle_dataset[kaggle_dataset['style'].isin(styles.index)]

print("Genres:")
genres = kaggle_dataset.groupby('genre').count()['filename'].sort_values()
print(genres)

print('Styles:')
styles = kaggle_dataset.groupby('style').count()['filename'].sort_values()
print(styles)

#kaggle_dataset = kaggle_dataset.drop('in_train', axis=1)
#kaggle_dataset = kaggle_dataset.drop('artist_group', axis=1)

train_kaggle, test_kaggle = train_test_split(kaggle_dataset, test_size=6000, shuffle=True, random_state=2)

train_kaggle.to_csv('kaggle_art_dataset_train.csv')
test_kaggle.to_csv('kaggle_art_dataset_test.csv')


#genres.to_csv('genres.csv')
#styles.to_csv('styles.csv')
#"""

#
#"""
kaggle_dataset = pd.read_csv('kaggle_art_dataset_train.csv')

print('\n train dataset \n')
print(kaggle_dataset.columns)

print(f"Number data entries: {len(kaggle_dataset.index)}") #, test entries: {len(kaggle_dataset[kaggle_dataset['in_train'] == False].index)}, train entries: {len(kaggle_dataset[kaggle_dataset['in_train'] == True].index)}")
print(f"Number of artists: {len(kaggle_dataset['artist'].drop_duplicates().index)}, Number of genres: {kaggle_dataset['genre'].drop_duplicates().size}, Number of styles: {kaggle_dataset['style'].drop_duplicates().size}")
print("--------------")
print("Genres:")
genres = kaggle_dataset.groupby('genre').count()['filename'].sort_values()
print(genres)
print("--------------")
print('Styles:')
styles = kaggle_dataset.groupby('style').count()['filename'].sort_values()
print(styles)


kaggle_dataset = pd.read_csv('kaggle_art_dataset_test.csv')

print("\n test dataset \n")
print(kaggle_dataset.columns)

print(f"Number data entries: {len(kaggle_dataset.index)}") #, test entries: {len(kaggle_dataset[kaggle_dataset['in_train'] == False].index)}, train entries: {len(kaggle_dataset[kaggle_dataset['in_train'] == True].index)}")
print(f"Number of artists: {len(kaggle_dataset['artist'].drop_duplicates().index)}, Number of genres: {kaggle_dataset['genre'].drop_duplicates().size}, Number of styles: {kaggle_dataset['style'].drop_duplicates().size}")
print("--------------")
print("Genres:")
genres = kaggle_dataset.groupby('genre').count()['filename'].sort_values()
print(genres)
print("--------------")
print('Styles:')
styles = kaggle_dataset.groupby('style').count()['filename'].sort_values()
print(styles)
#"""