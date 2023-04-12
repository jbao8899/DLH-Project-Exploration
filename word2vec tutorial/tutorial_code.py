from gensim.models import Word2Vec
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

df = pd.read_csv('data.csv')
print(df.head())

df['Maker_Model']= df['Make']+ " " + df['Model']

# Select features from original dataset to form a new dataframe 
df1 = df[['Engine Fuel Type', 'Transmission Type', 'Driven_Wheels', 'Market Category', 'Vehicle Size', 'Vehicle Style', 'Maker_Model']]

# For each row, combine all the columns into one column
df2 = df1.apply(lambda x: ','.join(x.astype(str)), axis=1)

# Store them in a pandas dataframe
df_clean = pd.DataFrame({'clean': df2})

# Create the list of list format of the custom corpus for gensim modeling 
sent = [row.split(',') for row in df_clean['clean']]

print("show the example of list of list format of the custom corpus for gensim modeling") 
print(sent[:2])
print()

# We can train the genism word2vec model with our own custom corpus as following:
model = Word2Vec(sent, min_count=1, vector_size=50, workers=3, window=3, sg=1)

print("model.wv['Toyota Camry']")
print(model.wv['Toyota Camry'])
print()

print("model.wv.similarity('Porsche 718 Cayman', 'Nissan Van')")
print(model.wv.similarity('Porsche 718 Cayman', 'Nissan Van'))
print()

print("model.wv.similarity('Porsche 718 Cayman', 'Mercedes-Benz SLK-Class')")
print(model.wv.similarity('Porsche 718 Cayman', 'Mercedes-Benz SLK-Class'))
print()

print("model.wv.most_similar('Mercedes-Benz SLK-Class', topn=5)")
for similar in model.wv.most_similar('Mercedes-Benz SLK-Class', topn=5):
    print(similar)
print()

print("Vocabulary Size")
print("len(model.wv)")
print(len(model.wv))
print("len(model.wv.most_similar('Mercedes-Benz SLK-Class', topn=None))")
print(len(model.wv.most_similar('Mercedes-Benz SLK-Class', topn=None)))
print()

mini_corpus = [['premium unleaded (required)', 'MANUAL', 'rear wheel drive', 'Factory Tuner', 'Luxury', 'High-Performance', 'Compact', 'Coupe', 'BMW 1 Series M'],
               ['premium unleaded (required)', 'MANUAL', 'rear wheel drive', 'Luxury', 'Performance', 'Compact', 'Convertible', 'BMW 1 Series']]

for doc in mini_corpus:
    print(len(doc))
    print(torch.stack([torch.from_numpy(model.wv[word]) for word in doc]).shape)

print(torch.stack([torch.from_numpy(model.wv[word]) for word in mini_corpus[0]]))
