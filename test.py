# https://pypi.org/project/gensim/
# Did not use openblas or ATLAS
# Installed smart_open https://radimrehurek.com/gensim/
# https://pypi.org/project/gensim/ citing gensim !!!!???
# https://radimrehurek.com/gensim/models/word2vec.html
# Linked tutorial from 2014 says need Cython for multiple workers
# I installed it successfully, don't know if I have MSVC, can use workers = 2 with or without
# Cython installed
# Site itself doesn't mention Cython
# Maybe newer version doesn't need it?

# Was his word2vec code included?-> don't think so

# START HERE:

# How do you store large amounts of embedded documents before operating on them?
# Within a pytorch dataset?
# Can store however you want, just make sure to return one X and one y in __getitem__()
# Probably not too large, gensim embedding can be done separately
# What about before the dataset is made

# Maybe make at least one CNN, one baseline
# One CNN is fine for draft, maybe a few for final, don't need every one

# How close????

# 70% train, 10% validation, 20% test???? Hyperparameters????

# How many baselines -> Don't need to do them. Maybe do a few if we have time. Use their hyperparameters for baselines

# Start with default vector size for embedding

# Try to be close to original preprocessing -> don't need to be exactly the same

# Ask Luke:
# Preprocessing???
# Has "Dementia" field, but not investigated?
# .lower() was commented out???

import numpy as np
import pandas as pd

clinical_notes_df = pd.read_csv("NOTEEVENTS.csv")

print(clinical_notes_df.shape)
print(clinical_notes_df.head(1))

# clinical_notes_df.groupby(["HADM_ID", "SUBJECT_ID"]).size().reset_index().rename(columns = {0 : 'count'})
# combinations of HADM_ID and SUBJECT_ID are not unique
# HADM_IDs alone aren't unique, either

# annotations_df = pd.read_csv("annotations.csv")
# max(annotations_df.groupby(["Hospital.Admission.ID", "subject.id"]).size().reset_index().rename(columns = {0 : 'count'})["count"])
# 3 -> so combinations of "Hospital.Admission.ID" and "subject.id" aren't unique

# temp = annotations_df.groupby(["Hospital.Admission.ID", "subject.id"]).size().reset_index().rename(columns = {0 : 'count'})

# merged = pd.merge(temp, clinical_notes_df, left_on = ["Hospital.Admission.ID", "subject.id"], right_on = ["HADM_ID", "SUBJECT_ID"])
# # 52870 notes

# np.sum(annotations_df.groupby(["Hospital.Admission.ID", "subject.id"]).size().reset_index().rename(columns = {0 : 'count'})["count"] == 1)
# 1517 / 1610 combinations of "Hospital.Admission.ID" and "subject.id"
# appear only once

# np.sum(annotations_df.groupby(["Hospital.Admission.ID", "subject.id"]).size().reset_index().rename(columns = {0 : 'count'})["count"] == 1)
# 39 / 1610 combinations of "Hospital.Admission.ID" and "subject.id"
# appear twice

# np.sum(annotations_df.groupby(["Hospital.Admission.ID", "subject.id"]).size().reset_index().rename(columns = {0 : 'count'})["count"] == 1)
# 5 / 1610 combinations of "Hospital.Admission.ID" and "subject.id"
# appear 3 times

# 1517 + 39 * 2 + 5 * 3 = 1610 (correct)

# len(pd.unique(merged["HADM_ID"]))
# 1561
# This equals 1517 + 39 + 5

# len(pd.unique(merged["SUBJECT_ID"]))
# 1045

# merged.loc[(merged["HADM_ID"] == 116034) & (merged["SUBJECT_ID"] == 5205) & (merged["CATEGORY"] == "Discharge summary")]
# Have 4 discharge summaries for this combination of HADM_ID and subject ID, only 3 in the 1610
# merged.loc[(merged["HADM_ID"] == 116034) & (merged["SUBJECT_ID"] == 5205) & (merged["CATEGORY"] == "Discharge summary")]["CHARTDATE"]
# 8220    2139-01-14
# 8221    2138-12-31
# 8222    2139-01-06
# 8223    2139-01-12

# merged.loc[merged["CATEGORY"] == "Discharge summary"].groupby(["HADM_ID", "SUBJECT_ID"]).size().reset_index().rename(columns = {0 : 'count'})
# Only 1560?

# Representing documents?

# Where model is a word2vec model
# example_doc = ["Lotus Exige", "rear wheel drive", "Porsche 718 Cayman"]
# np.stack([model.wv[word] for word in example_doc])
