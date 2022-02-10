import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold



plays_df = pd.read_csv('NFL Play by Play data (2009-18).csv')

play_types = {'pass':0,'run':1,'field_goal':2,'punt':3}

plays_df = plays_df[plays_df.play_type.isin(list(play_types.keys()))]
plays_df = plays_df.dropna()
convert_dict = {'ydstogo_field': int,
'ydstogo_down': int,
'game_seconds_remaining':int,
'down':int,
'score_differential_post':int,
'half_seconds_remaining':int
}
plays_df = plays_df.astype(convert_dict)

plays_df["is_1st_down"] = [1 if elem == 1 else 0 for elem in list(plays_df.down)]
plays_df["is_2nd_down"] = [1 if elem == 2 else 0 for elem in list(plays_df.down)]
plays_df["is_3rd_down"] = [1 if elem == 3 else 0 for elem in list(plays_df.down)]
plays_df["is_4th_down"] = [1 if elem == 4 else 0 for elem in list(plays_df.down)]
plays_df["is_home_team"] = [1 if elem == 'home' else 0 for elem in list(plays_df.posteam_type)]

features = ['ydstogo_down','ydstogo_field','half_seconds_remaining','game_seconds_remaining','score_differential_post',
'is_1st_down','is_2nd_down','is_3rd_down','is_4th_down','is_home_team']

plays_X = plays_df[features]
plays_X = np.array(plays_X)

plays_Y = plays_df.play_type
plays_Y = [play_types[elem] for elem in list(plays_Y)]
plays_Y = np.array(plays_Y)

#train test split

train_split = 0.8
X_train, X_test, Y_train, Y_test = train_test_split(plays_X, plays_Y, train_size=train_split,shuffle=True)

k_fold = False
if k_fold:
    kf = KFold(n_splits=10)

    curr_fold = 0
    accuracy = []
    for train_idx, val_idx in kf.split(X_train):

        curr_fold = curr_fold + 1
        print(f'Current fold: {curr_fold}')

        X_train_fold, X_val_fold = X_train[train_idx,:], X_train[val_idx,:]
        Y_train_fold, Y_val_fold = Y_train[train_idx], Y_train[val_idx]

        model = RandomForestClassifier()
        # Train the model on training data
        model.fit(X_train_fold, Y_train_fold);

        predictions = model.predict(X_val_fold)

        accuracy.append(sum(predictions == Y_val_fold) / len(predictions))


#Overall model building

model = RandomForestClassifier()

model.fit(X_train, Y_train)

predictions = model.predict(X_test)

accuracy = sum(predictions == Y_test) / len(predictions)


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

CM = confusion_matrix(Y_test,predictions)


disp = ConfusionMatrixDisplay(confusion_matrix=CM, display_labels=list(play_types.keys()))
disp.plot()
plt.show()






cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X_train, Y_train, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')


print(2)