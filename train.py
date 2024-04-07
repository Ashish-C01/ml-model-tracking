import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, roc_curve
import json
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
df = pd.read_csv('data_processed.csv')
y = df.pop("cons_general").to_numpy()
y[y < 4] = 0
y[y >= 4] = 1

X = df.to_numpy()
X = preprocessing.scale(X)
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
X = imp.fit_transform(X)

clf = LogisticRegression()
yhat = cross_val_predict(clf, X, y, cv=5)
acc = np.mean(yhat == y)
tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
specificity = tn/(tn+fp)
sensitivity = tp/(tp+fn)
with open('metrics.json', 'w') as f:
    json.dump({'accuracy': acc, "specificity": specificity,
              "sensitivity": sensitivity}, f)

score = yhat == y
score_int = [int(s) for s in score]
df['pred_accuracy'] = score_int

# Bar plot
sns.set_color_codes('dark')
ax = sns.barplot(x='region', y='pred_accuracy', data=df, palette='Greens_d')
ax.set(xlabel='Region', ylabel='Model accuracy')
plt.savefig('by_region.png', dpi=80)
