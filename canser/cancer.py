import pandas as pd
can_cer = pd.read_csv('breast_cancer_wisconsin_data.csv')

can_cer.head()


import numpy as np
X = np.array(can_cer[["radius_mean"]])
y = np.array(can_cer[["diagnosis"]])


print(X.shape)
print(y.shape)


# カテゴリ値の数値化
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(["B", "M"])                         # 良性：0, 悪性：1
y = le.transform(y.flatten())

# 数値化した状態を確認してみる
print(y)



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, train_size = 0.8, shuffle = True)


from sklearn.svm import SVC
classifier = SVC(kernel = "linear")

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print(y_pred)
print(y_test)


from sklearn import metrics

print(metrics.accuracy_score(y_test, y_pred))

print(metrics.confusion_matrix(y_test, y_pred))

print(metrics.classification_report(y_test, y_pred))
