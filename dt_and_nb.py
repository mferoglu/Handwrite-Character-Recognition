
def calculate_mem_use(snapshot, key_type='lineno', func=""):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)
    total = 0
    for index, stat in enumerate(top_stats, 1):
        frame = stat.traceback[0]
        if func in frame.filename or "data" in frame.filename:
            total = total + stat.size
    return total


import time
import tracemalloc
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from time import perf_counter
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


raw_data = pd.read_csv('data//A_Z Handwritten Data.csv')
print(raw_data.info())

pca_model = PCA(n_components = 128)
components = pca_model.fit_transform(raw_data)
sampled_data = pd.DataFrame(data = components)

min_max_scaler = preprocessing.MinMaxScaler()
scaled = min_max_scaler.fit_transform(sampled_data)
sampled_data = pd.DataFrame(scaled)
print(sampled_data.head(5))
X = sampled_data


Y = raw_data["0"]
del raw_data



X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1) # 70% training and 30% test

word_dict = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
print("Success!")



#-------------------------------------Decision Tree With Gain---------------------------------------------------------------------------------------------------------
print("start")
start_timer= perf_counter()
tracemalloc.start()
clf = DecisionTreeClassifier(criterion="entropy")
clf = clf.fit(X_train,y_train)
y_pred_gain = clf.predict(X_test)
snapshot = tracemalloc.take_snapshot()
mem_use = calculate_mem_use(snapshot, func="entropy")
print("Mem. use in DT(Entropy): %1.f B" %(mem_use))

end_timer= perf_counter()
print("\n")
print(f"Execution time of Decision Tree with Entropy Criterion {end_timer-start_timer:0.4f} seconds")
print("\n")
print("Decision Tree with Entropy Criterion:",metrics.accuracy_score(y_test, y_pred_gain))
print("\n")
print( classification_report(y_test, y_pred_gain,target_names = word_dict))

fig, ax = plt.subplots(figsize=(20,10)) 

cf_matrix = confusion_matrix(y_test,y_pred_gain)
sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
            fmt='.2%', cmap='Blues',xticklabels=word_dict, yticklabels=word_dict)

plt.title("Decision Tree with entropy criterion")

plt.show()
plt.close()
print("end")





#-------------------------------------Decision Tree With Gini---------------------------------------------------------------------------------------------------------
print("start")
start_timer= perf_counter()
tracemalloc.start()
clf_gini = DecisionTreeClassifier(criterion="gini")
clf_gini = clf_gini.fit(X_train,y_train)
snapshot = tracemalloc.take_snapshot()
mem_use = calculate_mem_use(snapshot, func="gini")
y_pred_gini = clf_gini.predict(X_test)
print("Mem. use DT(Gini): %1.f B" %(mem_use))
end_timer= perf_counter()
print("\n")
print(f"Execution time of Decision Tree With Gini Index {end_timer-start_timer:0.4f} seconds")
print("\n")
print("Decision Tree With Gini Index Accuracy:",metrics.accuracy_score(y_test, y_pred_gini))
print("\n")
print( classification_report(y_test, y_pred_gini,target_names = word_dict))


fig, ax = plt.subplots(figsize=(20,10)) 

cf_matrix = confusion_matrix(y_test,y_pred_gini)
sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
            fmt='.2%', cmap='Blues',xticklabels=word_dict, yticklabels=word_dict)

plt.title("Decision Tree with gini criterion")

plt.show()
plt.close()
print("end")


print("\n")
print("----------------------- NAIVE BAYES -----------------------")
start_timer= perf_counter()
tracemalloc.start()
nb=GaussianNB()
nb.fit(X_train, y_train) #train data
y_pred=nb.predict(X_test)  #the result of training (after test) 
snapshot = tracemalloc.take_snapshot()
mem_use = calculate_mem_use(snapshot, func="nb")
print("Mem. use Naive Bayes: %1.f B" %(mem_use))
end_timer=perf_counter()
time.sleep(3)

cf_matrix1 = confusion_matrix(y_test,y_pred)
print(f"Execution time of Naive Bayes {end_timer-start_timer:0.4f} seconds")
print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred, normalize=True))
print("\n")
print(  classification_report(y_test, y_pred,target_names = word_dict))


fig, ax = plt.subplots(figsize=(20,10)) 
sns.heatmap(cf_matrix1/np.sum(cf_matrix1), annot=True, 
            fmt='.2%', cmap='Blues',xticklabels=word_dict, yticklabels=word_dict)

plt.title("Naive Bayes")

plt.show()
plt.close()
print("end")
