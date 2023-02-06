
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

cancer_ds=datasets.load_breast_cancer()
cancer_ds

clf=LogisticRegression()
clf.fit(cancer_ds.data, cancer_ds.target)

clf.score(cancer_ds.data, cancer_ds.target)

clf.predict_proba(cancer_ds.data)[3]

#0-->correct classification
#non 0-->incorrect classification


clf.predict(cancer_ds.data) - cancer_ds.target