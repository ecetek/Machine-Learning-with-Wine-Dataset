from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate
from tabulate import tabulate
import numpy as np


# verisetinin yüklenmesi
myData =load_wine()
# X matrisine  özelliklerin aktarılması
x = myData.data
# y dizisine etiketlerinin atanması
y = myData.target
 
# Karar Ağacı Sınıflandırıcısının Modelinin Oluşturulması
clf = DecisionTreeClassifier(random_state=0)
# Verinin %70'ini Eğitim, %20'sini test verisi olarak ayırıyoruz
X_train, X_test, y_train, y_test = train_test_split(x,y, train_size = 0.7, test_size = 0.2, random_state = 0)
# Eğitim Verisi ile eğitimi gerçekleştiriyoruz
clf.fit(X_train,y_train)
test_sonuc = clf.predict(X_test)
print(test_sonuc)
cm = confusion_matrix(y_test, test_sonuc)
#hata matrisi yazdırılıyor
plt.matshow(cm)
plt.title('Confusion matrix Decision Tree')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
print(cm)
scores = cross_val_score(clf, x, y, cv=10)
# sınıflandırıcının başarı değerleri ölçülüp tabloya kaydediliyor
scoring = ['precision_macro', 'recall_macro']
scores1 = cross_validate(clf, x, y, scoring=scoring)
result=tabulate({"Accuracy":scores,"Precision":scores1["test_precision_macro"],"Recall":scores1["test_recall_macro"]},headers="keys")
print(result)

 
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train) 
knn.predict(X_test)
knn.score(X_test, y_test)
cm = confusion_matrix(y_test, test_sonuc)

plt.matshow(cm)
plt.title('Confusion matrix KNN ')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
print(cm)
scores2 = cross_val_score(knn, x, y, cv=10)
scoring = ['precision_macro', 'recall_macro']
scores3 = cross_validate(clf, x, y, scoring=scoring)
result=tabulate({"Accuracy":scores2,"Precision":scores3["test_precision_macro"],"Recall":scores3["test_recall_macro"]},headers="keys")
print(result)
 

#karar agaclari hem sayisal hem kategorik verileri isleyebilir
girdi=input("13 özellik değerini aralara virgül koyarak giriniz:")
values=girdi.split(",")#girdiyi uygun formata getiriyor
values1=[0]*13
for i in range(13):
    values1[i]=values[i]
values1=np.reshape(values1,(1,-1))#girdiyi şekillendiriyor
result=clf.predict(values1)#tahmin etme işlemi
print("Tahmin edilen şarap türü:",result)

