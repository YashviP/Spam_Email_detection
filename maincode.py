import os
import time
import numpy as np
from collections import Counter
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.metrics import confusion_matrix

#dic is a dictionary containing all spam words
dic = dict()

#function to count files in a folder 
def count_files_in_folder(folder):
    list_of_files=os.listdir(folder)
    return len(list_of_files)


#function to create dictionary of spam words 

def make_Dictionary(folder):
    files = [os.path.join(folder,f) for f in os.listdir(folder)]   

    words = []       
    for file in files:   
        with open(file) as f:
            for i,line in enumerate(f):
                if i == 2:  
                    word= line.split()
                    words += words
    
    dic = Counter(words)


def word_count_vector(mail_dir): 
    ct=1
    print "----------------------------------------------"
    files = [os.path.join(mail_dir,fi) for fi in os.listdir(mail_dir)]
    files.sort()
    features_matrix = np.zeros((len(files),20488))
    docID = 0;
    for fil in files:
      ct=ct+1
      with open(fil) as fi:
        for i,line in enumerate(fi):
          if i == 2:
            words = line.split()
            for word in words:
              wordID = 0
              for i,d in enumerate(dic):
                if d[0] == word:
                  wordID = i
                  features_matrix[docID,wordID] = words.count(word)
        docID = docID + 1     
    return features_matrix



def classify(mat_train,mat_test):
    
    train_labels = np.zeros(702)
    train_labels[351:701] = 1
    # Training SVM and Naive bayes classifier

    model1 = MultinomialNB()
    model2 = LinearSVC()
    model1.fit(mat_train,train_labels)
    model2.fit(mat_train,train_labels)

    # Test the unseen mails for Spam
    test_labels = np.zeros(260)
    test_labels[130:260] = 1
  
    result1 = model1.predict(mat_test)
    result2 = model2.predict(mat_test)
    print confusion_matrix(test_labels,result1)
    print confusion_matrix(test_labels,result2)



def feature_extr(folder):
    feature_mat=word_count_vector(folder)
    return feature_mat
    

#main function 


train='/home/yashvi/spamEmail/train'
test='/home/yashvi/spamEmail/test'

make_Dictionary(train)
mat_train=feature_extr(train)
mat_test=feature_extr(test)


classify(mat_train,mat_test)


