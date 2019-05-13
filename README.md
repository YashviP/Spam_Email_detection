# Spam_Email_detection



Spam filtering is a beginner’s example of document classification task which involves classifying an email as spam or non-spam (a.k.a. ham) mail. Spam box in your Gmail account is the best example of this. So lets get started in building a spam filter on a publicly available mail corpus. I have extracted equal number of spam and non-spam emails from Ling-spam corpus.

The emails in Ling-spam corpus have been already preprocessed in the following ways:

a) Removal of stop words – Stop words like “and”, “the”, “of”, etc are very common in all English sentences and are not very meaningful in deciding spam or legitimate status, so these words have been removed from the emails.

b) Lemmatization – It is the process of grouping together the different inflected forms of a word so they can be analysed as a single item. For example, “include”, “includes,” and “included” would all be represented as “include”. The context of the sentence is also preserved in lemmatization as opposed to stemming (another buzz word in text mining which does not consider meaning of the sentence).

Required Packages :

1. Numpy
2. os
3. collections
4. sklearn

Python version -2.7


Data sets:

1. training set - 702 files
2. testing set - 260 files


Here, I will be using scikit-learn ML library for training classifiers.I have trained two models here namely Naive Bayes classifier and Support Vector Machines (SVM). 

Test-set contains 130 spam emails and 130 non-spam emails. If you have come so far, you will find below results. I have shown the confusion matrix of the test-set for both the models. The diagonal elements represents the correctly identified(a.k.a. true identification) mails where as non-diagonal elements represents wrong classification (false identification) of mails.


       
