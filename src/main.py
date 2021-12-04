import warnings
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier as ORC
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import get_images
import get_landmarks
import performance_plots

import numpy as np
np.set_printoptions(suppress=True)


''' Load the data and labels for keystrokes '''
image_directory = "../keystroke.csv"
data = pd.read_csv(image_directory)
subjects = data["person"].unique()

class SVMDetector:
    def __init__(self, subjects):
        self.user_scores = []
        self.imposter_scores = []
        self.mean_vector = []
        self.subjects = subjects

    def training(self):
        self.clf = OneClassSVM(kernel='rbf', gamma=26)
        self.clf.fit(self.train)

    def testing(self):
        self.user_scores = -self.clf.decision_function(self.test_genuine)
        self.imposter_scores = -self.clf.decision_function(self.test_imposter)
        self.user_scores = list(self.user_scores)
        self.imposter_scores = list(self.imposter_scores)

    def evaluate(self):
        for subject in subjects:
            genuine_user_data = data.loc[data.person == subject,
                                         "H.9":"H.enter"]
            imposter_data = data.loc[data.person != subject, :]

            self.train = genuine_user_data[:50]
            self.test_genuine = genuine_user_data[50:]
            self.test_imposter = imposter_data.groupby(
                "person").head(5).loc[:, "H.9":"H.enter"]

            self.training()
            self.testing()
            
        return self.user_scores, self.imposter_scores 


def warn(*args, **kwargs):
    pass


def split_dataset(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.10, random_state=42)

    return X_train, X_test, X_val, y_train, y_test, y_val


def train_kNN_NB(X_train, y_train, X_val, y_val, X_test, y_test):
    ''' Matching and Decision - Classifer 1 '''
    clf = ORC(knn())
    clf.fit(X_train, y_train)
    matching_scores_knn = clf.predict_proba(X_val)

    ''' Matching and Decision - Classifer 2 '''
    clf = ORC(GaussianNB())
    clf.fit(X_train, y_train)
    matching_scores_NB = clf.predict_proba(X_val)

    ''' Fusing score kNN and NB of validated data and calculate the threshold'''
    matching_scores_knn_NB = np.mean(
        np.array([matching_scores_knn, matching_scores_NB]), axis=0)

    # Tuning the sytem
    gen_scores = []
    imp_scores = []
    classes = clf.classes_
    matching_scores_knn_NB = pd.DataFrame(
        matching_scores_knn_NB, columns=classes)

    for i in range(len(y_val)):
        scores = matching_scores_knn_NB.loc[i]
        mask = scores.index.isin([y_val[i]])
        gen_scores.extend(scores[mask])
        imp_scores.extend(scores[~mask])

    threshold_knn_NB = performance_plots.performance(
        gen_scores, imp_scores, 'kNN_NB_decision_fusion', 100)

    '''---'''

    # Testing the system - getting a decision for kNN classsifier
    matching_scores_knn = clf.predict_proba(X_test)

    # Testing the system - getting a decision for NB classsifier
    matching_scores_NB = clf.predict_proba(X_test)

    ''' Fusing score kNN and NB of testing data'''
    matching_scores_knn_NB = np.mean(
        np.array([matching_scores_knn, matching_scores_NB]), axis=0)
    matching_scores_knn_NB = pd.DataFrame(
        matching_scores_knn_NB, columns=classes)

    for i in range(len(y_test)):
        scores = matching_scores_knn_NB.loc[i]
        mask = scores.index.isin([y_test[i]])
        gen_scores_knn_NB.extend(scores[mask])
        imp_scores_knn_NB.extend(scores[~mask])

    return threshold_knn_NB, gen_scores_knn_NB, imp_scores_knn_NB


def fuse_decision(gen_scores_knn_NB, imp_scores_knn_NB, gen_scores_svm, imp_scores_svm, threshold_knn_NB, threshold_svm):
    ''' Fuse decisions '''
    correct_authentications = 0
    
    range1 = len(gen_scores_knn_NB) if len(gen_scores_knn_NB) < len(gen_scores_svm) else len(gen_scores_knn_NB)
    range2 = len(imp_scores_knn_NB) if len(imp_scores_knn_NB) < len(imp_scores_svm) else len(imp_scores_svm)
    
    for i in range(range1):
        decision_knn_NB = False
        decision_svm = False
        if gen_scores_knn_NB[i] >= threshold_knn_NB:
            decision_knn_NB = True
            if gen_scores_svm[i] >= threshold_svm:
                decision_svm = True
                if decision_knn_NB and decision_svm:
                    correct_authentications += 1

    for i in range(range2):
        decision_knn_NB = False
        decision_svm = False
        if imp_scores_knn_NB[i] < threshold_knn_NB:
            decision_knn_NB = True
            if imp_scores_svm[i] < threshold_svm:
                decision_svm = True
                if decision_knn_NB and decision_svm:
                    correct_authentications += 1

    all_authentications = len(gen_scores_knn_NB) + len(imp_scores_knn_NB)
    accuracy = correct_authentications / all_authentications

    return accuracy


def main():
    warnings.warn = warn

    ''' Load the data and their labels for keystrokes'''
    image_directory = '../Project 1 Database'
    face_imgs, labels, imgs_quality = get_images.get_images(image_directory)

    ''' Get distances between face landmarks in the images '''
    landmarks, labels = get_landmarks.get_landmarks(
        face_imgs, labels, 'landmarks/', 68, False)
    
    ''' Split dataset '''
    X_train, X_test, X_val, y_train, y_test, y_val = split_dataset(landmarks, labels)
    threshold_knn_NB, gen_scores_knn_NB, imp_scores_knn_NB = train_kNN_NB(X_train, y_train, X_val, y_val, X_test, y_test)

    
    ''' Get genuine and impostor scores for keystrokes '''
    gen_scores_svm, imp_scores_svm = SVMDetector(subjects).evaluate()
    threshold_svm = performance_plots.performance(gen_scores_svm, imp_scores_svm, 'SVM_decision', 100)
    
    accuracy = fuse_decision(gen_scores_knn_NB, imp_scores_knn_NB,
                                  gen_scores_svm, imp_scores_svm, threshold_knn_NB, threshold_svm)
    
    print("Accuracy: %.6f" % accuracy)



main()
