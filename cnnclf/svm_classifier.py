from __future__ import print_function
import pickle
import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn import svm
from sklearn import preprocessing
from sklearn import decomposition
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title,fontsize=30)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['scientific','humanity'], rotation=45,fontsize=25)
    plt.yticks(tick_marks, ['scientific','humanity'],fontsize=25)
    plt.tight_layout()
    plt.ylabel('True label',fontsize=28)
    plt.xlabel('Predicted label',fontsize=28)

def load_data():

    with open('./data/features/train_fts.p','rb') as f:
        Xt = pickle.load(f)
    with open('./data/features/train_lbs.p','rb') as f:
        Yt = pickle.load(f)
    with open('./data/features/test_fts.p','rb') as f:
        Xtest = pickle.load(f)
    with open('./data/features/test_lbs.p','rb') as f:
        Ytest = pickle.load(f)
    #data preprocessing to eliminate 0s columns
    mod_x = np.abs(Xt)
    sums = np.sum(Xt,axis=0)
    constants = np.logical_not(sums == 0)
    Xt = Xt[:,constants]
    Xtest = Xtest[:,constants]

    return Xt,Yt,Xtest,Ytest


def train_classifier(Xt,Yt):
    n_comp = 700
    anova_filter = SelectKBest(f_classif, k=n_comp)


    clf = svm.SVC(kernel='rbf',C=3.8,gamma=0.0008)
    anova_svm = make_pipeline(anova_filter, preprocessing.StandardScaler(),clf)

    print('Cross validating ...')
    anova_score = cross_val_score(anova_svm,Xt,Yt,cv=5,n_jobs=-1)
    #simple_score = cross_val_score(clf,Xt,Yt,cv=5,n_jobs=-1)

    print('anova_svm cv score is {}'.format(anova_score.mean()))
    #print('simple_svm score is {}'.format(simple_score.mean()))

    anova_svm.fit(Xt,Yt)

    return anova_svm

def test_classifier(clf, Xtest, Ytest):

    Y_pred = clf.predict(Xtest)
    # make confusion matrix
    print('anova confusion_matrix\n')
    cm = metrics.confusion_matrix(Ytest,Y_pred)

    test_score_a = clf.score(Xtest,Ytest)
    #later, for looking at where the error are
    #positions = np.where( np.logical_not( Ytest == Y_pred ) )

    #test_path = pickle.load(open('./test_path.p','rb'))

    #out_file = open('wrong_position.txt','wb')

    #for i in range(len(positions[0])):
    #    out_file.write('{}\n'.format(test_path[positions[0][i]]))

    # Plot Confusion Matrix
    plt.figure()
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plot_confusion_matrix(cm_normalized)
    plt.show()

    print('anova_svm test score is {}'.format(test_score_a))
    #print('simple_svm test score is {}'.format(test_score))

def main():

    Xt, Yt, Xtest, Ytest = load_data()
    clf = train_classifier(Xt,Yt)
    test_classifier(clf,Xtest,Ytest)


if __name__ == "__main__":
    main()
