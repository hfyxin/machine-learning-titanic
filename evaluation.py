from sklearn.model_selection import cross_validate
from numpy import mean as np_mean


def cross_val_info(model, features, labels, cv=4):
    '''
    Print cross validation accuracy result.
    '''
    cv_results = cross_validate(model, features, labels,
                            cv=4, return_train_score=True)
    print()
    for i in range(cv_results['fit_time'].shape[0]):
        print("CV {:1d}. Training score {:.3f}, testing score: {:.3f}".format(
                    i,cv_results['train_score'][i],cv_results['test_score'][i]))
    print("Average test score (probability): {:.3f}".format(
                    np_mean(cv_results['test_score'])))