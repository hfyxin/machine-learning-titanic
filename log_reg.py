from preprocessing import titanic_pipeline
from preprocessing import load_data
from evaluation import cross_val_info
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

# load data
titanic_features, titanic_labels = load_data("data/train.csv")
titanic_features.info()

# preprocessing pipeline
pipe = titanic_pipeline()
feature_prepared = pipe.fit_transform(titanic_features)

# The logistic regression method
log_reg = LogisticRegression(solver='liblinear')
model = log_reg

# Grid search
param_grid = [
    {'solver':['liblinear'], 'penalty':['l1', 'l2'], 'C':[0.5, 0.7, 0.8, 0.9, 1, 1.5]},
    {'solver':['lbfgs'], 'penalty':['l2'], 'C':[0.5, 1, 1.5]},
]
grid_search = GridSearchCV(model, param_grid, cv=4)
grid_search.fit(feature_prepared, titanic_labels)
print("\nAccuracy of all param combination:", grid_search.cv_results_['mean_test_score'])
print("Best paramter:", grid_search.best_params_)
print("Best test score: {:.3f}".format(grid_search.best_score_))

# Display cross Validation score
best_C, best_penalty, best_solver = (grid_search.best_params_['C'],
                                     grid_search.best_params_['penalty'],
                                     grid_search.best_params_['solver'])
log_reg = LogisticRegression(C=best_C, 
            penalty=best_penalty, solver=best_solver)
cross_val_info(log_reg, feature_prepared, titanic_labels)

# output file