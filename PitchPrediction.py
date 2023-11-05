import pandas as pd
from sklearn import svm
from pybaseball import statcast

#stats = statcast(start_dt="2022-06-24", end_dt="2022-06-28")
stats = pd.read_pickle('stats.pickle')
independent_vars = stats[['release_pos_x', 'release_pos_y']]
independent_vars_training = independent_vars.head(len(independent_vars)-100)
independent_vars_test = independent_vars.tail(100)
dependent_var = stats['pitch_type']
dependent_var_training = dependent_var.head(len(dependent_var)-100)
dependent_var_test = dependent_var.tail(100)
print(independent_vars_training)
print(dependent_var_test)
#stats.to_pickle('stats.pickle')
clf = svm.LinearSVC()
clf.fit(independent_vars_training, dependent_var_training)
dependent_var_predicted = clf.predict(independent_vars_test)
print(list(dependent_var_test))
print(dependent_var_predicted)