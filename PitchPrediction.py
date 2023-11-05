import pandas as pd
from sklearn import svm
from pybaseball import statcast
import numpy as np
import matplotlib.pyplot as plt

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
# Generate some sample data
x = independent_vars_test['release_pos_x']
y = independent_vars_test['release_pos_y']
pitch_to_num = {v:i for i, v in enumerate(set(dependent_var_predicted))}
classes = [pitch_to_num[v] for v in dependent_var_predicted]
# Create the scatter plot
plt.scatter(x, y, c=classes)
# Add labels and show the plot
plt.xlabel('X')
plt.ylabel('Y')
plt.show()