import pandas as pd
from sklearn import svm
from pybaseball import statcast
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def predict_from_release_pos():
    #stats = statcast(start_dt="2022-06-24", end_dt="2022-06-28")
    stats = pd.read_pickle('stats.pickle')
    print(set(stats))
    stats['fastball'] = stats['pitch_type'].map(lambda r: r in ['FF', 'SI', 'FC'])
    x_divider = 0
    #y_divider = 54.2
    stats = stats.loc[stats['release_pos_x']<=x_divider]
    first_dep_col = 'release_pos_x'
    second_dep_col = 'strikes'
    independent_col = 'fastball'
    independent_vars = stats[[first_dep_col, second_dep_col]]
    independent_vars_training = independent_vars.head(len(independent_vars)-100)
    independent_vars_test = independent_vars.tail(100)
    dependent_var = stats[independent_col]
    dependent_var_training = dependent_var.head(len(dependent_var)-100)
    dependent_var_test = dependent_var.tail(100)
    print(independent_vars_training)
    print(f"{dependent_var_test=}")
    #stats.to_pickle('stats.pickle')
    clf = make_pipeline(StandardScaler(), svm.SVC(kernel = 'rbf'))
    clf.fit(independent_vars_training, dependent_var_training)
    dependent_var_predicted = clf.predict(independent_vars_test)
    print(list(dependent_var_test))
    print(dependent_var_predicted)
    # Generate some sample data
    x = independent_vars_test[first_dep_col]
    y = independent_vars_test[second_dep_col]
    
    display_vars = dependent_var_predicted
    # display_vars = dependent_var_test
    
    pitch_to_num = {v:i for i, v in enumerate(sorted(set(display_vars)))}
    classes = [pitch_to_num[v] for v in display_vars]

    to_display = {}
    for xdp, ydp, p in zip(x, y, display_vars):
        if p not in to_display:
            to_display[p] = ([], [])
        to_display[p][0].append(xdp)
        to_display[p][1].append(ydp)

    for p, (x, y) in to_display.items():
        # Create the scatter plot
        plt.scatter(x, y, label=p)
    # Add labels and show the plot
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()

predict_from_release_pos()