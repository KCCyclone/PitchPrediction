import pandas as pd
from sklearn import svm
from pybaseball import statcast
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

def categorize(pitch_type):
    if pitch_type in ['FF', 'SI', 'FC']:
        pitch_type = 'Fastball'
    elif pitch_type in ['CH', 'FS', 'FO', 'SC', 'CU', 'KC', 'CS', 'SL', 'ST', 'SV', 'KN', 'EP']:
        pitch_type = 'Offspeed'
    #elif pitch_type in ['CU', 'KC', 'CS', 'SL', 'ST', 'SV', 'KN', 'EP']:
        #pitch_type = 'Breaking'
    else:
        pitch_type = 'Other'
    return pitch_type
def predict_from_release_pos():
    #stats = statcast(start_dt="2022-06-24", end_dt="2022-06-28")
    stats = pd.read_pickle('stats.pickle')
    print(set(stats))
    stats['pitch_type_cat'] = stats['pitch_type'].map(categorize)
    #filtered_stats = stats[stats['pitch_type_cat'] != 'Fastball']
    x_divider = 0
    #y_divider = 54.2
    #stats = stats.loc[stats['balls']<=x_divider]
    test_group_size = 12
    first_dep_col = 'balls'
    second_dep_col = 'strikes'
    independent_col = 'pitch_type_cat'
    independent_vars = stats[[first_dep_col, second_dep_col]] #add filtered_ in front of stats on this line and line 42 and uncomment line 25 to exclude fastball
    independent_vars_training = independent_vars.head(len(independent_vars)-test_group_size)
    # independent_vars_test = independent_vars.tail(test_group_size)
    independent_vars_test = pd.DataFrame(columns=[first_dep_col, second_dep_col])
    for ball_count in range(4):
        for strike_count in range(3):
            independent_vars_test = pd.concat([independent_vars_test, 
                                               pd.DataFrame([[ball_count, strike_count]], columns=independent_vars_test.columns)], 
                                               ignore_index=True)
    dependent_var = stats[independent_col] #READ COMMENT ON LINE 33, filtered_
    dependent_var_training = dependent_var.head(len(dependent_var)-test_group_size)
    dependent_var_test = dependent_var.tail(test_group_size)
    print(independent_vars_training)
    print(f"{dependent_var_test=}")
    #stats.to_pickle('stats.pickle')
    clf = make_pipeline(StandardScaler(), svm.SVC(kernel = 'rbf'))
    clf = CalibratedClassifierCV(clf)
    clf.fit(independent_vars_training, dependent_var_training)
    dependent_var_predicted = clf.predict(independent_vars_test)
    dependent_var_probabilities = clf.predict_proba(independent_vars_test)
    print(list(dependent_var_test))
    print(dependent_var_predicted)
    print(dependent_var_probabilities)

    # Generate some sample data
    x = independent_vars_test[first_dep_col]
    y = independent_vars_test[second_dep_col]
    
    # Custom labels for True and False
    # custom_labeler = lambda b: "Fastball" if b else "Offspeed"
    custom_labeler = lambda s: s

    
    display_vars = dependent_var_predicted
    # display_vars = dependent_var_test

     # Map predicted values to custom labels for better visualization
    labels = [custom_labeler(v) for v in display_vars]
    
    pitch_to_num = {v:i for i, v in enumerate(sorted(set(display_vars)))}
    classes = [pitch_to_num[v] for v in display_vars]

    to_display = {}
    for xdp, ydp, p in zip(x, y, display_vars):
        if p not in to_display:
            to_display[p] = ([], [])
        to_display[p][0].append(xdp)
        to_display[p][1].append(ydp)
    
    for p, (xs, ys) in to_display.items():
        # Create the scatter plot
        plt.scatter(xs, ys, label=p)
    
    for (_i, row), prob in zip(independent_vars_test.iterrows(), dependent_var_probabilities):
        plt.text(row[first_dep_col], row[second_dep_col], f'{round(max(prob), 3):.1%}', fontsize=8, ha='left', va='bottom')
    
    # Add labels and show the plot
    plt.xlabel('Balls')
    plt.ylabel('Strikes')
    plt.legend()
    # Custom legend labels
    '''handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label=custom_labels[True]),
               plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label=custom_labels[False])]
    
    plt.legend(handles=handles)'''
    plt.show()

predict_from_release_pos()