import pandas as pd
from sklearn import svm
from pybaseball import statcast
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier

def categorize(pitch_type):
    if pitch_type in ['FF', 'SI', 'FC']:
        pitch_type = 'Fastball'
    elif pitch_type in ['CH', 'FS', 'FO', 'SC']:
        pitch_type = 'Offspeed'
    elif pitch_type in ['CU', 'KC', 'CS', 'SL', 'ST', 'SV', 'KN', 'EP']:
        pitch_type = 'Breaking'
    elif pitch_type in ['PO', 'IN']:
        pitch_type = "PO/IN"
    else:
        pitch_type = 'Other'
    return pitch_type

def cat_to_num():
    cat_map = {}
    i = 0
    def cmap(cat):
        nonlocal cat_map
        nonlocal i
        if cat in cat_map:
            return cat_map[cat]
        cat_map[cat] = i
        i += 1
        return i-1
    return cmap

def create_independent_vars_test(indep_column_names, first_indep_max, second_indep_max):
    independent_vars_test = pd.DataFrame(columns=indep_column_names)
    for ball_count in range(first_indep_max):
        for strike_count in range(second_indep_max):
            independent_vars_test = pd.concat([independent_vars_test, 
                                               pd.DataFrame([[ball_count, strike_count]], columns=independent_vars_test.columns)], 
                                               ignore_index=True)
    return independent_vars_test

def predict_from_release_pos():
    #stats = statcast(start_dt="2022-06-24", end_dt="2022-06-28")
    stats = pd.read_pickle('stats.pickle')
    print(set(stats))
    stats['if_fielding_alignment_num'] = stats['if_fielding_alignment'].map(cat_to_num())
    stats = stats[stats['release_spin_rate'].notnull()]
    stats['pitch_type_cat'] = stats['pitch_type'].map(categorize)
    #stats = stats[stats['pitch_type_cat'] != 'Fastball']
    
    # For plotting pitch type on scatterplot
    # stats['pitch_type_cat_num'] = stats['pitch_type_cat'].map(cat_to_num())

    x_divider = 0
    #y_divider = 54.2
    #stats = stats.loc[stats['balls']<=x_divider]
    test_group_size = 100
    indep_column_names = ['if_fielding_alignment_num', 'release_spin_rate']
    dependent_col = 'pitch_type_cat'

    # Show predictive power of independent var
    # plt.scatter(stats[second_dep_col], stats['pitch_type_cat_num'])
    # plt.show()

    independent_vars = stats[indep_column_names] #add filtered_ in front of stats on this line and line 45 and uncomment line 33 to exclude fastball
    independent_vars_training = independent_vars.head(len(independent_vars)-test_group_size)
    #independent_vars_test = independent_vars.tail(test_group_size)
    independent_vars_test = create_independent_vars_test(indep_column_names, 8, 10)
    dependent_var = stats[dependent_col] #READ COMMENT ON LINE 41, filtered_
    dependent_var_training = dependent_var.head(len(dependent_var)-test_group_size)
    dependent_var_test = dependent_var.tail(test_group_size)
    print(independent_vars_training)
    print(f"{dependent_var_test=}")
    #stats.to_pickle('stats.pickle')
    clf = make_pipeline(StandardScaler(), MLPClassifier(random_state=1, max_iter=300))#svm.SVC(kernel = 'rbf'))
    clf = CalibratedClassifierCV(clf)
    clf.fit(independent_vars_training, dependent_var_training)
    dependent_var_predicted = clf.predict(independent_vars_test)
    dependent_var_probabilities = clf.predict_proba(independent_vars_test)
    print(list(dependent_var_test))
    print(dependent_var_predicted)
    print(dependent_var_probabilities)

    # Generate some sample data
    x = independent_vars_test[indep_column_names[0]]
    y = independent_vars_test[indep_column_names[1]]
    
    # Custom labels for True and False
    # custom_labeler = lambda b: "Fastball" if b else "Offspeed"
    custom_labeler = lambda s: s

    
    display_vars = dependent_var_predicted
    # display_vars = dependent_var_test

     # Map predicted values to custom labels for better visualization
    labels = [custom_labeler(v) for v in display_vars]
    
    pitch_to_num = {v:i for i, v in enumerate(sorted(set(display_vars)))}
    classes = [pitch_to_num[v] for v in display_vars]

    x_to_num = None
    # x_to_num = {x: i for i, x in enumerate(sorted(set(x)))}

    to_display = {}
    for xdp, ydp, p in zip(x, y, display_vars):
        if p not in to_display:
            to_display[p] = ([], [])
        if x_to_num is not None:
            xdp = x_to_num[xdp]
        to_display[p][0].append(xdp)
        to_display[p][1].append(ydp)
    
    for p, (xs, ys) in to_display.items():
        # Create the scatter plot
        plt.scatter(xs, ys, label=p)
    
    for (_i, row), prob in zip(independent_vars_test.iterrows(), dependent_var_probabilities):
        plt.text(row[indep_column_names[0]], row[indep_column_names[1]], f'{round(max(prob), 3):.1%}', fontsize=8, ha='left', va='bottom')
    
    # Add labels and show the plot
    plt.xlabel(indep_column_names[0])
    plt.ylabel(indep_column_names[1])
    plt.legend()
    # Custom legend labels
    '''handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label=custom_labels[True]),
               plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label=custom_labels[False])]
    
    plt.legend(handles=handles)'''
    plt.show()

predict_from_release_pos()