import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV, cross_validate
from sklearn import model_selection, metrics
import matplotlib.pyplot as plt
from snorkel.labeling import labeling_function, PandasLFApplier, LFAnalysis
from snorkel.labeling.model import MajorityLabelVoter, LabelModel
majority_model = MajorityLabelVoter()
label_model = LabelModel(cardinality=2, verbose=True)


def load_data(round=1):
    demographics = pd.read_csv('../cohort_selection/demographics.csv', sep='\t', 
                           parse_dates=['VISIT_START_DATETIME', 'VISIT_END_DATETIME', 'BIRTH_DATETIME'])
    demographics.columns = demographics.columns.str.lower()
    
    psi11 = pd.read_csv('../cohort_selection/ahrq_psi11.csv', sep='|',
                       parse_dates=['PROC_EARLIEST', 'PROCEDURE_DATE', 'NUMERATOR_CONDITION_START_DATE', 
                                   'NUMERATOR_PROCEDURE_DATE'])
    psi11.columns = psi11.columns.str.lower()
    
    df = demographics[['visit_occurrence_id', 'visit_start_datetime', 'visit_end_datetime', 'age_on_admission', 
                       'gender']] \
        .merge(psi11[['visit_occurrence_id', 'person_id', 'grid', 'study_group', 'respiratory_failure_any', 
                        'eligible_vent', 'numerator_condition_start_date', 'numerator_procedure_date']],
                 on=['visit_occurrence_id'],
                 how='outer')
    assert df.shape[0] == psi11.shape[0]
    
    """
    Snorkel has many more resources for unstructured/text than tabular data. 
    Out-of-the-box snorkel implementations are really good for taking a single document and 
    applying the learning function to different pieces of the document. 
    With tabular data in the biomedical space, we frequently have multiple rows (time points) 
    for the same patient or visit for which we're trying to make a prediction. 

    I am modifying the original dataframe so that all of the rows for a single visit are combined 
    into 1 row with 2 semicolons between each entry. 
    This should allow me to make predictions at the visit level while still writing rules as if 
    "looking through the entire chart."
    """
    # use a `map` to ensure all values are treated as strings for concatenation
    SEP = ';;'
    join_as_strings = lambda x: SEP.join(map(str, x))

    dfg = df.groupby(['visit_occurrence_id', 'grid', 'person_id', 'study_group', 'visit_start_datetime', 
                      'age_on_admission', 'gender']) \
        ['respiratory_failure_any', 'eligible_vent', 'numerator_condition_start_date', 'numerator_procedure_date'] \
        .agg(join_as_strings) \
        .reset_index()
    # NOTE: the grouping drops some patients who had a missing visit_start_datetime - this variable
    #   is sort of necessary, and it only influenced a few visits (10 visits, 6 patients)
    # can explore with:
    # df.isnull().sum()
    # len(df[df['visit_start_datetime'].isnull()]['person_id'].unique())
    # len(df[df['visit_start_datetime'].isnull()]['visit_occurrence_id'].unique())
        
    # in first round, we're creating the originaly splits
    if round == 1:
        return df, dfg
    
    # test set remains the same throughout
    df_test = pd.read_csv('./test_set.csv')
    
    # subsequent rounds, data are already split
    if round == 2:
        # import updated/manually-reviewed sets
        df_train = pd.read_csv('./train_set2.csv')
        df_dev = pd.read_csv('./dev_set2_labeled.csv')
        df_valid = pd.read_csv('./valid_set2_labeled.csv')
        
        return df_train, df_dev, df_valid, df_test
        
    if round == 3:
        df_train = pd.read_csv('./train_set3.csv')
        df_dev = pd.read_csv('./dev_set3_labeled.csv')
        df_valid = pd.read_csv('./valid_set3_labeled.csv')
        
        return df_train, df_dev, df_valid, df_test
        
    if round == 4:
        df_train = pd.read_csv('./train_set4.csv')
        df_dev = pd.read_csv('./dev_set4_labeled.csv')
        df_valid = pd.read_csv('./valid_set4_labeled.csv')
        
        return df_train, df_dev, df_valid, df_test
        
    if round == 5:
        df_train = pd.read_csv('./train_set5.csv')
        df_dev = pd.read_csv('./dev_set5_labeled.csv')
        # the "_corrected_labeled_of_missing_only" version replaces the original 17 "controls" that were duplicated from the dev set
        # Note: this did not change any training sets or any learning along the way because the 
        #   17 original controls were simply stored in a separate object & not saved to file
        # the updated "valid_set5_labeled_duplicates_removed" removes those 17 "controls"
        df_valid1 = pd.read_csv('./valid_set5_labeled_duplicates_removed.csv')
        df_valid2 = pd.read_csv('./valid_set_corrected_labeled_of_missing_only.csv')
        # drop those erroneous visits that were also in the dev set
        df_valid = df_valid1[~df_valid1['visit_occurrence_id'].isin(np.unique(df_dev['visit_occurrence_id']))]
        df_valid = df_valid.merge(df_valid2, how='outer', on=['visit_occurrence_id', 'grid', 'person_id', 'study_group', 
                                                              'visit_start_datetime', 'age_on_admission', 'gender', 
                                                              'respiratory_failure_any', 'eligible_vent', 'numerator_condition_start_date',
                                                              'numerator_procedure_date', 'naloxone_admin_prob',
                                                              'review_notes', 'label'])
        assert df_valid.shape[0] == 90
        
        return df_train, df_dev, df_valid, df_test
        
    return None


def add_numeric_data(dfg):
    # naloxone administration probabilities
    naloxone_admin = pd.read_csv('../sd_structured/naloxone/naloxone_exposure_with_labels_all.csv', sep='\t')
    naloxone_admin.rename(columns={'tf_model_prob': 'naloxone_admin_prob'}, inplace=True)
    dfg = dfg.merge(naloxone_admin[['visit_occurrence_id', 'naloxone_admin_prob']], 
                    on=['visit_occurrence_id'], how='left')
    assert dfg.shape[0] == 52861
    
    # counts from regular expressions
    regex_counts = pd.read_csv('../sd_text/regex_counts.csv', sep=',')
    dfg = dfg.merge(regex_counts, on=['visit_occurrence_id'], how='left')
    assert dfg.shape[0] == 52861
    
    # ICD code information
    # load visits where CCS label-based ICD code for a diagnosis that can also cause respiratory failure 
    # (e.g., sepsis, stroke) had its first occurrence on the same day as the first occurrence of a 
    # respiratory failure CCS label-based ICD code
    confounding_diagnosis_present = pd.read_csv('../sd_structured/icd/visits_with_confounding_icd_codes.csv')
    dfg = dfg.merge(confounding_diagnosis_present, on=['visit_occurrence_id'], how='left')
    assert dfg.shape[0] == 52861
    
    return dfg


def reattach_numeric_data(df_train, df_dev, df_valid=None, df_test=None):    
    regex_counts = pd.read_csv('../sd_text/regex_counts.csv', sep=',')
    confounding_diagnosis_present = pd.read_csv('../sd_structured/icd/visits_with_confounding_icd_codes.csv')
    
    df_train.drop(list(df_train.filter(regex = 'counts_')), axis = 1, inplace = True)
    df_train.drop(list(df_train.filter(regex = 'cond_')), axis = 1, inplace = True)
    df_train = df_train.merge(regex_counts, on=['visit_occurrence_id'], how='left')
    df_train = df_train.merge(confounding_diagnosis_present, on=['visit_occurrence_id'], how='left')

    df_dev.drop(list(df_dev.filter(regex = 'counts_')), axis = 1, inplace = True)
    df_dev.drop(list(df_dev.filter(regex = 'cond_')), axis = 1, inplace = True)
    df_dev = df_dev.merge(regex_counts, on=['visit_occurrence_id'], how='left')
    df_dev = df_dev.merge(confounding_diagnosis_present, on=['visit_occurrence_id'], how='left')
    
    if df_valid is not None and df_test is None:
        df_valid.drop(list(df_valid.filter(regex = 'counts_')), axis = 1, inplace = True)
        df_valid.drop(list(df_valid.filter(regex = 'cond_')), axis = 1, inplace = True)
        df_valid = df_valid.merge(regex_counts, on=['visit_occurrence_id'], how='left')
        df_valid = df_valid.merge(confounding_diagnosis_present, on=['visit_occurrence_id'], how='left')
        
        return df_train, df_dev, df_valid
    
    if df_test is not None:
        # repeated from above
        df_valid.drop(list(df_valid.filter(regex = 'counts_')), axis = 1, inplace = True)
        df_valid.drop(list(df_valid.filter(regex = 'cond_')), axis = 1, inplace = True)
        df_valid = df_valid.merge(regex_counts, on=['visit_occurrence_id'], how='left')
        df_valid = df_valid.merge(confounding_diagnosis_present, on=['visit_occurrence_id'], how='left')
        
        # new
        df_test.drop(list(df_test.filter(regex = 'counts_')), axis = 1, inplace = True)
        df_test.drop(list(df_test.filter(regex = 'cond_')), axis = 1, inplace = True)
        df_test = df_test.merge(regex_counts, on=['visit_occurrence_id'], how='left')
        df_test = df_test.merge(confounding_diagnosis_present, on=['visit_occurrence_id'], how='left')
        
        return df_train, df_dev, df_valid, df_test
    
    return df_train, df_dev


def make_splits(dfg):
    demographics = pd.read_csv('../cohort_selection/demographics.csv', sep='\t', 
                           parse_dates=['VISIT_START_DATETIME', 'VISIT_END_DATETIME', 'BIRTH_DATETIME'])
    demographics.columns = demographics.columns.str.lower()
    # hold out test set will be those that were sent for crowdsourcing
    crowdsource_cases = pd.read_csv('../cohort_selection/cases_for_crowdsourcing.csv')
    crowdsource_controls = pd.read_csv('../cohort_selection/controls_for_crowdsourcing.csv')

    df_test = dfg[(dfg['visit_occurrence_id'].isin(crowdsource_cases['visit_occurrence_id'])) \
                 | (dfg['visit_occurrence_id'].isin(crowdsource_controls['visit_occurrence_id']))]
    assert df_test.shape[0] == 764
    
    df_train = dfg[~dfg['visit_occurrence_id'].isin(df_test['visit_occurrence_id'])]
    assert df_train.shape[0] + df_test.shape[0] == demographics.shape[0]
    
    # create dev & validation sets, oversampling for cases of interest
    df_train_cases = df_train[df_train['study_group']=='case']
    df_train_controls = df_train[df_train['study_group']=='control']

    visit_size = 50

    # for a total of 50 test visits, pull 2/3rd from the "cases"
    df_train_cases, df_dev_cases = train_test_split(df_train_cases, 
                                                    test_size=int(np.round(visit_size*2/3)), random_state=123)
    df_train_cases, df_valid_cases = train_test_split(df_train_cases, 
                                                      test_size=int(np.round(visit_size*2/3)), random_state=123)

    # for a total of 50 test visits, pull 1/3rd from the "controls"
    df_train_controls, df_dev_controls = train_test_split(df_train_controls, 
                                                          test_size=int(np.round(visit_size/3)), random_state=123)
    df_train_controls, df_valid_controls = train_test_split(df_train_controls, 
                                                            test_size=int(np.round(visit_size/3)), random_state=123)

    # combine cases & controls into the same data set
    df_train = pd.concat([df_train_cases, df_train_controls], ignore_index=True)
    df_dev = pd.concat([df_dev_cases, df_dev_controls], ignore_index=True)    
    # 6/21/21: error noted on above line - I accidentally included "df_dev_controls" instead of "df_valid_controls"
    #df_valid = pd.concat([df_valid_cases, df_dev_controls], ignore_index=True) # original
    df_valid = pd.concat([df_valid_cases, df_valid_controls], ignore_index=True)
    
    assert df_dev.shape[1] == df_valid.shape[1] == df_train.shape[1]
    assert df_train.shape[0] + df_dev.shape[0] + df_valid.shape[0] + df_test.shape[0] == demographics.shape[0]

    # We had previously discussed whether to conduct the analysis at the patient vs. visit level. 
    # To be conservative with our outcome labeling, we are dropping any *patients* who are in the test. 
    # How many test set patients in the train/dev/test sets?
    #print(sum(df_train['person_id'].isin(df_test['person_id'])))
    #print(sum(df_dev['person_id'].isin(df_test['person_id'])))
    #print(sum(df_valid['person_id'].isin(df_test['person_id'])))

    # drop any visits from the training set that are associated with a person in the test set
    # (because some patients had multiple visits)
    df_excluded_visits = df_train[df_train['person_id'].isin(df_test['person_id'])]
    df_train = df_train[~df_train['person_id'].isin(df_test['person_id'])]
    assert sum(df_train['person_id'].isin(df_test['person_id'])) == 0
    assert sum(df_test['person_id'].isin(df_train['person_id'])) == 0
    
    # how many unique patients?
    #len(df_excluded_visits['person_id'].unique())
    
    # There were 285 visits removed from training set due to being assoociated with a person
    #   who was also in the test set. Those visits comprised 203 unique patients. 
    
    return df_train, df_dev, df_valid, df_test


def label_model_tuning(lfs, df_cols, L_train, L_dev, Y_dev, n_epochs, lr, lr_scheduler, class_balance, seed):
    df_tune = pd.DataFrame(columns=df_cols)
    # loop through all possibilities, storing results along the way
    for i in n_epochs:
        for j in lr:
            for k in lr_scheduler: 
                for s in seed:
                    label_model.fit(L_train=L_train, 
                                Y_dev = Y_dev, 
                                n_epochs = i, 
                                lr = j,
                                #l2 = 0.01,
                                optimizer = 'adamax', # 'sgd', 'adam', 'adamax'
                                lr_scheduler = k, # 'constant', 'linear', 'exponential', 'step'
                                #prec_init = 0.7,
                                class_balance = class_balance, 
                                log_freq = 100,
                                seed = s)
                    analysis = LFAnalysis(L=L_train, lfs=lfs).lf_summary(est_weights=label_model.get_weights())
                    df_row = [i, j, k, s]
                    df_row.extend(analysis['Learned Weight'])
                    df_row = pd.DataFrame([df_row], columns=df_cols)
            
                    # store accuracy on development set
                    df_row['accuracy'] = label_model.score(L=L_dev, Y=Y_dev)["accuracy"]

                    df_tune = pd.concat([df_tune, df_row], ignore_index=True, sort=False)
    
    # melt for seaborn
    df_tune_long = pd.melt(df_tune, id_vars=['n_epochs', 'lr', 'lr_scheduler', 'seed'], 
                           var_name='learning_function', value_name='learned_weight')
                           
    return df_tune, df_tune_long

    
def plot_probabilities_histogram(Y):
    plt.hist(Y, bins=25)
    plt.xlabel("Probability of Visit having OIRD")
    plt.ylabel("Number of data points")
    plt.show()
    

def find_flags(labels_array, indices, learning_functions):
    """
    Loop through the array of applied learning functions to identify which rules applied a non-abstain vote. 
    """
    for idx in indices:
        print('Row ' + str(idx))
        for lf in range(len(learning_functions)):
            if labels_array[idx][lf] > -1: 
                print(labels_array[idx][lf], learning_functions[lf].name)
        print('~'*30)


def nested_cv(X_train, y_train, model, search_params, verbosity=1, tune_metric='f1',
              k_outer=10, k_inner=3, seed=987, sample_weight=None):
    """
    Performs nested cross-validation on training data to estimate generalized error. 
    Adapted from https://machinelearningmastery.com/nested-cross-validation-for-machine-learning-with-python/
    """
    
    # configure the cross-validation procedure
    cv_outer = KFold(n_splits=k_outer, shuffle=True, random_state=seed)

    outer_results = dict({'acc': [], 'f1': [], 'mse': [], 'auc': [],
                          'best_params_inner_cv': []})
    
    # enumerate splits
    for train_ix, test_ix in cv_outer.split(X_train):
        # split data
        X_train_cv, X_test_cv = X_train[train_ix, :], X_train[test_ix, :]
        y_train_cv, y_test_cv = y_train[train_ix], y_train[test_ix]            
        # configure the cross-validation procedure
        cv_inner = KFold(n_splits=k_inner, shuffle=True, random_state=seed)
        # define search space
        space = search_params
        # define search
        search = GridSearchCV(model, space, scoring=tune_metric, cv=cv_inner, n_jobs=1, refit=True, verbose=verbosity)    
        # execute search
        if sample_weight is not None:
            X_train_weights = sample_weight[train_ix]
            result = search.fit(X_train_cv, y_train_cv, sample_weight=X_train_weights)
        else:
            result = search.fit(X_train_cv, y_train_cv)
        # get the best performing model fit on the whole training set
        best_model = result.best_estimator_
        # evaluate model on the hold out dataset
        yhat = best_model.predict(X_test_cv)
        # if predictions are continuous (e.g., regression), make binary
        if len(np.unique(yhat)) > 2:
            # if actuals are on log scale, exponentiate predictions & actuals first
            # note: not checking predictions because they could fall outside this range in regression
            if np.all((np.exp(y_test_cv)>=0) & (np.exp(y_test_cv)<=1)):
                y_test_cv = np.exp(y_test_cv)
                yhat = np.exp(yhat)
            yhat = np.where(yhat >= 0.5, 1, 0)
            y_test_cv = np.where(y_test_cv >= 0.5, 1, 0)
        # evaluate the model & store results
        # first make sure there's at least 1 value in each class
        if len(np.unique(yhat)) == 2 & len(np.unique(y_test_cv)) == 2:
            outer_results['acc'].append(metrics.accuracy_score(y_test_cv, yhat))
            outer_results['f1'].append(metrics.f1_score(y_test_cv, yhat))
            outer_results['auc'].append(metrics.roc_auc_score(y_test_cv, yhat))
        else:
            outer_results['acc'].append(np.NaN)
            outer_results['f1'].append(np.NaN)
            outer_results['auc'].append(np.NaN)
        outer_results['mse'].append(metrics.mean_squared_error(y_test_cv, yhat))
        outer_results['best_params_inner_cv'].append(result.best_params_)
        # report progress
        if verbosity:
            print('> F1=%.3f, est=%.3f, best=%s' % (metrics.f1_score(y_test_cv, yhat), result.best_score_, result.best_params_))
    
    return outer_results


def summarize_model_performance(model_results):
    """
    Summarizes estimated performance of CV model
    """
    print('F1 : %.3f (%.3f)' % (np.mean(model_results['f1']), np.std(model_results['f1'])))
    print('AUC: %.3f (%.3f)' % (np.mean(model_results['auc']), np.std(model_results['auc'])))
    print('MSE: %.3f (%.3f)' % (np.mean(model_results['mse']), np.std(model_results['mse'])))
    print('Acc: %.3f (%.3f)' % (np.mean(model_results['acc']), np.std(model_results['acc'])))
    
    return None