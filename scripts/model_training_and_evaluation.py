from sklearn.ensemble import RandomForestClassifier
import numpy as np
import optuna
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import logger
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from numpy.polynomial import Polynomial
from sklearn.decomposition import PCA
import pandas as pd

def fit_polynomial(pulse, degree):
    # x = np.arange(len(pulse))
    y = pulse
    # p = Polynomial.fit(x, y, degree)
    # coeffs = p.convert().coef
    return pulse

def pca_feature_engineering(X, y):
    
    logger.log("PCA###########")
    logger.log("shape of X before PCA: ", X.shape)
    logger.log("shape of y before PCA: ", y.shape)

    pca = PCA()
    pca.fit(X)
    X_pca = pca.transform(X)

    # calculating the correlation between the principal components and the target variable
    correlations = []
    for i in range(X_pca.shape[1]):
        corr = np.corrcoef(X_pca[:, i], y)[0, 1]
        correlations.append(corr)

    correlation_df = pd.DataFrame({
        'Principal Component': [f'PC{i+1}' for i in range(X_pca.shape[1])],
        'Correlation with Class': correlations
    })


    # Filter PCs by ansolute correlation 
    threshold = 0.05

    selected_pcs_indices = correlation_df[correlation_df['Correlation with Class'].abs() > threshold].index

    X_pca = X_pca[:, selected_pcs_indices]



    return X_pca, pca, selected_pcs_indices

def data_augmentation(X, y, all_scans, groups):
    # This method contains various data augmentation techniques
    # They are changed by using a return statement
    
    np.random.seed(42)

    return X, y

    X_augmented = []
    y_augmented = []

    # simple augmentation
    # pairwise averaging
    for i in range(len(X)):
        id = groups[i]
        label = y[i]
        forward_scans = all_scans[id]
        # i know there are 10 forward scans
        # we do pairwise averaging so we will have 3 augmented scans
        for j in range(0, 6, 2):
            scan1 = forward_scans[j]
            scan2 = forward_scans[j+1]
            avg_scan = np.mean([scan1, scan2], axis=0)
            X_augmented.append(avg_scan)
            y_augmented.append(label)
    
    return np.array(X_augmented), np.array(y_augmented)

    # getting all forward scans
    for i in range(len(X)):
        id = groups[i]
        label = y[i]
        forward_scans = all_scans[id]
        # getting 3 forward scans
        for j in range(6):
            scan = forward_scans[j]
            X_augmented.append(scan)
            y_augmented.append(label)
        
    return np.array(X_augmented), np.array(y_augmented)

    # time shift
    for i in range(len(X)):
        pulse = X[i]
        label = y[i]

        X_augmented.append(pulse)
        y_augmented.append(label)

        # time shift
        for shift in range(-5, 6):
            new_pulse = np.roll(pulse, shift)
            X_augmented.append(new_pulse)
            y_augmented.append(label)


    return np.array(X_augmented), np.array(y_augmented)

def train_and_evaluate_model_KNN(data, parameters):
    np.random.seed(42)

    X, y, groups, all_scans = data

    params = parameters

    k_outer = params['k_outer']
    k_inner = params['k_inner']

    outer_cv = StratifiedGroupKFold(n_splits=k_outer)

    training_accuracies = []
    test_accuracies = []
    balanced_training_accuracies = []
    balanced_test_accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    final_confusion_matrix = np.zeros((2, 2))
    true_labels = []
    predicted_labels = []

    # This is needed to convert back from 0s and 1s to the original labels
    le = LabelEncoder()
    le.classes_ = np.array(["g/PBS", "PBS"])

    # printing unique labels and their counts
    s = "Class counts in entire dataset: "
    unique_labels, counts = np.unique(y, return_counts=True)
    unique_labels = le.inverse_transform(unique_labels)
    for label, count in zip(unique_labels, counts):
        s += f"{label}: {count}, "
    logger.log(s)

    logger.log(f"Number of total datapoints: {len(y)}")

    outer_fold_number = 1
    for train_outer_indicies, test_outer_indicies in tqdm(outer_cv.split(X, y, groups), desc='Outer CV Split', total=k_outer):
        X_train_outer, X_test_outer = X[train_outer_indicies], X[test_outer_indicies]
        y_train_outer, y_test_outer = y[train_outer_indicies], y[test_outer_indicies]
        groups_train_outer = groups[train_outer_indicies]

        def inner_fold_training(train_inner_indices, test_inner_indices, X_train_outer, y_train_outer, model_for_tuning, degree):
            X_train_inner, X_test_inner = X_train_outer[train_inner_indices], X_train_outer[test_inner_indices]
            y_train_inner, y_test_inner = y_train_outer[train_inner_indices], y_train_outer[test_inner_indices]

            X_train_inner_poly = [fit_polynomial(pulse, degree) for pulse in X_train_inner]
            X_test_inner_poly = [fit_polynomial(pulse, degree) for pulse in X_test_inner]

            X_train_inner_poly = np.array(X_train_inner_poly)
            X_test_inner_poly = np.array(X_test_inner_poly)

            model_for_tuning.fit(X_train_inner_poly, y_train_inner)

            validation_accuracy = model_for_tuning.score(X_test_inner_poly, y_test_inner)
            return validation_accuracy

        def objective(trial):
            degree_of_polynomial = trial.suggest_int('degree_of_polynomial', params['degree_of_polynomial']['min'], params['degree_of_polynomial']['max'])
            n_neighbors = trial.suggest_int('n_neighbors', params['n_neighbors']['min'], params['n_neighbors']['max'])
            weights = trial.suggest_categorical('weights', params['weights'])
            algorithm = trial.suggest_categorical('algorithm', params['algorithm'])
            leaf_size = trial.suggest_int('leaf_size', params['leaf_size']['min'], params['leaf_size']['max'])
            p = trial.suggest_int('p', params['p']['min'], params['p']['max'])

            model_for_tuning = KNeighborsClassifier(
                n_neighbors=n_neighbors,
                weights=weights,
                algorithm=algorithm,
                leaf_size=leaf_size,
                p=p
            )

            inner_cv = StratifiedGroupKFold(n_splits=k_inner)

            validation_accuracies = Parallel(n_jobs=-1)(delayed(inner_fold_training)(train_inner_indices, test_inner_indices, X_train_outer, y_train_outer, model_for_tuning, degree_of_polynomial)
                for train_inner_indices, test_inner_indices in inner_cv.split(X_train_outer, y_train_outer, groups_train_outer)
            )

            return np.mean(validation_accuracies)
    

        optuna.logging.set_verbosity(optuna.logging.ERROR)
        study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner(), sampler=optuna.samplers.TPESampler(seed=42)) 
        study.optimize(objective, n_trials=params['optuna_n_trials'])

        best_params = study.best_params

        degree_of_polynomial = best_params['degree_of_polynomial']
        n_neighbors = best_params['n_neighbors']
        weights = best_params['weights']
        algorithm = best_params['algorithm']
        leaf_size = best_params['leaf_size']
        p = best_params['p']

        # Training the model on the outer fold using best parameters
        model_outer = KNeighborsClassifier(
            n_neighbors=n_neighbors, 
            weights=weights, 
            algorithm=algorithm, 
            leaf_size=leaf_size, 
            p=p
        )

        X_train_outer_poly = [fit_polynomial(pulse, degree_of_polynomial) for pulse in X_train_outer]
        X_test_outer_poly = [fit_polynomial(pulse, degree_of_polynomial) for pulse in X_test_outer]

        X_train_outer_poly = np.array(X_train_outer_poly)
        X_test_outer_poly = np.array(X_test_outer_poly)

        model_outer.fit(X_train_outer_poly, y_train_outer)

        y_train_outer_pred = model_outer.predict(X_train_outer_poly)
        y_test_outer_pred = model_outer.predict(X_test_outer_poly)
        y_pred_test_outer_proba = model_outer.predict_proba(X_test_outer_poly)[:, 1]

        training_accuracy = accuracy_score(y_train_outer, y_train_outer_pred)
        test_accuracy = accuracy_score(y_test_outer, y_test_outer_pred)
        balanced_train_accuracy = balanced_accuracy_score(y_train_outer, y_train_outer_pred)
        balanced_test_accuracy = balanced_accuracy_score(y_test_outer, y_test_outer_pred)

        precision = precision_score(y_test_outer, y_test_outer_pred)
        recall = recall_score(y_test_outer, y_test_outer_pred)
        cm = confusion_matrix(y_test_outer, y_test_outer_pred)
        f1 = f1_score(y_test_outer, y_test_outer_pred)

        training_accuracies.append(training_accuracy)
        test_accuracies.append(test_accuracy)
        balanced_training_accuracies.append(balanced_train_accuracy)
        balanced_test_accuracies.append(balanced_test_accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        final_confusion_matrix += cm
        true_labels.extend(y_test_outer)
        predicted_labels.extend(y_pred_test_outer_proba)


        ###################  Logging  ###################
        logger.log(f"Outer fold number: {outer_fold_number}")

        # Logging the count of true classes from training
        s = "Outer loop true training class counts: "
        unique_labels, counts = np.unique(y_train_outer, return_counts=True)
        unique_labels = le.inverse_transform(unique_labels)
        for label, count in zip(unique_labels, counts):
            s += f"{label}: {count}, "
        logger.log(s)

        # Logging the count of the prediction classes from training
        s = "Outer loop training predictions class counts: "
        unique_labels, counts = np.unique(y_train_outer_pred, return_counts=True)
        unique_labels = le.inverse_transform(unique_labels)
        for label, count in zip(unique_labels, counts):
            s += f"{label}: {count}, "
        logger.log(s)

        # Logging the count of the true classes
        s = "Outer loop true test class counts: "
        unique_labels, counts = np.unique(y_test_outer, return_counts=True)
        unique_labels = le.inverse_transform(unique_labels)
        for label, count in zip(unique_labels, counts):
            s += f"{label}: {count}, "
        logger.log(s)
        
        # Logging the count of the prediction classes from testing
        s = "Outer loop test predicted class counts: "
        unique_labels, counts = np.unique(y_test_outer_pred, return_counts=True)
        unique_labels = le.inverse_transform(unique_labels)
        for label, count in zip(unique_labels, counts):
            s += f"{label}: {count}, "
        logger.log(s)

        # Logging the parameters
        logger.log("Best parameters:")
        logger.log(best_params)

        # Logging the training and test accuracies
        logger.log(f"Training accuracy: {training_accuracy}")
        logger.log(f"Test accuracy: {test_accuracy}")
        
        logger.log("")

        outer_fold_number += 1
        
    # Doing one final parameter search on the entire training set to get best possible model for production
    # However, we cannot use this final model to estimate the accuracy as it has seen the entire training set

    def objective(trial):
            degree_of_polynomial = trial.suggest_int('degree_of_polynomial', params['degree_of_polynomial']['min'], params['degree_of_polynomial']['max'])
            n_neighbors = trial.suggest_int('n_neighbors', params['n_neighbors']['min'], params['n_neighbors']['max'])
            weights = trial.suggest_categorical('weights', params['weights'])
            algorithm = trial.suggest_categorical('algorithm', params['algorithm'])
            leaf_size = trial.suggest_int('leaf_size', params['leaf_size']['min'], params['leaf_size']['max'])
            p = trial.suggest_int('p', params['p']['min'], params['p']['max'])

            model_for_tuning = KNeighborsClassifier(
                n_neighbors=n_neighbors,
                weights=weights,
                algorithm=algorithm,
                leaf_size=leaf_size,
                p=p
            )
            
            X_poly = [fit_polynomial(pulse, degree_of_polynomial) for pulse in X]
            X_poly = np.array(X_poly)

            model_for_tuning.fit(X_poly, y)
            score = model_for_tuning.score(X_poly, y)

            return score
    
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    final_study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner(), sampler=optuna.samplers.TPESampler(seed=42)) 
    final_study.optimize(objective, n_trials=params['optuna_n_trials'], n_jobs=-1)

    final_hyperparameters = final_study.best_params

    degree_of_polynomial = final_hyperparameters['degree_of_polynomial']
    n_neighbors = final_hyperparameters['n_neighbors']
    weights = final_hyperparameters['weights']
    algorithm = final_hyperparameters['algorithm']
    leaf_size = final_hyperparameters['leaf_size']
    p = final_hyperparameters['p']

    model_final = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        algorithm=algorithm,
        leaf_size=leaf_size,
        p=p
    )

    X_poly = [fit_polynomial(pulse, degree_of_polynomial) for pulse in X]
    X_poly = np.array(X_poly)

    model_final.fit(X_poly, y)

    # Averaging the final metrics
    final_training_accuracy = np.mean(training_accuracies)
    final_test_accuracy = np.mean(test_accuracies)
    final_balanced_train_accuracy = np.mean(balanced_training_accuracies)
    final_balanced_test_accuracy = np.mean(balanced_test_accuracies)
    final_precision = np.mean(precisions)
    final_recall = np.mean(recalls)
    TP, FP, FN, TN = final_confusion_matrix.ravel()
    final_specificity = TN / (TN + FP)
    final_sensitivity = TP / (TP + FN)
    final_f1 = np.mean(f1_scores)
    roc_auc = roc_auc_score(true_labels, predicted_labels)

    # standard deviation of the metrics
    std_training_accuracy = np.std(training_accuracies)
    std_test_accuracy = np.std(test_accuracies)
    std_balanced_train_accuracy = np.std(balanced_training_accuracies)
    std_balanced_test_accuracy = np.std(balanced_test_accuracies)
    std_precision = np.std(precisions)
    std_recall = np.std(recalls)
    std_f1 = np.std(f1_scores)


    # plotting confusion matrix
    cm_display = ConfusionMatrixDisplay(final_confusion_matrix, display_labels=le.classes_)
    cm_display.plot(values_format='.2f')
    plt.savefig("temp_plots/confusion_matrix.png")
    plt.close()

    # plotting roc curve
    fpr, tpr, _ = roc_curve(true_labels, predicted_labels)
    plt.plot(fpr, tpr, label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Random guessing (baseline)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend()
    plt.grid(True)
    plt.savefig("temp_plots/roc_curve.png")
    plt.close()
    
    # logging final metrics
    logger.log(f"Final best hyperparameters: {final_hyperparameters}")
    logger.log(f"Final estimated training accuracy by averaging: {round(final_training_accuracy, 3)}")
    logger.log(f"Final estimated test accuracy by averaging: {round(final_test_accuracy, 3)}")
    logger.log(f"Final estimated balanced training accuracy by averaging: {round(final_balanced_train_accuracy, 3)}")
    logger.log(f"Final estimated balanced test accuracy by averaging: {round(final_balanced_test_accuracy, 3)}")
    logger.log(f"Final estimated precision by averaging: {round(final_precision, 3)}")
    logger.log(f"Final estimated recall by averaging: {round(final_recall, 3)}")
    logger.log(f"Final estimated f1 score by averaging: {round(final_f1, 3)}")
    logger.log(f"Final estimated specificity: {round(final_specificity, 3)}")
    logger.log(f"Final estimated sensitivity: {round(final_sensitivity, 3)}")
    logger.log(f"Final estimated roc auc score: {round(roc_auc, 3)}")

    # logging standard deviation of the metrics
    logger.log(f"Standard deviation of training accuracy: {round(std_training_accuracy, 3)}")
    logger.log(f"Standard deviation of test accuracy: {round(std_test_accuracy, 3)}")
    logger.log(f"Standard deviation of balanced training accuracy: {round(std_balanced_train_accuracy, 3)}")
    logger.log(f"Standard deviation of balanced test accuracy: {round(std_balanced_test_accuracy, 3)}")
    logger.log(f"Standard deviation of precision: {round(std_precision, 3)}")
    logger.log(f"Standard deviation of recall: {round(std_recall, 3)}")
    logger.log(f"Standard deviation of f1 score: {round(std_f1, 3)}")


    metrics = {}
    metrics['train_accuracy'] = round(final_training_accuracy, 3)
    metrics['std_train_accuracy'] = round(std_training_accuracy, 3)
    metrics['test_accuracy'] = round(final_test_accuracy, 3)
    metrics['std_test_accuracy'] = round(std_test_accuracy, 3)
    metrics['balanced_train_accuracy'] = round(final_balanced_train_accuracy, 3)
    metrics['std_balanced_train_accuracy'] = round(std_balanced_train_accuracy, 3)
    metrics['balanced_test_accuracy'] = round(final_balanced_test_accuracy, 3)
    metrics['std_balanced_test_accuracy'] = round(std_balanced_test_accuracy, 3)
    metrics['precision'] = round(final_precision, 3)
    metrics['std_precision'] = round(std_precision, 3)
    metrics['recall'] = round(final_recall, 3)
    metrics['std_recall'] = round(std_recall, 3)
    metrics['specificity'] = round(final_specificity, 3)
    metrics['sensitivity'] = round(final_sensitivity, 3)
    metrics['f1'] = round(final_f1, 3)
    metrics['std_f1'] = round(std_f1, 3)
    metrics['roc_auc'] = round(roc_auc, 3)


    return model_final, metrics

def train_and_evaluate_model_SVM(data, parameters):
    np.random.seed(42)

    X, y, groups, all_scans = data

    params = parameters

    k_outer = params['k_outer']
    k_inner = params['k_inner']

    outer_cv = StratifiedGroupKFold(n_splits=k_outer)

    """ fractions = [.25, .5, .75, 1]
    models_dict = {
        "fractions": [],
        "models": [],
        "training_accuracies": [],
        "test_accuracies": [],
        "balanced_training_accuracies": [],
        "balanced_test_accuracies": [],
        "precisions": [],
        "recalls": [],
        "f1_scores": [],
    } """

    training_accuracies = []
    test_accuracies = []
    balanced_training_accuracies = []
    balanced_test_accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    final_confusion_matrix = np.zeros((2, 2))
    true_labels = []
    predicted_labels = []

    # This is needed to convert back from 0s and 1s to the original labels
    le = LabelEncoder()
    le.classes_ = np.array(["g/PBS", "PBS"])

    # printing unique labels and their counts
    s = "Class counts in entire dataset: "
    unique_labels, counts = np.unique(y, return_counts=True)
    unique_labels = le.inverse_transform(unique_labels)
    for label, count in zip(unique_labels, counts):
        s += f"{label}: {count}, "
    logger.log(s)

    logger.log(f"Number of total datapoints: {len(y)}")

    outer_fold_number = 1
    for train_outer_indicies, test_outer_indicies in tqdm(outer_cv.split(X, y, groups), desc='Outer CV Split', total=k_outer):
        X_train_outer, X_test_outer = X[train_outer_indicies], X[test_outer_indicies]
        y_train_outer, y_test_outer = y[train_outer_indicies], y[test_outer_indicies]
        groups_train_outer = groups[train_outer_indicies]

        def inner_fold_training(train_inner_indices, test_inner_indices, X_train_outer, y_train_outer, model_for_tuning, degree):
            X_train_inner, X_test_inner = X_train_outer[train_inner_indices], X_train_outer[test_inner_indices]
            y_train_inner, y_test_inner = y_train_outer[train_inner_indices], y_train_outer[test_inner_indices]
            groups_train_inner = groups_train_outer[train_inner_indices]

            X_train_inner, y_train_inner = data_augmentation(X_train_inner, y_train_inner, all_scans, groups_train_inner)

            X_train_inner_poly = [fit_polynomial(pulse, degree) for pulse in X_train_inner]
            X_test_inner_poly = [fit_polynomial(pulse, degree) for pulse in X_test_inner]

            X_train_inner_poly = np.array(X_train_inner_poly)
            X_test_inner_poly = np.array(X_test_inner_poly)

            X_train_inner_pca, pca, selected_pcs = pca_feature_engineering(X_train_inner_poly, y_train_inner)
            X_test_inner_pca = pca.transform(X_test_inner_poly)
            X_test_inner_pca = X_test_inner_pca[:, selected_pcs]

            model_for_tuning.fit(X_train_inner_pca, y_train_inner)

            validation_accuracy = model_for_tuning.score(X_test_inner_pca, y_test_inner)
            return validation_accuracy

        def objective(trial):
            #degree_of_polynomial = trial.suggest_int('degree_of_polynomial', params['degree_of_polynomial']['min'], params['degree_of_polynomial']['max'])
            required_variance_explained = trial.suggest_float('pca_required_variance_explained', params['pca_required_variance_explained']['min'], params['pca_required_variance_explained']['max'])
            C = trial.suggest_float('C', params['C']['min'], params['C']['max'], log=True)
            kernel = trial.suggest_categorical('kernel', params['kernel'])
            gamma = trial.suggest_categorical('gamma', params['gamma'])
            degree = trial.suggest_int('degree', params['degree']['min'], params['degree']['max'])
            coef0 = trial.suggest_float('coef0', params['coef0']['min'], params['coef0']['max'])
            tol = trial.suggest_float('tol', params['tol']['min'], params['tol']['max'], log=True)

            model_for_tuning = SVC(
                C=C,
                kernel=kernel,
                gamma=gamma,
                degree=degree,
                coef0=coef0,
                tol=tol,
                random_state=42
            )

            inner_cv = StratifiedGroupKFold(n_splits=k_inner)

            validation_accuracies = Parallel(n_jobs=-1)(delayed(inner_fold_training)(train_inner_indices, test_inner_indices, X_train_outer, y_train_outer, model_for_tuning, required_variance_explained)
                for train_inner_indices, test_inner_indices in inner_cv.split(X_train_outer, y_train_outer, groups_train_outer)
            )

            return np.mean(validation_accuracies)
    

        optuna.logging.set_verbosity(optuna.logging.ERROR)
        study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner(), sampler=optuna.samplers.TPESampler(seed=42)) 
        study.optimize(objective, n_trials=params['optuna_n_trials'])

        best_params = study.best_params

        #degree_of_polynomial = best_params['degree_of_polynomial']
        required_variance_explained = best_params['pca_required_variance_explained']
        C = best_params['C']
        kernel = best_params['kernel']
        gamma = best_params['gamma']
        degree = best_params['degree']
        coef0 = best_params['coef0']
        tol = best_params['tol']

        # Training the model on the outer fold using best parameters
        model_outer = SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            degree=degree,
            coef0=coef0,
            tol=tol,
            probability=True,
            random_state=42
        )

        X_train_outer, y_train_outer = data_augmentation(X_train_outer, y_train_outer, all_scans, groups_train_outer)

        """ X_train_outer_poly = [fit_polynomial(pulse, degree_of_polynomial) for pulse in X_train_outer]
        X_test_outer_poly = [fit_polynomial(pulse, degree_of_polynomial) for pulse in X_test_outer]

        X_train_outer_poly = np.array(X_train_outer_poly)
        X_test_outer_poly = np.array(X_test_outer_poly)

        model_outer.fit(X_train_outer_poly, y_train_outer)

        

        y_train_outer_pred = model_outer.predict(X_train_outer_poly)
        y_test_outer_pred = model_outer.predict(X_test_outer_poly)
        y_pred_test_outer_proba = model_outer.predict_proba(X_test_outer_poly)[:, 1] """


        # pca feature engineering
        X_train_outer_pca, pca, selected_pcs = pca_feature_engineering(X_train_outer, y_train_outer)
        X_test_outer_pca = pca.transform(X_test_outer)
        X_test_outer_pca = X_test_outer_pca[:, selected_pcs]

        X_train_outer_pca = np.array(X_train_outer_pca)
        X_test_outer_pca = np.array(X_test_outer_pca)

        model_outer.fit(X_train_outer_pca, y_train_outer)

        y_train_outer_pred = model_outer.predict(X_train_outer_pca)
        y_test_outer_pred = model_outer.predict(X_test_outer_pca)
        y_pred_test_outer_proba = model_outer.predict_proba(X_test_outer_pca)[:, 1]

        training_accuracy = accuracy_score(y_train_outer, y_train_outer_pred)
        test_accuracy = accuracy_score(y_test_outer, y_test_outer_pred)
        balanced_train_accuracy = balanced_accuracy_score(y_train_outer, y_train_outer_pred)
        balanced_test_accuracy = balanced_accuracy_score(y_test_outer, y_test_outer_pred)

        precision = precision_score(y_test_outer, y_test_outer_pred)
        recall = recall_score(y_test_outer, y_test_outer_pred)
        cm = confusion_matrix(y_test_outer, y_test_outer_pred)
        f1 = f1_score(y_test_outer, y_test_outer_pred)

        training_accuracies.append(training_accuracy)
        test_accuracies.append(test_accuracy)
        balanced_training_accuracies.append(balanced_train_accuracy)
        balanced_test_accuracies.append(balanced_test_accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        final_confusion_matrix += cm
        true_labels.extend(y_test_outer)
        predicted_labels.extend(y_pred_test_outer_proba)


        """ for fraction in fractions:
            # creating subset based on fraction
            X_train_outer_subset = X_train_outer_poly[:int(fraction*len(X_train_outer_poly))]
            y_train_outer_subset = y_train_outer[:int(fraction*len(y_train_outer))]

            model_outer = SVC(
                C=C,
                kernel=kernel,
                gamma=gamma,
                degree=degree,
                coef0=coef0,
                tol=tol,
                probability=True,
                random_state=42
            )

            model_outer.fit(X_train_outer_subset, y_train_outer_subset)

            y_train_outer_pred = model_outer.predict(X_train_outer_subset)
            y_test_outer_pred = model_outer.predict(X_test_outer_poly)
            y_pred_test_outer_proba = model_outer.predict_proba(X_test_outer_poly)[:, 1]

            training_accuracy = accuracy_score(y_train_outer_subset, y_train_outer_pred)
            test_accuracy = accuracy_score(y_test_outer, y_test_outer_pred)
            balanced_train_accuracy = balanced_accuracy_score(y_train_outer_subset, y_train_outer_pred)
            balanced_test_accuracy = balanced_accuracy_score(y_test_outer, y_test_outer_pred)

            precision = precision_score(y_test_outer, y_test_outer_pred)
            recall = recall_score(y_test_outer, y_test_outer_pred)
            f1 = f1_score(y_test_outer, y_test_outer_pred)


            models_dict["fractions"].append(fraction)
            models_dict["models"].append(model_outer)
            models_dict["training_accuracies"].append(training_accuracy)
            models_dict["test_accuracies"].append(test_accuracy)
            models_dict["balanced_training_accuracies"].append(balanced_train_accuracy)
            models_dict["balanced_test_accuracies"].append(balanced_test_accuracy)
            models_dict["precisions"].append(precision)
            models_dict["recalls"].append(recall)
            models_dict["f1_scores"].append(f1) """


        ###################  Logging  ###################
        logger.log(f"Outer fold number: {outer_fold_number}")

        # Logging the count of true classes from training
        s = "Outer loop true training class counts: "
        unique_labels, counts = np.unique(y_train_outer, return_counts=True)
        unique_labels = le.inverse_transform(unique_labels)
        for label, count in zip(unique_labels, counts):
            s += f"{label}: {count}, "
        logger.log(s)

        # Logging the count of the prediction classes from training
        s = "Outer loop training predictions class counts: "
        unique_labels, counts = np.unique(y_train_outer_pred, return_counts=True)
        unique_labels = le.inverse_transform(unique_labels)
        for label, count in zip(unique_labels, counts):
            s += f"{label}: {count}, "
        logger.log(s)

        # Logging the count of the true classes
        s = "Outer loop true test class counts: "
        unique_labels, counts = np.unique(y_test_outer, return_counts=True)
        unique_labels = le.inverse_transform(unique_labels)
        for label, count in zip(unique_labels, counts):
            s += f"{label}: {count}, "
        logger.log(s)
        
        # Logging the count of the prediction classes from testing
        s = "Outer loop test predicted class counts: "
        unique_labels, counts = np.unique(y_test_outer_pred, return_counts=True)
        unique_labels = le.inverse_transform(unique_labels)
        for label, count in zip(unique_labels, counts):
            s += f"{label}: {count}, "
        logger.log(s)

        # Logging the parameters
        logger.log("Best parameters:")
        logger.log(best_params)

        # Logging the training and test accuracies
        logger.log(f"Training accuracy: {training_accuracy}")
        logger.log(f"Test accuracy: {test_accuracy}")
        
        logger.log("")

        outer_fold_number += 1
        
    # Doing one final parameter search on the entire training set to get best possible model for production
    # However, we cannot use this final model to estimate the accuracy as it has seen the entire training set

    X, y = data_augmentation(X, y, all_scans, groups)

    def objective(trial):
            #degree_of_polynomial = trial.suggest_int('degree_of_polynomial', params['degree_of_polynomial']['min'], params['degree_of_polynomial']['max'])
            required_variance_explained = trial.suggest_float('pca_required_variance_explained', params['pca_required_variance_explained']['min'], params['pca_required_variance_explained']['max'])
            C = trial.suggest_float('C', params['C']['min'], params['C']['max'], log=True)
            kernel = trial.suggest_categorical('kernel', params['kernel'])
            gamma = trial.suggest_categorical('gamma', params['gamma'])
            degree = trial.suggest_int('degree', params['degree']['min'], params['degree']['max'])
            coef0 = trial.suggest_float('coef0', params['coef0']['min'], params['coef0']['max'])
            tol = trial.suggest_float('tol', params['tol']['min'], params['tol']['max'], log=True)

            model_for_tuning = SVC(
                C=C,
                kernel=kernel,
                gamma=gamma,
                degree=degree,
                coef0=coef0,
                tol=tol,
                random_state=42
            )
            
            X_pca, pca, selected_pcs = pca_feature_engineering(X, y)
            X_pca = np.array(X_pca)

            #X_poly = [fit_polynomial(pulse, degree_of_polynomial) for pulse in X]
            #X_poly = np.array(X_poly)

            #model_for_tuning.fit(X_poly, y)
            model_for_tuning.fit(X_pca, y)
            score = model_for_tuning.score(X_pca, y)

            return score
    
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    final_study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner(), sampler=optuna.samplers.TPESampler(seed=42)) 
    final_study.optimize(objective, n_trials=params['optuna_n_trials'], n_jobs=-1)

    final_hyperparameters = final_study.best_params

    #degree_of_polynomial = final_hyperparameters['degree_of_polynomial']
    required_variance_explained = final_hyperparameters['pca_required_variance_explained']
    C = final_hyperparameters['C']
    kernel = final_hyperparameters['kernel']
    gamma = final_hyperparameters['gamma']
    degree = final_hyperparameters['degree']
    coef0 = final_hyperparameters['coef0']
    tol = final_hyperparameters['tol']


    model_final = SVC(
        C=C,
        kernel=kernel,
        gamma=gamma,
        degree=degree,
        coef0=coef0,
        tol=tol,
        random_state=42
    )

    X_pca, pca, selected_pcs = pca_feature_engineering(X, y)
    X_pca = np.array(X_pca)

    model_final.fit(X_pca, y)

    """ X_poly = [fit_polynomial(pulse, degree_of_polynomial) for pulse in X]
    X_poly = np.array(X_poly)

    model_final.fit(X_poly, y) """

    # Averaging the final metrics
    final_training_accuracy = np.mean(training_accuracies)
    final_test_accuracy = np.mean(test_accuracies)
    final_balanced_train_accuracy = np.mean(balanced_training_accuracies)
    final_balanced_test_accuracy = np.mean(balanced_test_accuracies)
    final_precision = np.mean(precisions)
    final_recall = np.mean(recalls)
    TP, FP, FN, TN = final_confusion_matrix.ravel()
    final_specificity = TN / (TN + FP)
    final_sensitivity = TP / (TP + FN)
    final_f1 = np.mean(f1_scores)
    roc_auc = roc_auc_score(true_labels, predicted_labels)

    # standard deviation of the metrics
    std_training_accuracy = np.std(training_accuracies)
    std_test_accuracy = np.std(test_accuracies)
    std_balanced_train_accuracy = np.std(balanced_training_accuracies)
    std_balanced_test_accuracy = np.std(balanced_test_accuracies)
    std_precision = np.std(precisions)
    std_recall = np.std(recalls)
    std_f1 = np.std(f1_scores)


    # plotting confusion matrix
    cm_display = ConfusionMatrixDisplay(final_confusion_matrix, display_labels=le.classes_)
    cm_display.plot(values_format='.2f')
    plt.savefig("temp_plots/confusion_matrix.png")
    plt.close()

    # plotting roc curve
    fpr, tpr, _ = roc_curve(true_labels, predicted_labels)
    plt.plot(fpr, tpr, label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Random guessing (baseline)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend()
    plt.grid(True)
    plt.savefig("temp_plots/roc_curve.png")
    plt.close()
    
    # logging final metrics
    logger.log(f"Final best hyperparameters: {final_hyperparameters}")
    logger.log(f"Final estimated training accuracy by averaging: {round(final_training_accuracy, 3)}")
    logger.log(f"Final estimated test accuracy by averaging: {round(final_test_accuracy, 3)}")
    logger.log(f"Final estimated balanced training accuracy by averaging: {round(final_balanced_train_accuracy, 3)}")
    logger.log(f"Final estimated balanced test accuracy by averaging: {round(final_balanced_test_accuracy, 3)}")
    logger.log(f"Final estimated precision by averaging: {round(final_precision, 3)}")
    logger.log(f"Final estimated recall by averaging: {round(final_recall, 3)}")
    logger.log(f"Final estimated f1 score by averaging: {round(final_f1, 3)}")
    logger.log(f"Final estimated specificity: {round(final_specificity, 3)}")
    logger.log(f"Final estimated sensitivity: {round(final_sensitivity, 3)}")
    logger.log(f"Final estimated roc auc score: {round(roc_auc, 3)}")

    # logging standard deviation of the metrics
    logger.log(f"Standard deviation of training accuracy: {round(std_training_accuracy, 3)}")
    logger.log(f"Standard deviation of test accuracy: {round(std_test_accuracy, 3)}")
    logger.log(f"Standard deviation of balanced training accuracy: {round(std_balanced_train_accuracy, 3)}")
    logger.log(f"Standard deviation of balanced test accuracy: {round(std_balanced_test_accuracy, 3)}")
    logger.log(f"Standard deviation of precision: {round(std_precision, 3)}")
    logger.log(f"Standard deviation of recall: {round(std_recall, 3)}")
    logger.log(f"Standard deviation of f1 score: {round(std_f1, 3)}")


    metrics = {}
    metrics['train_accuracy'] = round(final_training_accuracy, 3)
    metrics['std_train_accuracy'] = round(std_training_accuracy, 3)
    metrics['test_accuracy'] = round(final_test_accuracy, 3)
    metrics['std_test_accuracy'] = round(std_test_accuracy, 3)
    metrics['balanced_train_accuracy'] = round(final_balanced_train_accuracy, 3)
    metrics['std_balanced_train_accuracy'] = round(std_balanced_train_accuracy, 3)
    metrics['balanced_test_accuracy'] = round(final_balanced_test_accuracy, 3)
    metrics['std_balanced_test_accuracy'] = round(std_balanced_test_accuracy, 3)
    metrics['precision'] = round(final_precision, 3)
    metrics['std_precision'] = round(std_precision, 3)
    metrics['recall'] = round(final_recall, 3)
    metrics['std_recall'] = round(std_recall, 3)
    metrics['specificity'] = round(final_specificity, 3)
    metrics['sensitivity'] = round(final_sensitivity, 3)
    metrics['f1'] = round(final_f1, 3)
    metrics['std_f1'] = round(std_f1, 3)
    metrics['roc_auc'] = round(roc_auc, 3)


    # handling the models_dict
    """ for fraction in fractions:
        # getting the indicies of the fractions in the list
        indices = [i for i, x in enumerate(models_dict["fractions"]) if x == fraction]
        
        # calculating the average of the metrics
        training_accuracy = np.mean([models_dict["training_accuracies"][i] for i in indices])
        test_accuracy = np.mean([models_dict["test_accuracies"][i] for i in indices])
        balanced_train_accuracy = np.mean([models_dict["balanced_training_accuracies"][i] for i in indices])
        balanced_test_accuracy = np.mean([models_dict["balanced_test_accuracies"][i] for i in indices])
        precision = np.mean([models_dict["precisions"][i] for i in indices])
        recall = np.mean([models_dict["recalls"][i] for i in indices])
        f1 = np.mean([models_dict["f1_scores"][i] for i in indices])

        # logging the metrics
        logger.log(f"########################## logging for subsets of training data ##########################")
        logger.log(f"Fraction: {fraction}")
        logger.log(f"Training accuracy: {training_accuracy}")
        logger.log(f"Test accuracy: {test_accuracy}")
        logger.log(f"Balanced training accuracy: {balanced_train_accuracy}")
        logger.log(f"Balanced test accuracy: {balanced_test_accuracy}")
        logger.log(f"Precision: {precision}")
        logger.log(f"Recall: {recall}")
        logger.log(f"F1: {f1}")
        logger.log("") """


    return model_final, metrics

def train_and_evaluate_model_LR(data, parameters):
    np.random.seed(42)

    X, y, groups, all_scans = data

    params = parameters

    k_outer = params['k_outer']
    k_inner = params['k_inner']

    outer_cv = StratifiedGroupKFold(n_splits=k_outer)

    """ fractions = [.25, .5, .75, 1]
    models_dict = {
        "fractions": [],
        "models": [],
        "training_accuracies": [],
        "test_accuracies": [],
        "balanced_training_accuracies": [],
        "balanced_test_accuracies": [],
        "precisions": [],
        "recalls": [],
        "f1_scores": [],
    } """

    training_accuracies = []
    test_accuracies = []
    balanced_training_accuracies = []
    balanced_test_accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    final_confusion_matrix = np.zeros((2, 2))
    true_labels = []
    predicted_labels = []

    # This is needed to convert back from 0s and 1s to the original labels
    le = LabelEncoder()
    le.classes_ = np.array(["g/PBS", "PBS"])

    # printing unique labels and their counts
    s = "Class counts in entire dataset: "
    unique_labels, counts = np.unique(y, return_counts=True)
    unique_labels = le.inverse_transform(unique_labels)
    for label, count in zip(unique_labels, counts):
        s += f"{label}: {count}, "
    logger.log(s)

    logger.log(f"Number of total datapoints: {len(y)}")

    outer_fold_number = 1
    for train_outer_indicies, test_outer_indicies in tqdm(outer_cv.split(X, y, groups), desc='Outer CV Split', total=k_outer):
        X_train_outer, X_test_outer = X[train_outer_indicies], X[test_outer_indicies]
        y_train_outer, y_test_outer = y[train_outer_indicies], y[test_outer_indicies]
        groups_train_outer = groups[train_outer_indicies]

        def inner_fold_training(train_inner_indices, test_inner_indices, X_train_outer, y_train_outer, model_for_tuning, degree, groups_outer):
            X_train_inner, X_test_inner = X_train_outer[train_inner_indices], X_train_outer[test_inner_indices]
            y_train_inner, y_test_inner = y_train_outer[train_inner_indices], y_train_outer[test_inner_indices]
            groups_train_inner = groups_outer[train_inner_indices]

            X_train_inner, y_train_inner = data_augmentation(X_train_inner, y_train_inner, all_scans, groups_train_inner)

            X_train_inner_poly = [fit_polynomial(pulse, degree) for pulse in X_train_inner]
            X_test_inner_poly = [fit_polynomial(pulse, degree) for pulse in X_test_inner]

            X_train_inner_poly = np.array(X_train_inner_poly)
            X_test_inner_poly = np.array(X_test_inner_poly)

            model_for_tuning.fit(X_train_inner_poly, y_train_inner)

            validation_accuracy = model_for_tuning.score(X_test_inner_poly, y_test_inner)
            return validation_accuracy

        def objective(trial):
            degree_of_polynomial = trial.suggest_int('degree_of_polynomial', params['degree_of_polynomial']['min'], params['degree_of_polynomial']['max'])
            C = trial.suggest_float('C', params['C']['min'], params['C']['max'], log=True)
            max_iter = trial.suggest_int('max_iter', params['max_iter']['min'], params['max_iter']['max'])
            penalty = trial.suggest_categorical('penalty', params['penalty'])
            tol = trial.suggest_float('tol', params['tol']['min'], params['tol']['max'], log=True)

            model_for_tuning = LogisticRegression(
                C=C,
                max_iter=max_iter,
                penalty=penalty,
                solver='liblinear', # this is the only one that makes sense with our dataset according to the documentation
                tol=tol,
                random_state=42
            )

            inner_cv = StratifiedGroupKFold(n_splits=k_inner)

            validation_accuracies = Parallel(n_jobs=-1)(delayed(inner_fold_training)(train_inner_indices, test_inner_indices, X_train_outer, y_train_outer, model_for_tuning, degree_of_polynomial, groups_train_outer)
                for train_inner_indices, test_inner_indices in inner_cv.split(X_train_outer, y_train_outer, groups_train_outer)
            )

            return np.mean(validation_accuracies)
    

        optuna.logging.set_verbosity(optuna.logging.ERROR)
        study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner(), sampler=optuna.samplers.TPESampler(seed=42)) 
        study.optimize(objective, n_trials=params['optuna_n_trials'])

        best_params = study.best_params

        degree_of_polynomial = best_params['degree_of_polynomial']
        C = best_params['C']
        max_iter = best_params['max_iter']
        penalty = best_params['penalty']
        tol = best_params['tol']

        # Training the model on the outer fold using best parameters
        model_outer = LogisticRegression(
            C=C,
            max_iter=max_iter,
            penalty=penalty,
            solver='liblinear', # this is the only one that makes sense with our dataset according to the documentation
            tol=tol,
            random_state=42
        )

        X_train_outer, y_train_outer = data_augmentation(X_train_outer, y_train_outer, all_scans, groups_train_outer)

        X_train_outer_poly = [fit_polynomial(pulse, degree_of_polynomial) for pulse in X_train_outer]
        X_test_outer_poly = [fit_polynomial(pulse, degree_of_polynomial) for pulse in X_test_outer]

        X_train_outer_poly = np.array(X_train_outer_poly)
        X_test_outer_poly = np.array(X_test_outer_poly)

        model_outer.fit(X_train_outer_poly, y_train_outer)

        y_train_outer_pred = model_outer.predict(X_train_outer_poly)
        y_test_outer_pred = model_outer.predict(X_test_outer_poly)
        y_pred_test_outer_proba = model_outer.predict_proba(X_test_outer_poly)[:, 1]

        training_accuracy = accuracy_score(y_train_outer, y_train_outer_pred)
        test_accuracy = accuracy_score(y_test_outer, y_test_outer_pred)
        balanced_train_accuracy = balanced_accuracy_score(y_train_outer, y_train_outer_pred)
        balanced_test_accuracy = balanced_accuracy_score(y_test_outer, y_test_outer_pred)

        precision = precision_score(y_test_outer, y_test_outer_pred)
        recall = recall_score(y_test_outer, y_test_outer_pred)
        cm = confusion_matrix(y_test_outer, y_test_outer_pred)
        f1 = f1_score(y_test_outer, y_test_outer_pred)

        training_accuracies.append(training_accuracy)
        test_accuracies.append(test_accuracy)
        balanced_training_accuracies.append(balanced_train_accuracy)
        balanced_test_accuracies.append(balanced_test_accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        final_confusion_matrix += cm
        true_labels.extend(y_test_outer)
        predicted_labels.extend(y_pred_test_outer_proba)

        """ for fraction in fractions:
            # creating subset based on fraction
            X_train_outer_subset = X_train_outer_poly[:int(fraction*len(X_train_outer_poly))]
            y_train_outer_subset = y_train_outer[:int(fraction*len(y_train_outer))]

            model_outer = LogisticRegression(
            C=C,
            max_iter=max_iter,
            penalty=penalty,
            solver='liblinear', # this is the only one that makes sense with our dataset according to the documentation
            tol=tol,
            random_state=42
        )

            model_outer.fit(X_train_outer_subset, y_train_outer_subset)

            y_train_outer_pred = model_outer.predict(X_train_outer_subset)
            y_test_outer_pred = model_outer.predict(X_test_outer_poly)
            y_pred_test_outer_proba = model_outer.predict_proba(X_test_outer_poly)[:, 1]

            training_accuracy = accuracy_score(y_train_outer_subset, y_train_outer_pred)
            test_accuracy = accuracy_score(y_test_outer, y_test_outer_pred)
            balanced_train_accuracy = balanced_accuracy_score(y_train_outer_subset, y_train_outer_pred)
            balanced_test_accuracy = balanced_accuracy_score(y_test_outer, y_test_outer_pred)

            precision = precision_score(y_test_outer, y_test_outer_pred)
            recall = recall_score(y_test_outer, y_test_outer_pred)
            f1 = f1_score(y_test_outer, y_test_outer_pred)


            models_dict["fractions"].append(fraction)
            models_dict["models"].append(model_outer)
            models_dict["training_accuracies"].append(training_accuracy)
            models_dict["test_accuracies"].append(test_accuracy)
            models_dict["balanced_training_accuracies"].append(balanced_train_accuracy)
            models_dict["balanced_test_accuracies"].append(balanced_test_accuracy)
            models_dict["precisions"].append(precision)
            models_dict["recalls"].append(recall)
            models_dict["f1_scores"].append(f1) """


        ###################  Logging  ###################
        logger.log(f"Outer fold number: {outer_fold_number}")

        # Logging the count of true classes from training
        s = "Outer loop true training class counts: "
        unique_labels, counts = np.unique(y_train_outer, return_counts=True)
        unique_labels = le.inverse_transform(unique_labels)
        for label, count in zip(unique_labels, counts):
            s += f"{label}: {count}, "
        logger.log(s)

        # Logging the count of the prediction classes from training
        s = "Outer loop training predictions class counts: "
        unique_labels, counts = np.unique(y_train_outer_pred, return_counts=True)
        unique_labels = le.inverse_transform(unique_labels)
        for label, count in zip(unique_labels, counts):
            s += f"{label}: {count}, "
        logger.log(s)

        # Logging the count of the true classes
        s = "Outer loop true test class counts: "
        unique_labels, counts = np.unique(y_test_outer, return_counts=True)
        unique_labels = le.inverse_transform(unique_labels)
        for label, count in zip(unique_labels, counts):
            s += f"{label}: {count}, "
        logger.log(s)
        
        # Logging the count of the prediction classes from testing
        s = "Outer loop test predicted class counts: "
        unique_labels, counts = np.unique(y_test_outer_pred, return_counts=True)
        unique_labels = le.inverse_transform(unique_labels)
        for label, count in zip(unique_labels, counts):
            s += f"{label}: {count}, "
        logger.log(s)

        # Logging the parameters
        logger.log("Best parameters:")
        logger.log(best_params)

        # Logging the training and test accuracies
        logger.log(f"Training accuracy: {training_accuracy}")
        logger.log(f"Test accuracy: {test_accuracy}")
        
        logger.log("")

        outer_fold_number += 1
        
    # Doing one final parameter search on the entire training set to get best possible model for production
    # However, we cannot use this final model to estimate the accuracy as it has seen the entire training set

    X, y = data_augmentation(X, y, all_scans, groups)

    def objective(trial):
            degree_of_polynomial = trial.suggest_int('degree_of_polynomial', params['degree_of_polynomial']['min'], params['degree_of_polynomial']['max'])
            C = trial.suggest_float('C', params['C']['min'], params['C']['max'], log=True)
            max_iter = trial.suggest_int('max_iter', params['max_iter']['min'], params['max_iter']['max'])
            penalty = trial.suggest_categorical('penalty', params['penalty'])
            tol = trial.suggest_float('tol', params['tol']['min'], params['tol']['max'], log=True)

            model_for_tuning = LogisticRegression(
                C=C,
                max_iter=max_iter,
                penalty=penalty,
                solver='liblinear', # this is the only one that makes sense with our dataset according to the documentation
                tol=tol,
                random_state=42
            )

            X_poly = [fit_polynomial(pulse, degree_of_polynomial) for pulse in X]
            X_poly = np.array(X_poly)

            model_for_tuning.fit(X_poly, y)
            score = model_for_tuning.score(X_poly, y)

            return score
    
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    final_study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner(), sampler=optuna.samplers.TPESampler(seed=42)) 
    final_study.optimize(objective, n_trials=params['optuna_n_trials'], n_jobs=-1)

    final_hyperparameters = final_study.best_params

    degree_of_polynomial = final_hyperparameters['degree_of_polynomial']
    C = final_hyperparameters['C']
    max_iter = final_hyperparameters['max_iter']
    penalty = final_hyperparameters['penalty']
    tol = final_hyperparameters['tol']


    model_final = LogisticRegression(
        C=C,
        max_iter=max_iter,
        penalty=penalty,
        solver='liblinear', # this is the only one that makes sense with our dataset according to the documentation
        tol=tol,
        random_state=42
    )

    X_poly = [fit_polynomial(pulse, degree_of_polynomial) for pulse in X]
    X_poly = np.array(X_poly)

    model_final.fit(X_poly, y)

    # Averaging the final metrics
    final_training_accuracy = np.mean(training_accuracies)
    final_test_accuracy = np.mean(test_accuracies)
    final_balanced_train_accuracy = np.mean(balanced_training_accuracies)
    final_balanced_test_accuracy = np.mean(balanced_test_accuracies)
    final_precision = np.mean(precisions)
    final_recall = np.mean(recalls)
    TP, FP, FN, TN = final_confusion_matrix.ravel()
    final_specificity = TN / (TN + FP)
    final_sensitivity = TP / (TP + FN)
    final_f1 = np.mean(f1_scores)
    roc_auc = roc_auc_score(true_labels, predicted_labels)

    # standard deviation of the metrics
    std_training_accuracy = np.std(training_accuracies)
    std_test_accuracy = np.std(test_accuracies)
    std_balanced_train_accuracy = np.std(balanced_training_accuracies)
    std_balanced_test_accuracy = np.std(balanced_test_accuracies)
    std_precision = np.std(precisions)
    std_recall = np.std(recalls)
    std_f1 = np.std(f1_scores)


    # plotting confusion matrix
    cm_display = ConfusionMatrixDisplay(final_confusion_matrix, display_labels=le.classes_)
    cm_display.plot(values_format='.2f')
    plt.savefig("temp_plots/confusion_matrix.png")
    plt.close()

    # plotting roc curve
    fpr, tpr, _ = roc_curve(true_labels, predicted_labels)
    plt.plot(fpr, tpr, label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Random guessing (baseline)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend()
    plt.grid(True)
    plt.savefig("temp_plots/roc_curve.png")
    plt.close()
    
    # logging final metrics
    logger.log(f"Final best hyperparameters: {final_hyperparameters}")
    logger.log(f"Final estimated training accuracy by averaging: {round(final_training_accuracy, 3)}")
    logger.log(f"Final estimated test accuracy by averaging: {round(final_test_accuracy, 3)}")
    logger.log(f"Final estimated balanced training accuracy by averaging: {round(final_balanced_train_accuracy, 3)}")
    logger.log(f"Final estimated balanced test accuracy by averaging: {round(final_balanced_test_accuracy, 3)}")
    logger.log(f"Final estimated precision by averaging: {round(final_precision, 3)}")
    logger.log(f"Final estimated recall by averaging: {round(final_recall, 3)}")
    logger.log(f"Final estimated f1 score by averaging: {round(final_f1, 3)}")
    logger.log(f"Final estimated specificity: {round(final_specificity, 3)}")
    logger.log(f"Final estimated sensitivity: {round(final_sensitivity, 3)}")
    logger.log(f"Final estimated roc auc score: {round(roc_auc, 3)}")

    # logging standard deviation of the metrics
    logger.log(f"Standard deviation of training accuracy: {round(std_training_accuracy, 3)}")
    logger.log(f"Standard deviation of test accuracy: {round(std_test_accuracy, 3)}")
    logger.log(f"Standard deviation of balanced training accuracy: {round(std_balanced_train_accuracy, 3)}")
    logger.log(f"Standard deviation of balanced test accuracy: {round(std_balanced_test_accuracy, 3)}")
    logger.log(f"Standard deviation of precision: {round(std_precision, 3)}")
    logger.log(f"Standard deviation of recall: {round(std_recall, 3)}")
    logger.log(f"Standard deviation of f1 score: {round(std_f1, 3)}")


    metrics = {}
    metrics['train_accuracy'] = round(final_training_accuracy, 3)
    metrics['std_train_accuracy'] = round(std_training_accuracy, 3)
    metrics['test_accuracy'] = round(final_test_accuracy, 3)
    metrics['std_test_accuracy'] = round(std_test_accuracy, 3)
    metrics['balanced_train_accuracy'] = round(final_balanced_train_accuracy, 3)
    metrics['std_balanced_train_accuracy'] = round(std_balanced_train_accuracy, 3)
    metrics['balanced_test_accuracy'] = round(final_balanced_test_accuracy, 3)
    metrics['std_balanced_test_accuracy'] = round(std_balanced_test_accuracy, 3)
    metrics['precision'] = round(final_precision, 3)
    metrics['std_precision'] = round(std_precision, 3)
    metrics['recall'] = round(final_recall, 3)
    metrics['std_recall'] = round(std_recall, 3)
    metrics['specificity'] = round(final_specificity, 3)
    metrics['sensitivity'] = round(final_sensitivity, 3)
    metrics['f1'] = round(final_f1, 3)
    metrics['std_f1'] = round(std_f1, 3)
    metrics['roc_auc'] = round(roc_auc, 3)

    # handling the models_dict
    """ for fraction in fractions:
        # getting the indicies of the fractions in the list
        indices = [i for i, x in enumerate(models_dict["fractions"]) if x == fraction]
        
        # calculating the average of the metrics
        training_accuracy = np.mean([models_dict["training_accuracies"][i] for i in indices])
        test_accuracy = np.mean([models_dict["test_accuracies"][i] for i in indices])
        balanced_train_accuracy = np.mean([models_dict["balanced_training_accuracies"][i] for i in indices])
        balanced_test_accuracy = np.mean([models_dict["balanced_test_accuracies"][i] for i in indices])
        precision = np.mean([models_dict["precisions"][i] for i in indices])
        recall = np.mean([models_dict["recalls"][i] for i in indices])
        f1 = np.mean([models_dict["f1_scores"][i] for i in indices])

        # logging the metrics
        logger.log(f"########################## logging for subsets of training data ##########################")
        logger.log(f"Fraction: {fraction}")
        logger.log(f"Training accuracy: {training_accuracy}")
        logger.log(f"Test accuracy: {test_accuracy}")
        logger.log(f"Balanced training accuracy: {balanced_train_accuracy}")
        logger.log(f"Balanced test accuracy: {balanced_test_accuracy}")
        logger.log(f"Precision: {precision}")
        logger.log(f"Recall: {recall}")
        logger.log(f"F1: {f1}")
        logger.log("") """


    return model_final, metrics

def train_and_evaluate_model_RF(data, parameters):
    np.random.seed(42)

    X, y, groups, all_scans = data

    params = parameters

    k_outer = params['k_outer']
    k_inner = params['k_inner']

    outer_cv = StratifiedGroupKFold(n_splits=k_outer)

    training_accuracies = []
    test_accuracies = []
    balanced_training_accuracies = []
    balanced_test_accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    final_confusion_matrix = np.zeros((2, 2))
    true_labels = []
    predicted_labels = []

    # This is needed to convert back from 0s and 1s to the original labels
    le = LabelEncoder()
    le.classes_ = np.array(["g/PBS", "PBS"])

    # printing unique labels and their counts
    s = "Class counts in entire dataset: "
    unique_labels, counts = np.unique(y, return_counts=True)
    unique_labels = le.inverse_transform(unique_labels)
    for label, count in zip(unique_labels, counts):
        s += f"{label}: {count}, "
    logger.log(s)

    logger.log(f"Number of total datapoints: {len(y)}")

    outer_fold_number = 1
    for train_outer_indicies, test_outer_indicies in tqdm(outer_cv.split(X, y, groups), desc='Outer CV Split', total=k_outer):
        X_train_outer, X_test_outer = X[train_outer_indicies], X[test_outer_indicies]
        y_train_outer, y_test_outer = y[train_outer_indicies], y[test_outer_indicies]
        groups_train_outer = groups[train_outer_indicies]

        def inner_fold_training(train_inner_indices, test_inner_indices, X_train_outer, y_train_outer, model_for_tuning, degree):
            X_train_inner, X_test_inner = X_train_outer[train_inner_indices], X_train_outer[test_inner_indices]
            y_train_inner, y_test_inner = y_train_outer[train_inner_indices], y_train_outer[test_inner_indices]

            X_train_inner_poly = [fit_polynomial(pulse, degree) for pulse in X_train_inner]
            X_test_inner_poly = [fit_polynomial(pulse, degree) for pulse in X_test_inner]

            X_train_inner_poly = np.array(X_train_inner_poly)
            X_test_inner_poly = np.array(X_test_inner_poly)

            model_for_tuning.fit(X_train_inner_poly, y_train_inner)

            validation_accuracy = model_for_tuning.score(X_test_inner_poly, y_test_inner)
            return validation_accuracy

        def objective(trial):
            degree_of_polynomial = trial.suggest_int('degree_of_polynomial', params['degree_of_polynomial']['min'], params['degree_of_polynomial']['max'])
            n_estimators = trial.suggest_int('n_estimators', params['n_estimators']['min'], params['n_estimators']['max'])
            max_depth = trial.suggest_int('max_depth', params['max_depth']['min'], params['max_depth']['max'])
            min_samples_split = trial.suggest_int('min_samples_split', params['min_samples_split']['min'], params['min_samples_split']['max'])
            min_samples_leaf = trial.suggest_int('min_samples_leaf', params['min_samples_leaf']['min'], params['min_samples_leaf']['max'])
            bootstrap = trial.suggest_categorical('bootstrap', params['bootstrap'])
            criterion = trial.suggest_categorical('criterion', params['criterion'])

            model_for_tuning = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                bootstrap=bootstrap,
                criterion=criterion,
                random_state=42,
            ) 


            inner_cv = StratifiedGroupKFold(n_splits=k_inner)

            validation_accuracies = Parallel(n_jobs=-1)(delayed(inner_fold_training)(train_inner_indices, test_inner_indices, X_train_outer, y_train_outer, model_for_tuning, degree_of_polynomial)
                for train_inner_indices, test_inner_indices in inner_cv.split(X_train_outer, y_train_outer, groups_train_outer)
            )

            return np.mean(validation_accuracies)
    

        optuna.logging.set_verbosity(optuna.logging.ERROR)
        study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner(), sampler=optuna.samplers.TPESampler(seed=42)) 
        study.optimize(objective, n_trials=params['optuna_n_trials'])

        best_params = study.best_params

        degree_of_polynomial = best_params['degree_of_polynomial']
        n_estimators = best_params['n_estimators']
        max_depth = best_params['max_depth']
        min_samples_split = best_params['min_samples_split']
        min_samples_leaf = best_params['min_samples_leaf']
        bootstrap = best_params['bootstrap']
        criterion = best_params['criterion']

        # Training the model on the outer fold using best parameters
        model_outer = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            bootstrap=bootstrap,
            criterion=criterion,
            random_state=42,
        )

        X_train_outer_poly = [fit_polynomial(pulse, degree_of_polynomial) for pulse in X_train_outer]
        X_test_outer_poly = [fit_polynomial(pulse, degree_of_polynomial) for pulse in X_test_outer]

        X_train_outer_poly = np.array(X_train_outer_poly)
        X_test_outer_poly = np.array(X_test_outer_poly)

        model_outer.fit(X_train_outer_poly, y_train_outer)

        y_train_outer_pred = model_outer.predict(X_train_outer_poly)
        y_test_outer_pred = model_outer.predict(X_test_outer_poly)
        y_pred_test_outer_proba = model_outer.predict_proba(X_test_outer_poly)[:, 1]

        training_accuracy = accuracy_score(y_train_outer, y_train_outer_pred)
        test_accuracy = accuracy_score(y_test_outer, y_test_outer_pred)
        balanced_train_accuracy = balanced_accuracy_score(y_train_outer, y_train_outer_pred)
        balanced_test_accuracy = balanced_accuracy_score(y_test_outer, y_test_outer_pred)

        precision = precision_score(y_test_outer, y_test_outer_pred)
        recall = recall_score(y_test_outer, y_test_outer_pred)
        cm = confusion_matrix(y_test_outer, y_test_outer_pred)
        f1 = f1_score(y_test_outer, y_test_outer_pred)

        training_accuracies.append(training_accuracy)
        test_accuracies.append(test_accuracy)
        balanced_training_accuracies.append(balanced_train_accuracy)
        balanced_test_accuracies.append(balanced_test_accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        final_confusion_matrix += cm
        true_labels.extend(y_test_outer)
        predicted_labels.extend(y_pred_test_outer_proba)


        ###################  Logging  ###################
        logger.log(f"Outer fold number: {outer_fold_number}")

        # Logging the count of true classes from training
        s = "Outer loop true training class counts: "
        unique_labels, counts = np.unique(y_train_outer, return_counts=True)
        unique_labels = le.inverse_transform(unique_labels)
        for label, count in zip(unique_labels, counts):
            s += f"{label}: {count}, "
        logger.log(s)

        # Logging the count of the prediction classes from training
        s = "Outer loop training predictions class counts: "
        unique_labels, counts = np.unique(y_train_outer_pred, return_counts=True)
        unique_labels = le.inverse_transform(unique_labels)
        for label, count in zip(unique_labels, counts):
            s += f"{label}: {count}, "
        logger.log(s)

        # Logging the count of the true classes
        s = "Outer loop true test class counts: "
        unique_labels, counts = np.unique(y_test_outer, return_counts=True)
        unique_labels = le.inverse_transform(unique_labels)
        for label, count in zip(unique_labels, counts):
            s += f"{label}: {count}, "
        logger.log(s)
        
        # Logging the count of the prediction classes from testing
        s = "Outer loop test predicted class counts: "
        unique_labels, counts = np.unique(y_test_outer_pred, return_counts=True)
        unique_labels = le.inverse_transform(unique_labels)
        for label, count in zip(unique_labels, counts):
            s += f"{label}: {count}, "
        logger.log(s)

        # Logging the parameters
        logger.log("Best parameters:")
        logger.log(best_params)

        # Logging the training and test accuracies
        logger.log(f"Training accuracy: {training_accuracy}")
        logger.log(f"Test accuracy: {test_accuracy}")
        
        logger.log("")

        outer_fold_number += 1
        
    # Doing one final parameter search on the entire training set to get best possible model for production
    # However, we cannot use this final model to estimate the accuracy as it has seen the entire training set

    def objective(trial):
            degree_of_polynomial = trial.suggest_int('degree_of_polynomial', params['degree_of_polynomial']['min'], params['degree_of_polynomial']['max'])
            n_estimators = trial.suggest_int('n_estimators', params['n_estimators']['min'], params['n_estimators']['max'])
            max_depth = trial.suggest_int('max_depth', params['max_depth']['min'], params['max_depth']['max'])
            min_samples_split = trial.suggest_int('min_samples_split', params['min_samples_split']['min'], params['min_samples_split']['max'])
            min_samples_leaf = trial.suggest_int('min_samples_leaf', params['min_samples_leaf']['min'], params['min_samples_leaf']['max'])
            bootstrap = trial.suggest_categorical('bootstrap', params['bootstrap'])
            criterion = trial.suggest_categorical('criterion', params['criterion'])

            model_for_tuning = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                bootstrap=bootstrap,
                criterion=criterion,
                random_state=42,
            ) 
            
            X_poly = [fit_polynomial(pulse, degree_of_polynomial) for pulse in X]
            X_poly = np.array(X_poly)

            model_for_tuning.fit(X_poly, y)
            score = model_for_tuning.score(X_poly, y)

            return score
    
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    final_study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner(), sampler=optuna.samplers.TPESampler(seed=42)) 
    final_study.optimize(objective, n_trials=params['optuna_n_trials'], n_jobs=-1)

    final_hyperparameters = final_study.best_params

    degree_of_polynomial = final_hyperparameters['degree_of_polynomial']
    n_estimators = final_hyperparameters['n_estimators']
    max_depth = final_hyperparameters['max_depth']
    min_samples_split = final_hyperparameters['min_samples_split']
    min_samples_leaf = final_hyperparameters['min_samples_leaf']
    bootstrap = final_hyperparameters['bootstrap']
    criterion = final_hyperparameters['criterion']

    model_final = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        bootstrap=bootstrap,
        criterion=criterion,
        random_state=42,
    )

    X_poly = [fit_polynomial(pulse, degree_of_polynomial) for pulse in X]
    X_poly = np.array(X_poly)

    model_final.fit(X_poly, y)

    # Averaging the final metrics
    final_training_accuracy = np.mean(training_accuracies)
    final_test_accuracy = np.mean(test_accuracies)
    final_balanced_train_accuracy = np.mean(balanced_training_accuracies)
    final_balanced_test_accuracy = np.mean(balanced_test_accuracies)
    final_precision = np.mean(precisions)
    final_recall = np.mean(recalls)
    TP, FP, FN, TN = final_confusion_matrix.ravel()
    final_specificity = TN / (TN + FP)
    final_sensitivity = TP / (TP + FN)
    final_f1 = np.mean(f1_scores)
    roc_auc = roc_auc_score(true_labels, predicted_labels)

    # standard deviation of the metrics
    std_training_accuracy = np.std(training_accuracies)
    std_test_accuracy = np.std(test_accuracies)
    std_balanced_train_accuracy = np.std(balanced_training_accuracies)
    std_balanced_test_accuracy = np.std(balanced_test_accuracies)
    std_precision = np.std(precisions)
    std_recall = np.std(recalls)
    std_f1 = np.std(f1_scores)


    # plotting confusion matrix
    cm_display = ConfusionMatrixDisplay(final_confusion_matrix, display_labels=le.classes_)
    cm_display.plot(values_format='.2f')
    plt.savefig("temp_plots/confusion_matrix.png")
    plt.close()

    # plotting roc curve
    fpr, tpr, _ = roc_curve(true_labels, predicted_labels)
    plt.plot(fpr, tpr, label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Random guessing (baseline)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend()
    plt.grid(True)
    plt.savefig("temp_plots/roc_curve.png")
    plt.close()
    
    # logging final metrics
    logger.log(f"Final best hyperparameters: {final_hyperparameters}")
    logger.log(f"Final estimated training accuracy by averaging: {round(final_training_accuracy, 3)}")
    logger.log(f"Final estimated test accuracy by averaging: {round(final_test_accuracy, 3)}")
    logger.log(f"Final estimated balanced training accuracy by averaging: {round(final_balanced_train_accuracy, 3)}")
    logger.log(f"Final estimated balanced test accuracy by averaging: {round(final_balanced_test_accuracy, 3)}")
    logger.log(f"Final estimated precision by averaging: {round(final_precision, 3)}")
    logger.log(f"Final estimated recall by averaging: {round(final_recall, 3)}")
    logger.log(f"Final estimated f1 score by averaging: {round(final_f1, 3)}")
    logger.log(f"Final estimated specificity: {round(final_specificity, 3)}")
    logger.log(f"Final estimated sensitivity: {round(final_sensitivity, 3)}")
    logger.log(f"Final estimated roc auc score: {round(roc_auc, 3)}")

    # logging standard deviation of the metrics
    logger.log(f"Standard deviation of training accuracy: {round(std_training_accuracy, 3)}")
    logger.log(f"Standard deviation of test accuracy: {round(std_test_accuracy, 3)}")
    logger.log(f"Standard deviation of balanced training accuracy: {round(std_balanced_train_accuracy, 3)}")
    logger.log(f"Standard deviation of balanced test accuracy: {round(std_balanced_test_accuracy, 3)}")
    logger.log(f"Standard deviation of precision: {round(std_precision, 3)}")
    logger.log(f"Standard deviation of recall: {round(std_recall, 3)}")
    logger.log(f"Standard deviation of f1 score: {round(std_f1, 3)}")


    metrics = {}
    metrics['train_accuracy'] = round(final_training_accuracy, 3)
    metrics['std_train_accuracy'] = round(std_training_accuracy, 3)
    metrics['test_accuracy'] = round(final_test_accuracy, 3)
    metrics['std_test_accuracy'] = round(std_test_accuracy, 3)
    metrics['balanced_train_accuracy'] = round(final_balanced_train_accuracy, 3)
    metrics['std_balanced_train_accuracy'] = round(std_balanced_train_accuracy, 3)
    metrics['balanced_test_accuracy'] = round(final_balanced_test_accuracy, 3)
    metrics['std_balanced_test_accuracy'] = round(std_balanced_test_accuracy, 3)
    metrics['precision'] = round(final_precision, 3)
    metrics['std_precision'] = round(std_precision, 3)
    metrics['recall'] = round(final_recall, 3)
    metrics['std_recall'] = round(std_recall, 3)
    metrics['specificity'] = round(final_specificity, 3)
    metrics['sensitivity'] = round(final_sensitivity, 3)
    metrics['f1'] = round(final_f1, 3)
    metrics['std_f1'] = round(std_f1, 3)
    metrics['roc_auc'] = round(roc_auc, 3)


    return model_final, metrics