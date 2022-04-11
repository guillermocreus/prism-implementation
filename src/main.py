import sys
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.preprocessing import KBinsDiscretizer


def discretize_df(X_trn, X_tst, n_bins=3):

    cols_to_discretize = [
        col for col in X_trn.columns if X_trn[col].dtype == float or X_trn[col].dtype == int]

    if len(cols_to_discretize) == 0:
        return X_trn, X_tst

    est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')

    X_trn_aux = X_trn[cols_to_discretize].to_numpy()
    X_tst_aux = X_tst[cols_to_discretize].to_numpy()

    if len(X_trn_aux.shape) > 1:
        X_trn_aux = est.fit_transform(X_trn_aux)
        X_tst_aux = est.transform(X_tst_aux)

    else:
        X_trn_aux = X_trn_aux[:, None]
        X_tst_aux = X_tst_aux[:, None]

        X_trn_aux = est.fit_transform(X_trn_aux)
        X_tst_aux = est.transform(X_tst_aux)

        X_trn_aux = X_trn_aux.reshape(X_trn_aux.shape[0])
        X_tst_aux = X_tst_aux.reshape(X_tst_aux.shape[0])

    X_trn_aux = X_trn_aux.astype(int).astype(str)
    X_tst_aux = X_tst_aux.astype(int).astype(str)

    X_trn[cols_to_discretize] = X_trn_aux
    X_tst[cols_to_discretize] = X_tst_aux

    dict_replace = {
        3: {'0': 'L', '1': 'M', '2': 'H'},
        5: {'0': 'LL', '1': 'L', '2': 'M', '3': 'H', '4': 'HH'}
    }

    X_trn = X_trn.replace(dict_replace[n_bins])
    X_tst = X_tst.replace(dict_replace[n_bins])

    return X_trn, X_tst


def compute_p_delta_over_alpha(X, y, values_x, value_y):

    cond_x = np.ones(len(X), dtype=bool)
    for key, value in values_x.items():
        cond_x *= (X[key] == value).to_numpy()

    cond_x_and_y = (y == value_y).to_numpy() * cond_x

    count_alpha_x = cond_x.sum()
    count_delta_n_and_alpha_x = cond_x_and_y.sum()

    if count_alpha_x > 0:
        return count_delta_n_and_alpha_x / count_alpha_x, count_delta_n_and_alpha_x

    return np.nan, 0


def filter_dfs(X, y, rule, keep=True):
    cond_x = np.ones(len(X), dtype=bool)
    for key, value in rule.items():
        cond_x *= (X[key] == value).to_numpy()

    if not keep:
        cond_x = np.array(1 - cond_x, dtype=bool)

    X = X[cond_x]
    y = y[cond_x]

    return X, y


def fit_rules(X_trn, y_trn, set_attribute_values, class_unique_values):
    all_Rs = []

    for class_value in class_unique_values:

        all_Rs_class = []
        still_instances_delta_n = True

        X_remaining, y_remaining = X_trn.copy(), y_trn.copy()
        X_rule, y_rule = X_trn.copy(), y_trn.copy()
        while still_instances_delta_n:

            number_of_attributes_of_rule = 0
            creating_rule = True
            X_rule, y_rule = X_remaining.copy(), y_remaining.copy()
            Rule = {}
            set_attributes_not_used = set_attribute_values.copy()

            while creating_rule:
                all_ps = []
                for (attribute_name, attribute_value) in set_attributes_not_used:
                    values_x = {attribute_name: attribute_value}
                    p_delta_alpha, n_delta_alpha = compute_p_delta_over_alpha(
                        X_rule, y_rule, values_x, class_value)
                    all_ps.append((attribute_name, attribute_value,
                                  (p_delta_alpha, n_delta_alpha)))

                all_ps = [elem for elem in all_ps if not np.isnan(elem[-1][0])]
                all_ps.sort(key=lambda tup: tup[-1])

                if len(all_ps):
                    rule_attribute_name, rule_attribute_value, (
                        rule_p, rule_n) = all_ps[-1]
                    number_of_attributes_of_rule += 1

                    Rule[rule_attribute_name] = rule_attribute_value
                    set_attributes_not_used.remove(
                        (rule_attribute_name, rule_attribute_value))

                    X_rule, y_rule = filter_dfs(
                        X_rule, y_rule, {rule_attribute_name: rule_attribute_value}, keep=True)

                    n_rule = (y_rule == class_value).to_numpy().sum()
                    c_rule = len(y_rule)
                    if n_rule == c_rule:
                        creating_rule = False

                        p_rule = n_rule / c_rule

                        all_Rs_class.append(
                            (Rule, class_value, p_rule, c_rule))

                        X_remaining, y_remaining = filter_dfs(
                            X_remaining, y_remaining, Rule, keep=False)

                elif number_of_attributes_of_rule == len(X_trn.columns):
                    creating_rule = False

                    n_rule = (y_rule == class_value).to_numpy().sum()
                    c_rule = len(y_rule)
                    p_rule = n_rule / c_rule

                    all_Rs_class.append((Rule, class_value, p_rule, c_rule))

                    X_remaining, y_remaining = filter_dfs(
                        X_remaining, y_remaining, Rule, keep=False)

            cond_y = (y_remaining == class_value).to_numpy()
            if cond_y.sum() == 0:
                for rule in all_Rs_class:
                    all_Rs.append(rule)

                still_instances_delta_n = False

    return all_Rs


def predict(X, rules, y_trn, dtype=str):
    predictions = pd.Series(np.zeros(len(X)), index=X.index, dtype=dtype)
    predictions[:] = np.nan

    for ind, row in X.iterrows():
        valid_rules = []
        for rule in rules:
            rule_attribute_value, class_value, _, c_rule = rule
            this_rule = True
            for attribute_name, attribute_value in rule_attribute_value.items():
                if row[attribute_name] != attribute_value:
                    this_rule = False

            if this_rule:
                valid_rules.append(rule)

        if len(valid_rules):
            max_n = -1
            selected_class_value = None
            for rule in valid_rules:
                _, class_value, _, c_rule = rule
                if c_rule > max_n:
                    max_n = c_rule
                    selected_class_value = class_value

            predictions[ind] = selected_class_value

        elif np.isnan(predictions[ind]):
            predictions[ind] = y_trn.mode()

    return predictions


if __name__ == "__main__":
    np.random.seed(2)

    # _____ READ DATA _____
    dataset_size = sys.argv[1]
    if dataset_size not in {"small", "medium", "large"}:
        raise Exception("Choose dataset size from the following set: {small, medium, large}")

    dataset_info = {
        "small": {
            "dataset_name": "wine",
            "class_name": "Class",
            "drop_fields": []
        },
        "medium": {
            "dataset_name": "breast-cancer-wisconsin",
            "class_name": "Class",
            "drop_fields": ["Sample code number"]
        },
        "large": {
            "dataset_name": "seismic-bumps",
            "class_name": "class",
            "drop_fields": []
        }
    }

    dataset_name = dataset_info[dataset_size]["dataset_name"]
    class_name = dataset_info[dataset_size]["class_name"]
    drop_fields = dataset_info[dataset_size]["drop_fields"]

    df = pd.read_csv('../data/' + dataset_name + ".csv")
    df = df.drop(drop_fields, axis=1)
    df = df.iloc[np.random.permutation(len(df))]
    # _______________

    # _____ Creation of train and test sets _____
    n_cut = int(0.8*len(df))
    df_trn = df[:n_cut]
    df_tst = df[n_cut:]

    X_trn = df_trn.drop(class_name, axis=1)
    y_trn = df_trn[class_name]

    X_tst = df_tst.drop(class_name, axis=1)
    y_tst = df_tst[class_name]
    # _______________

    # _____ Imputation of missing values _____
    if dataset_name == "breast-cancer-wisconsin":
        aux_col = X_trn["Bare Nuclei"]
        values, counts = np.unique(
            aux_col[aux_col != "?"].astype(int), return_counts=True)

        most_frequent_value = values[np.argmax(counts)]
        aux_col = aux_col.replace({"?": str(most_frequent_value)})

        X_trn["Bare Nuclei"] = aux_col.to_numpy().astype(int)

        X_tst["Bare Nuclei"] = X_tst["Bare Nuclei"].replace(
            {"?": str(most_frequent_value)})
        X_tst["Bare Nuclei"] = X_tst["Bare Nuclei"].astype(int)
    # _______________

    # _____ Double check for NaN values _____
    print(f'Dataset {dataset_name}:')
    print("-"*15)
    
    print("\n")
    print("NaN values (Train):")
    print("-"*30)
    for col in X_trn.columns:
        if X_trn.dtypes[col] == int or X_trn.dtypes[col] == float:
            print(f'Column {col} has {np.isnan(X_trn[col]).sum()} NaN values')
        else:
            print(f'Column {col} has unique values {X_trn[col].unique()}')

    print()
    print("NaN values (Test):")
    print("-"*30)
    for col in X_tst.columns:
        if X_tst.dtypes[col] == int or X_tst.dtypes[col] == float:
            print(f'Column {col} has {np.isnan(X_tst[col]).sum()} NaN values')
        else:
            print(f'Column {col} has unique values {X_tst[col].unique()}')
    # _______________

    # _____ Discretization of datasets _____
    X_trn, X_tst = discretize_df(X_trn, X_tst)

    # _____ Unique attribute and class values _____
    attributes_unique_values = {col_name: X_trn[col_name].unique(
    ) for col_name in X_trn.columns if col_name != class_name}
    class_unique_values = y_trn.unique()

    set_attribute_values = set([(attribute_name, attribute_value)
                                for attribute_name in attributes_unique_values for attribute_value in attributes_unique_values[attribute_name]])
    # _______________

    # _____ Fitting rules _____
    all_Rs = fit_rules(X_trn, y_trn, set_attribute_values, class_unique_values)
    # _______________

    # _____ Printing rules _____
    print("\n"*2)
    print("PRINTING RULES")
    print("-"*10)
    latex_mode = False

    if latex_mode:
        print("\\begin{itemize}")
        for ind, rule in enumerate(sorted(all_Rs, key=lambda tup: tup[-1], reverse=True)):
            s = f'R{ind + 1}: '
            rule_attribute_value, class_value, p_rule, c_rule = rule
            for attribute_name, attribute_value in rule_attribute_value.items():
                s += "\\textbf{" + attribute_name + "} = " + \
                    attribute_value + " $\\wedge$ "
            s = s[:-10] + " $\\rightarrow \delta_" + str(class_value) + "$ (n = " + str(
                c_rule) + ", p = " + str(round(100 * p_rule, 1)) + "\%)"
            print("\t\\item", s)
        print("\\end{itemize}")

    else:
        for ind, rule in enumerate(sorted(all_Rs, key=lambda tup: tup[-1], reverse=True)):
            s = f'R{ind + 1}: '
            rule_attribute_value, class_value, p_rule, c_rule = rule
            for attribute_name, attribute_value in rule_attribute_value.items():
                s += attribute_name + " = " + attribute_value + " && "
            s = s[:-4] + " --> " + str(class_value) + " (n = " + \
                str(c_rule) + ", p = " + str(round(100 * p_rule, 1)) + "%)"
            print(s)
    # _______________

    # _____ Computing predictions _____
    y_tst_hat = predict(X_tst, all_Rs, y_trn, dtype=y_tst.dtype)
    y_trn_hat = predict(X_trn, all_Rs, y_trn, dtype=y_trn.dtype)

    # _____ TEST metrics _____
    acc = accuracy_score(y_tst, y_tst_hat)

    if dataset_name == "seismic-bumps":
        precision = precision_score(y_tst, y_tst_hat)
        recall = recall_score(y_tst, y_tst_hat)
        f1 = f1_score(y_tst, y_tst_hat)

    elif dataset_name == "breast-cancer-wisconsin":
        precision = precision_score(y_tst, y_tst_hat, pos_label=4)
        recall = recall_score(y_tst, y_tst_hat, pos_label=4)
        f1 = f1_score(y_tst, y_tst_hat, pos_label=4)

    else:
        precision = precision_score(y_tst, y_tst_hat, average='macro')
        recall = recall_score(y_tst, y_tst_hat, average='macro')
        f1 = f1_score(y_tst, y_tst_hat, average='macro')

    precision_w = precision_score(y_tst, y_tst_hat, average='weighted')
    recall_w = recall_score(y_tst, y_tst_hat, average='weighted')
    f1_w = f1_score(y_tst, y_tst_hat, average='weighted')

    print("\n"*2)
    print("TEST METRICS")
    print("-"*10)
    
    print(f'Accuracy: {acc}')

    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1: {f1}')
    print()
    print(f'Precision (weighted): {precision_w}')
    print(f'Recall (weighted): {recall_w}')
    print(f'F1 (weighted): {f1_w}')
    print("_"*20, "\n")

    # _______________

    # _____ TRAIN metrics _____
    acc = accuracy_score(y_trn, y_trn_hat)

    if dataset_name == "seismic-bumps":
        precision = precision_score(y_trn, y_trn_hat)
        recall = recall_score(y_trn, y_trn_hat)
        f1 = f1_score(y_trn, y_trn_hat)

    elif dataset_name == "breast-cancer-wisconsin":
        precision = precision_score(y_trn, y_trn_hat, pos_label=4)
        recall = recall_score(y_trn, y_trn_hat, pos_label=4)
        f1 = f1_score(y_trn, y_trn_hat, pos_label=4)

    else:
        precision = precision_score(y_trn, y_trn_hat, average='macro')
        recall = recall_score(y_trn, y_trn_hat, average='macro')
        f1 = f1_score(y_trn, y_trn_hat, average='macro')

    precision_w = precision_score(y_trn, y_trn_hat, average='weighted')
    recall_w = recall_score(y_trn, y_trn_hat, average='weighted')
    f1_w = f1_score(y_trn, y_trn_hat, average='weighted')

    print()
    print("TRAIN METRICS")
    print("-"*10)
    
    print(f'Accuracy: {acc}')
    
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1: {f1}')
    print()
    print(f'Precision (weighted): {precision_w}')
    print(f'Recall (weighted): {recall_w}')
    print(f'F1 (weighted): {f1_w}')
    print("_"*20)

    # _______________
