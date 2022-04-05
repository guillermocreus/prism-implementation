import sys
import pandas as pd


dataset_name = ["wine", "adult+stretch"][-1]
# dataset_name = sys.argv[1]

df = pd.read_csv('data/' + dataset_name + ".csv")

X = df.drop('Class', axis=1)
y = df['Class']

# X_ = X.to_numpy()
# y_ = y.to_numpy()


def count_instances(X, y, values_x, value_y):
    # values_x: {col_name: attr_value}
    for col_name in values_x:
        X = X[X[col_name] == values_x[col_name]]

    y = y.iloc[X.index]
    y = y[y == value_y]

    if X.empty:
        return 0

    return len(y) / len(X)


values_x = {
    'Color': 'YELLOW',
    'size': 'SMALL'
}
value_y = 'T'

dict_unique_values = 1

print(count_instances(X, y, values_x, value_y))
