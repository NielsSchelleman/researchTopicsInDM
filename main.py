import generator
import pandas as pd
import math
import ampute
import numpy as np

col = generator.column(datatype='ordinal', classes=6, proportions=[2, 5, 2, 1, 4, 3])
sample1 = pd.Series(col.generate_nominal_column(1000))
col2 = generator.column(datatype='numerical', classes=6, proportions=[2, 5, 2, 1, 4, 3])
sample2 = pd.Series(col2.generate_numerical_column(1000,3,5))
sample3 = pd.Series(col2.generate_numerical_column(1000,3,3))
sample4 = pd.Series(col2.generate_numerical_column(1000,2,3))
simple = pd.DataFrame([sample2,sample4,sample3]).transpose()


def ord_to_interval(column: pd.Series,
                    column_name: str = 'ordinal',
                    order: list = None,
                    proportion_based: bool = False,
                    relation=lambda x: x,
                    ):
    if not order:
        order = list(set(column))

    if proportion_based:
        counts = dict(column.value_counts())
    else:
        counts = dict.fromkeys(set(column), math.floor(len(column)/len(set(column))))

    init = 0
    ranges_start = {}
    ranges_end = {}

    for i in order:

        compute = relation(counts[i])
        ranges_start[i] = init
        ranges_end[i] = init+compute
        init = init+compute

    return pd.DataFrame([column.map(ranges_start), column.map(ranges_end)],
                        ['_start_'+column_name, '_end_'+column_name]).transpose()


intervals = ord_to_interval(sample1,
                            order=['rank0', 'rank1', 'rank2', 'rank3', 'rank4', 'rank5'],
                            column_name='sample1',
                            proportion_based=True,
                            relation= lambda x: x)


def doAmpute(ord_cols: list, data:pd.DataFrame, pattern, orders: list, lnames: list, lproportion_based: list, relations: list):
    interval_data = data.drop(ord_cols,axis=1)
    pattern = np.c_[pattern,np.ones(np.shape(pattern)[0])]
    for column in interval_data.columns:
        interval_data[column] = interval_data[column].astype('float')
    for i in range(len(ord_cols)):
        intervals = ord_to_interval(data[ord_cols[i]],
                                    order = orders[i],
                                    column_name= lnames[i],
                                    proportion_based= lproportion_based[i],
                                    relation= relations[i])
        interval_data = interval_data.join(intervals)

    a = (ampute.MultivariateAmputation(patterns=pattern).fit_transform(X=interval_data))
    rev_maps = {}
    for col in ord_cols:
        values = list(set(data[col].values))
        mapping = {}
        reverse_mapping = {}
        for i in range(len(values)):
            mapping[values[i]] = i
            reverse_mapping[i] = values[i]
        rev_maps[col] = reverse_mapping
        data[col] = data[col].map(mapping)
    data_removed = np.add(data, a)
    for col in ord_cols:
        data_removed[col] = data_removed[col].map(rev_maps[col])
    return data_removed


pattern = np.array([[0,1,1,1],[1,0,1,1],[1,1,0,1],[1,1,1,0]])
hos = pd.DataFrame([sample2,sample4,sample3,sample1]).transpose()

print(doAmpute(ord_cols=[3],
               data=hos,
               pattern = pattern,
               orders=[['rank0', 'rank1', 'rank2', 'rank3', 'rank4', 'rank5']],
               lnames=['sample1'],
               lproportion_based=[True],
               relations=[lambda x: x]
               ))




