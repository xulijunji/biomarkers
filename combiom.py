import itertools as it
import sklearn.linear_model as sk_lm
import sklearn.preprocessing as sk_pr
import sklearn.kernel_ridge as sk_kr
import pandas as pd
import csv
import numpy as np
import re
import string as st
import numexpr as ne


def init_iterators(parameters):

    iterators = {
            'a': list(range(parameters)),
            '1/a': list(range(parameters)),
            'a/b': list(it.permutations(range(parameters), 2)),
            'a/b/c': list(iter((x, y, z) for x, y, z in it.permutations(range(parameters), 3) if z > y)),
            'a*b/c': list(it.combinations(range(parameters), 3)),
            'a*b/c/d': list(iter((a, b, c, d) for a, b, c, d in it.permutations(range(parameters), 4) if b > a and d > c))
    }
    return iterators
        
    
def search(marker_names, marker_data, target_num, target_data, target_names, iterators, iterator_type='all'):

    results, temp = [], []
    queue = iterators.keys() if iterator_type == 'all' else iterator_type
        
    Pool = mp.Pool(mp.cpu_count())

    for it in queue:

        r = Pool.apply_async(__iterate_combinations, args=(marker_names, marker_data,
                                                           target_num, target_data, target_names,
                                                           iterators, it))
        temp.append(r)

    for r in temp:
        results.append(r.get())

    Pool.close()
    Pool.join()

    print('Search was successfully finished')
    return results


def to_dataframe(data):
        '''Data returned by .search() method'''

        # Importing results and adding to a list
        cols = ['Biomarker', 'Target marker',
                'M1', 'M2', 'M3', 'M4',
                'TS', 'R', 'KR', 'MID', 'TID', 'Type']
        dfl = [pd.DataFrame(i, columns=cols) for i in data]

        df = pd.concat(dfl)
        df = df.rename(columns={'TS': 'Theil-Sen Score', 'R': 'Ridge Score', 'KR': 'Kernel Ridge Score',
                                'M1': 'Marker 1', 'M2': 'Marker 2', 'M3': 'Marker 3', 'M4': 'Marker 4'})

        return df


def __iterate_combinations(marker_names, marker_data, target_num, target_data, target_names, iterators, iterator_type):

        output = {'Biomarker': [], 'Target marker': [],
                  'M1': [], 'M2': [], 'M3': [], 'M4': [],
                  'TS': [], 'R': [], 'KR': [],
                  'MID': [], 'TID': [], 'Type': []}

        print('Search:', iterator_type, 'started')
        
        for a, t in it.product(iterators[iterator_type], range(target_num)):

            list_ids = np.array([a]).ravel()
            list_ops = list(re.sub(r"\w", "", iterator_type))

            # Naming a combinatorial marker
            marker_name = __name(marker_names, iterator_type, list_ids)

            # Naming simple markers and padding list with np.nan to make it 4-element
            simple_markers_names = np.pad(marker_names[[list_ids]].astype('object'),
                                          (0, 4-len(list_ids)), mode='constant', constant_values=(0, np.nan))
            
            # Calculating marker
            marker_values = __calc_marker(marker_data, iterator_type, list_ids)

            # Preprocessing data: transforming NaN values into neighbours-mean
            # and normalizing data
            if np.isnan(marker_values).any():
                marker_values = __transform_nan(marker_values)
            marker_values_norm = __normalize(marker_values)
            marker_values_norm = marker_values_norm.reshape(-1, 1)

            # Marker IDs joining into a string
            mid = ', '.join(map(str, list_ids))

            # Naming a target
            target_name = target_names[t]
            target_values = target_data[t].reshape(-1, 1)
            tid = str(t)

            # Regression
            ts_score, ridge_score, kr_score = __regression(marker_values_norm, target_values)

            # Processing results
            if any(z > 0.8 for z in (np.around(ts_score, 1), np.around(ridge_score, 1), np.around(kr_score, 1))):

                for n, v in zip(['Biomarker', 'Target marker',
                                 'M1', 'M2',
                                 'M3', 'M4',
                                 'TS', 'R', 'KR', 'MID', 'TID', 'Type'],
                                [marker_name, target_name,
                                 simple_markers_names[0], simple_markers_names[1],
                                 simple_markers_names[2], simple_markers_names[3],
                                 ts_score, ridge_score, kr_score, mid, tid, iterator_type]):

                    output[n].append(v)

        print('Search:', iterator_type, 'finished')

        return output


def __regression(x, y, regressor_return=False):

    # TheilSen Regression
    ts_y = y.ravel()
    ts = sk_lm.TheilSenRegressor()
    ts.fit(x, ts_y)

    # r squared
    ts_y_pred = ts.predict(x)
    ts_y_mean = np.mean(ts_y)
    ts_ssr = np.sum((ts_y_pred - ts_y_mean) ** 2)
    ts_sst = np.sum((ts_y - ts_y_mean) ** 2)
    #ts_score = np.absolute(ts.score(index_norm, ts_y))
    ts_score = ts_ssr / ts_sst

    # Ridge Regression
    # Normalizating & reshaping
    ridge = sk_lm.Ridge(alpha=0.01, normalize=False)
    ridge.fit(x, y)
    ridge_score = np.absolute(ridge.score(x, y))

    # Kernel Ridge Regression
    kernel_ridge = sk_kr.KernelRidge(kernel='rbf', alpha=0.01)
    kernel_ridge.fit(x, y)
    kr_score = np.absolute(kernel_ridge.score(x, y))

    if regressor_return:
        return (ts, ridge, kernel_ridge)
    else:
        return (ts_score, ridge_score, kr_score)



def __transform_nan(x, strategy='mean'):

        imp = sk_pr.Imputer(missing_values='NaN', strategy=strategy)
        x = imp.fit_transform(x.reshape(-1, 1))

        return x


def __normalize(x, sample=None):

    if sample is None:
        sample = x
    return (x - np.mean(sample)) / np.std(sample)


def __name(marker_names, marker_type, ids):

    data = {}
    for l, p in zip(st.ascii_letters[: ids.size], marker_names[ids]):
        data[l] = p

    rep = dict((re.escape(k), v) for k, v in data.items())
    pattern = re.compile("|".join(rep.keys()))
    text = pattern.sub(lambda m: rep[re.escape(m.group(0))], marker_type)

    return text


def __calc_marker(marker_data, marker_type, ids):

    data = {}

    for l, p in zip(st.ascii_letters[: ids.size], marker_data[ids]):
        data[l] = p

    # Example: a*b/c (type), {a: ..., b: ..., c: ...} (data)
    marker = ne.evaluate(marker_type, data)
        
    return marker


def predict(marker_values, marker_data, markers_names, marker_names, marker_type, target_data, target_names, target_name):

    marker_ids = np.array([np.where(markers_names == n)[0][0] for n in marker_names])

    target_values = target_data[np.where(target_names == target_name)[0][0]].reshape(-1, 1)

    d = {}
    for i, (l, p) in enumerate(zip(st.ascii_letters[: len(marker_values)], marker_values)):
        d[l] = p

    marker_value = ne.evaluate(marker_type, d)

    # Calculating marker
    marker_train_values = __calc_marker(marker_data, marker_type, marker_ids)

    # Preprocessing data: transforming NaN values into neighbours-mean
    # and normalizing data 
    #marker_value = self.__transform_nan(marker_values)
    marker_value_norm = __normalize(marker_value, marker_train_values)
    marker_train_values_norm = __normalize(marker_train_values)
    marker_train_values_norm = marker_train_values_norm.reshape(-1, 1)

    # regression
    ts, ridge, kernel_ridge = __regression(marker_train_values_norm, target_values, regressor_return=True)
    return (10**ts.predict(marker_value_norm),
            10**ridge.predict(marker_value_norm),
            10**kernel_ridge.predict(marker_value_norm))


