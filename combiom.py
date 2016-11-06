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


class Combiom:
    """Combiom class"""

    def __init__(self, pars, obs, vols = None):

        if not all(isinstance(a, (float, int)) 
                   for a in [pars, obs, vols]):
            raise ValueError('Please define class arguments: they must be integers and coinсide with data shape')

        self.parameters = int(pars)
        self.observations = int(obs)
        self.volunteers = int(vols)

        self.iterators = {
            'a': range(self.parameters),
            '1/a': range(self.parameters),
            'a/b': it.permutations(range(self.parameters), 2),
            'a/b/c': iter((x, y, z) for x, y, z in it.permutations(range(self.parameters), 3) if z > y),
            'a*b/c': it.combinations(range(self.parameters), 3),
            'a*b/c/d': iter((a, b, c, d) for a, b, c, d in it.permutations(range(self.parameters), 4) if b > a and d > c)
        }


    def load_target_data(self, target_names, target_data):
        '''def load_target_data(self, target_names, target_data)'''

        self.target_names = target_names
        self.target_data = target_data
        self.target_num = len(target_names)


    def load_marker_data(self, marker_names, marker_data):
        '''def load_marker_data(self, marker_names, marker_data):'''

        self.marker_names = marker_names
        self.marker_data = marker_data


    def search(self, it_type='all'):

        results = []
        queue = self.iterators.keys() if it_type == 'all' else it_type

        for i in queue:

            r = self.__iterate_combinations(i)
            results.append(r)

        print('Search was successfully finished')
        return results


    def to_dataframe(self, data):
        '''Data must be output of .search() method'''

        dfl = []

        # Importing results and adding to a list
        for i in data:
            dfl.append(pd.DataFrame(i, columns=['Biomarker', 'Target marker',
                                                'M1', 'M2', 'M3', 'M4',
                                                'TS', 'R', 'KR', 'MID', 'TID', 'Type']))

        df = pd.concat(dfl)
        df = df.rename(columns={'TS': 'Theil-Sen Score', 'R': 'Ridge Score', 'KR': 'Kernel Ridge Score',
                                'M1': 'Marker 1', 'M2': 'Marker 2', 'M3': 'Marker 3', 'M4': 'Marker 4'})

        return df


    def __iterate_combinations(self, iterator_type):

        p_num = self.parameters
        p_names = self.marker_names
        p_data = self.marker_data
        t_num = self.target_num
        t_names = self.target_names
        t_data = self.target_data

        output = {'Biomarker': [], 'Target marker': [],
                  'M1': [], 'M2': [], 'M3': [], 'M4': [],
                  'TS': [], 'R': [], 'KR': [],
                  'MID': [], 'TID': [], 'Type': []}

        print('Search:', iterator_type, 'started')

        for a, t in it.product(self.iterators[iterator_type], range(t_num)):

            list_ids = np.array([a]).ravel()
            list_ops = list(re.sub(r"\w", "", iterator_type))

            # Naming a combinatorial marker
            marker_name = self.__name(iterator_type, list_ids)

            # Naming simple markers and padding list with np.nan to make it 4-element
            simple_markers_names = np.pad(self.marker_names[[list_ids]].astype('object'),
                                          (0, 4-len(list_ids)), mode='constant', constant_values=(0, np.nan))

            # Calculating marker
            marker_value = self.__calc_marker(iterator_type, list_ids)

            # Preprocessing data: transforming NaN values into neighbours-mean
            # and normalizing data
            marker_value = self.__transform_nan(marker_value)
            marker_value_norm = self.__normalize(marker_value)

            # Marker IDs joining into a string
            mid = ', '.join(map(str, list_ids))

            # Naming a target
            target_name = t_names[t]
            target_data = np.copy(t_data[t].reshape(-1, 1))
            tid = str(t)

            # Regression
            ts_score, ridge_score, kr_score = self.__regression(marker_value_norm, target_data)

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

    def __regression(self, x, y, regressor_return=False):

        # TheilSen Regression
        ts_y = np.copy(y.ravel())
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
        kernel_ridge = sk_kr.KernelRidge(kernel='rbf', alpha=0.0001)
        kernel_ridge.fit(x, y)
        kr_score = np.absolute(kernel_ridge.score(x, y))

        if regressor_return:
            return (ts, ridge, kernel_ridge)
        else:
            return (ts_score, ridge_score, kr_score)



    def __transform_nan(self, x, strategy='mean'):

        imp = sk_pr.Imputer(missing_values='NaN', strategy=strategy)
        x = imp.fit_transform(x.reshape(-1, 1))

        return x


    def __normalize(self, x):

        return sk_pr.normalize(x, axis=0)


    def __name(self, marker_type, ids):

        data = {}
        for l, p in zip(st.ascii_letters[: ids.size], self.marker_names[ids]):
            data[l] = p

        rep = dict((re.escape(k), v) for k, v in data.items())
        pattern = re.compile("|".join(rep.keys()))
        text = pattern.sub(lambda m: rep[re.escape(m.group(0))], marker_type)

        return text


    def __calc_marker(self, marker_type, ids):

        data = {}

        for l, p in zip(st.ascii_letters[: ids.size], self.marker_data[ids]):
            data[l] = p

        # Example: a*b/c (type), {a: ..., b: ..., c: ...} (data)
        marker = ne.evaluate(marker_type, data)

        return marker


    def retrain(self, marker_names, marker_type, target_name):

        marker_ids = []

        for n in markers_names:
            marker_ids.append(np.where(self.marker_names == n)[0][0])

        target_data = self.target_data[np.where(self.target_names == target_name)[0][0]]

        d = {}
        for i, (l, p) in enumerate(zip(st.ascii_letters[: len(marker_ids)], self.marker_data[[marker_ids]])):
            d[l] = p

        x = ne.evaluate(marker_type, d)

        # Calculating marker
        marker_value = self.__calc_marker(marker_type, marker_ids)

        # Preprocessing data: transforming NaN values into neighbours-mean
        # and normalizing data 
        marker_value = self.__transform_nan(marker_value)
        marker_value_norm = self.__normalize(marker_value)

        # regression
        ts, ridge, kernel_ridge = self.__regression(index_norm, targetdata, regressor_return=True)
        return (ts, ridge, kernel_ridge)

