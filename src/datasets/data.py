from typing import Optional
import os
from multiprocessing import Pool, cpu_count
import glob
import re
import logging
from itertools import repeat, chain

import numpy as np
import pandas as pd
from tqdm import tqdm
from sktime.utils import load_data

from sagemaker_platform_code.src.datasets import utils
from sagemaker_platform_code.src.datasets import load_mdd_data

logger = logging.getLogger('__main__')


class Normalizer(object):
    """
    Normalizes dataframe across ALL contained rows (time steps). Different from per-sample normalization.
    """

    def __init__(self, norm_type, mean=None, std=None, min_val=None, max_val=None):
        """
        Args:
            norm_type: choose from:
                "standardization", "minmax": normalizes dataframe across ALL contained rows (time steps)
                "per_sample_std", "per_sample_minmax": normalizes each sample separately (i.e. across only its own rows)
            mean, std, min_val, max_val: optional (num_feat,) Series of pre-computed values
        """

        self.norm_type = norm_type
        self.mean = mean
        self.std = std
        self.min_val = min_val
        self.max_val = max_val

    def normalize(self, df):
        """
        Args:
            df: input dataframe
        Returns:
            df: normalized dataframe
        """
        if self.norm_type == "standardization":
            if self.mean is None:
                self.mean = df.mean()
                self.std = df.std()
            return (df - self.mean) / (self.std + np.finfo(float).eps)

        elif self.norm_type == "minmax":
            if self.max_val is None:
                self.max_val = df.max()
                self.min_val = df.min()
            return (df - self.min_val) / (self.max_val - self.min_val + np.finfo(float).eps)

        elif self.norm_type == "per_sample_std":
            grouped = df.groupby(by=df.index)
            return (df - grouped.transform('mean')) / grouped.transform('std')

        elif self.norm_type == "per_sample_minmax":
            grouped = df.groupby(by=df.index)
            min_vals = grouped.transform('min')
            return (df - min_vals) / (grouped.transform('max') - min_vals + np.finfo(float).eps)

        else:
            raise (NameError(f'Normalize method "{self.norm_type}" not implemented'))


def interpolate_missing(y):
    """
    Replaces NaN values in pd.Series `y` using linear interpolation
    """
    if y.isna().any():
        y = y.interpolate(method='linear', limit_direction='both')
    return y


def subsample(y, limit=512, factor=2):
    """
    If a given Series is longer than `limit`, returns subsampled sequence by the specified integer factor
    """
    # print('---- in subsample ----')
    # print(len(y))
    if len(y) > limit:
        return y[::factor].reset_index(drop=True)
    return y


def from_array_to_dataframe(X):
    # 一共三个维度,将每个人每个脑区的时间序列存入一个单元格中
    b_s = X.shape[0]
    n_rois = X.shape[1]
    n_ts = X.shape[2]
    # 创建空dataframe
    df = pd.DataFrame(dtype=np.float32)
    # 循环将时间序列数据写入dataframe中
    for v1 in range(n_rois):
        df['dim_' + str(v1)] = [X[v0][v1] for v0 in range(b_s)]

    print("from_array_to_dataframe----------")
    print(type(df))
    return df


def contact_array_in_dataframe(x, y):
    """
    input: x : aal 模板数据 (num_sub, 116, 116)
            y: cc200 datat (num_sub, 200, 200)
    output: dataframe from x and y
    """
    df = pd.DataFrame(dtype=np.float32)
    num_sub = x.shape[0]
    x_roi = x.shape[1]
    x_length = x.shape[2]
    y_roi = y.shape[1]
    y_length = y.shape[2]
    for i in range(y_roi+x_roi):
        if i < y_roi:
            df["dim_"+str(i)] = [y[v][i].tolist() for v in range(num_sub)]
        else:
            temp_all = []
            for v in range(num_sub):
                temp = x[v][i-y_roi].tolist()
                for j in range(y_roi-x_roi+1):
                    temp.append(np.nan)
                temp_all.append(temp)

            df["dim_"+str(i)] = temp_all
    # print('----- in contact array in dataframe ------')
    # print(df)

    return df

class BaseData(object):

    def set_num_processes(self, n_proc):

        if (n_proc is None) or (n_proc <= 0):
            self.n_proc = cpu_count()  # max(1, cpu_count() - 1)
        else:
            self.n_proc = min(n_proc, cpu_count())


class WeldData(BaseData):
    """
    Dataset class for welding dataset.
    Attributes:
        all_df: dataframe indexed by ID, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
        max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
            (Moreover, script argument overrides this attribute)
    """

    def __init__(self, root_dir, file_list=None, pattern=None, n_proc=1, limit_size=None, length_size=None,
                 dim_size=None, config=None):

        self.set_num_processes(n_proc=n_proc)

        self.all_df = self.load_all(root_dir, file_list=file_list, pattern=pattern)
        self.all_df = self.all_df.sort_values(by=['weld_record_index'])  # datasets is presorted
        # TODO: There is a single ID that causes the model output to become nan - not clear why
        self.all_df = self.all_df[self.all_df['weld_record_index'] != 920397]  # exclude particular ID
        self.all_df = self.all_df.set_index('weld_record_index')
        self.all_IDs = self.all_df.index.unique()  # all sample (session) IDs
        self.max_seq_len = 66
        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]

        self.feature_names = ['wire_feed_speed', 'current', 'voltage', 'motor_current', 'power']
        self.feature_df = self.all_df[self.feature_names]

    def load_all(self, root_dir, file_list=None, pattern=None):
        """
        Loads datasets from csv files contained in `root_dir` into a dataframe, optionally choosing from `pattern`
        Args:
            root_dir: directory containing all individual .csv files
            file_list: optionally, provide a list of file paths within `root_dir` to consider.
                Otherwise, entire `root_dir` contents will be used.
            pattern: optionally, apply regex string to select subset of files
        Returns:
            all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
        """
        # each file name corresponds to another date. Also tools (A, B) and others.

        # Select paths for training and evaluation
        if file_list is None:
            data_paths = glob.glob(os.path.join(root_dir, '*'))  # list of all paths
        else:
            data_paths = [os.path.join(root_dir, p) for p in file_list]
        if len(data_paths) == 0:
            raise Exception('No files found using: {}'.format(os.path.join(root_dir, '*')))

        if pattern is None:
            # by default evaluate on
            selected_paths = data_paths
        else:
            selected_paths = list(filter(lambda x: re.search(pattern, x), data_paths))

        input_paths = [p for p in selected_paths if os.path.isfile(p) and p.endswith('.csv')]
        if len(input_paths) == 0:
            raise Exception("No .csv files found using pattern: '{}'".format(pattern))

        if self.n_proc > 1:
            # Load in parallel
            _n_proc = min(self.n_proc, len(input_paths))  # no more than file_names needed here
            logger.info("Loading {} datasets files using {} parallel processes ...".format(len(input_paths), _n_proc))
            with Pool(processes=_n_proc) as pool:
                all_df = pd.concat(pool.map(WeldData.load_single, input_paths))
        else:  # read 1 file at a time
            all_df = pd.concat(WeldData.load_single(path) for path in input_paths)

        return all_df

    @staticmethod
    def load_single(filepath):
        df = WeldData.read_data(filepath)
        df = WeldData.select_columns(df)
        num_nan = df.isna().sum().sum()
        if num_nan > 0:
            logger.warning("{} nan values in {} will be replaced by 0".format(num_nan, filepath))
            df = df.fillna(0)

        return df

    @staticmethod
    def read_data(filepath):
        """Reads a single .csv, which typically contains a day of datasets of various weld sessions.
        """
        df = pd.read_csv(filepath)
        return df

    @staticmethod
    def select_columns(df):
        """"""
        df = df.rename(columns={"per_energy": "power"})
        # Sometimes 'diff_time' is not measured correctly (is 0), and power ('per_energy') becomes infinite
        is_error = df['power'] > 1e16
        df.loc[is_error, 'power'] = df.loc[is_error, 'true_energy'] / df['diff_time'].median()

        df['weld_record_index'] = df['weld_record_index'].astype(int)
        keep_cols = ['weld_record_index', 'wire_feed_speed', 'current', 'voltage', 'motor_current', 'power']
        df = df[keep_cols]

        return df


class TSRegressionArchive(BaseData):
    """
    Dataset class for datasets included in:
        1) the Time Series Regression Archive (www.timeseriesregression.org), or
        2) the Time Series Classification Archive (www.timeseriesclassification.com)
    Attributes:
        all_df: (num_samples * seq_len, num_columns) dataframe indexed by integer indices, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: (num_samples * seq_len, feat_dim) dataframe; contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: (num_samples,) series of IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
        labels_df: (num_samples, num_labels) pd.DataFrame of label(s) for each sample
        max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
            (Moreover, script argument overrides this attribute)
    """

    def __init__(self, root_dir, file_list=None, pattern=None, n_proc=1, limit_size=None, length_size=None,
                 dim_size=None, config=None):

        # self.set_num_processes(n_proc=n_proc)

        self.config = config
        self.all_df = pd.DataFrame()

        temp_df, self.labels_df = self.load_all(root_dir, file_list=file_list, pattern=pattern)
        self.all_IDs = temp_df.index.unique()  # all sample IDs (integer indices 0 ... num_samples-1)

        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            temp_df = temp_df.loc[self.all_IDs]

        if dim_size is not None:
            if dim_size > 1:
                min_size = int(dim_size)
                temp_df = temp_df.iloc[:, 0:dim_size]

        temp_df_cut = pd.DataFrame()
        if length_size is not None:
            if length_size > 1:
                length_size = int(length_size)
                for i in range(len(self.all_IDs)):
                    temp = temp_df.loc[i]
                    temp_df_cut = temp_df_cut.append(temp.iloc[0:length_size, :])
                temp_df = temp_df_cut

        self.all_df = temp_df

        # use all features
        self.feature_names = self.all_df.columns
        self.feature_df = self.all_df

        print('----------data.py feature_df ------------')
        print(self.feature_df)

    def load_all(self, root_dir, file_list=None, pattern=None):
        """
        Loads datasets from csv files contained in `root_dir` into a dataframe, optionally choosing from `pattern`
        Args:
            root_dir: directory containing all individual .csv files
            file_list: optionally, provide a list of file paths within `root_dir` to consider.
                Otherwise, entire `root_dir` contents will be used.
            pattern: optionally, apply regex string to select subset of files
        Returns:
            all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
            labels_df: dataframe containing label(s) for each sample
        """

        # Select paths for training and evaluation
        if file_list is None:
            data_paths = glob.glob(os.path.join(root_dir, '*'))  # list of all paths
        else:
            data_paths = [os.path.join(root_dir, p) for p in file_list]
        if len(data_paths) == 0:
            raise Exception('No files found using: {}'.format(os.path.join(root_dir, '*')))

        if pattern is None:
            # by default evaluate on
            selected_paths = data_paths
        else:
            selected_paths = list(filter(lambda x: re.search(pattern, x), data_paths))

        input_paths = [p for p in selected_paths if os.path.isfile(p) and p.endswith('.ts')]
        if len(input_paths) == 0:
            raise Exception("No .ts files found using pattern: '{}'".format(pattern))

        print('------------- input paths ---------------')
        print(input_paths)

        all_df, labels_df = self.load_single(input_paths[0])  # a single file contains dataset

        print('---------- load all func --------------')
        print(all_df)

        return all_df, labels_df

    def load_single(self, filepath):

        # Every row of the returned df corresponds to a sample;
        # every column is a pd.Series indexed by timestamp and corresponds to a different dimension (feature)
        if self.config['task'] == 'regression':
            df, labels = utils.load_from_tsfile_to_dataframe(filepath, return_separate_X_and_y=True,
                                                             replace_missing_vals_with='NaN')
            labels_df = pd.DataFrame(labels, dtype=np.float32)
        elif self.config['task'] == 'classification':
            df, labels = load_data.load_from_tsfile_to_dataframe(filepath, return_separate_X_and_y=True,
                                                                 replace_missing_vals_with='NaN')
            labels = pd.Series(labels, dtype="category")
            self.class_names = labels.cat.categories
            labels_df = pd.DataFrame(labels.cat.codes,
                                     dtype=np.int8)  # int8-32 gives an error when using nn.CrossEntropyLoss
            print('---------- in load single -------------')
            print('------- the df ------------')
            print(df)


        else:  # e.g. imputation
            try:
                data = load_data.load_from_tsfile_to_dataframe(filepath, return_separate_X_and_y=True,
                                                               replace_missing_vals_with='NaN')
                if isinstance(data, tuple):
                    df, labels = data
                else:
                    df = data
            except:
                df, _ = utils.load_from_tsfile_to_dataframe(filepath, return_separate_X_and_y=True,
                                                            replace_missing_vals_with='NaN')
            labels_df = None

        lengths = df.applymap(
            lambda x: len(x)).values  # (num_samples, num_dimensions) array containing the length of each series
        horiz_diffs = np.abs(lengths - np.expand_dims(lengths[:, 0], -1))

        # most general check: len(np.unique(lengths.values)) > 1:  # returns array of unique lengths of sequences
        if np.sum(horiz_diffs) > 0:  # if any row (sample) has varying length across dimensions
            logger.warning(
                "Not all time series dimensions have same length - will attempt to fix by subsampling first dimension...")
            df = df.applymap(subsample)  # TODO: this addresses a very specific case (PPGDalia)

        if self.config['subsample_factor']:
            df = df.applymap(lambda x: subsample(x, limit=0, factor=self.config['subsample_factor']))

        lengths = df.applymap(lambda x: len(x)).values
        vert_diffs = np.abs(lengths - np.expand_dims(lengths[0, :], 0))
        if np.sum(vert_diffs) > 0:  # if any column (dimension) has varying length across samples
            self.max_seq_len = int(np.max(lengths[:, 0]))
            logger.warning("Not all samples have same length: maximum length set to {}".format(self.max_seq_len))
        else:
            self.max_seq_len = lengths[0, 0]

        print('the max length:{}'.format(self.max_seq_len))

        # First create a (seq_len, feat_dim) dataframe for each sample, indexed by a single integer ("ID" of the sample)
        # Then concatenate into a (num_samples * seq_len, feat_dim) dataframe, with multiple rows corresponding to the
        # sample index (i.e. the same scheme as all datasets in this project)
        df = pd.concat((pd.DataFrame({col: df.loc[row, col] for col in df.columns}).reset_index(drop=True).set_index(
            pd.Series(lengths[row, 0] * [row])) for row in range(df.shape[0])), axis=0)

        # Replace NaN values
        grp = df.groupby(by=df.index)
        df = grp.transform(interpolate_missing)

        return df, labels_df


class PMUData(BaseData):
    """
    Dataset class for Phasor Measurement Unit dataset.
    Attributes:
        all_df: dataframe indexed by ID, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
        max_seq_len: maximum sequence (time series) length (optional). Used only if script argument `max_seq_len` is not
            defined.
    """

    def __init__(self, root_dir, file_list=None, pattern=None, n_proc=1, limit_size=None, length_size=None,
                 dim_size=None, config=None):

        self.set_num_processes(n_proc=n_proc)

        self.all_df = self.load_all(root_dir, file_list=file_list, pattern=pattern)

        if config['data_window_len'] is not None:
            self.max_seq_len = config['data_window_len']
            # construct sample IDs: 0, 0, ..., 0, 1, 1, ..., 1, 2, ..., (num_whole_samples - 1)
            # num_whole_samples = len(self.all_df) // self.max_seq_len  # commented code is for more general IDs
            # IDs = list(chain.from_iterable(map(lambda x: repeat(x, self.max_seq_len), range(num_whole_samples + 1))))
            # IDs = IDs[:len(self.all_df)]  # either last sample is completely superfluous, or it has to be shortened
            IDs = [i // self.max_seq_len for i in range(self.all_df.shape[0])]
            self.all_df.insert(loc=0, column='ExID', value=IDs)
        else:
            # self.all_df = self.all_df.sort_values(by=['ExID'])  # dataset is presorted
            self.max_seq_len = 30

        self.all_df = self.all_df.set_index('ExID')
        # rename columns
        self.all_df.columns = [re.sub(r'\d+', str(i // 3), col_name) for i, col_name in
                               enumerate(self.all_df.columns[:])]
        # self.all_df.columns = ["_".join(col_name.split(" ")[:-1]) for col_name in self.all_df.columns[:]]
        self.all_IDs = self.all_df.index.unique()  # all sample (session) IDs

        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]

        self.feature_names = self.all_df.columns  # all columns are used as features
        self.feature_df = self.all_df[self.feature_names]

    def load_all(self, root_dir, file_list=None, pattern=None):
        """
        Loads datasets from csv files contained in `root_dir` into a dataframe, optionally choosing from `pattern`
        Args:
            root_dir: directory containing all individual .csv files
            file_list: optionally, provide a list of file paths within `root_dir` to consider.
                Otherwise, entire `root_dir` contents will be used.
            pattern: optionally, apply regex string to select subset of files
        Returns:
            all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
        """

        # Select paths for training and evaluation
        if file_list is None:
            data_paths = glob.glob(os.path.join(root_dir, '*'))  # list of all paths
        else:
            data_paths = [os.path.join(root_dir, p) for p in file_list]
        if len(data_paths) == 0:
            raise Exception('No files found using: {}'.format(os.path.join(root_dir, '*')))

        if pattern is None:
            # by default evaluate on
            selected_paths = data_paths
        else:
            selected_paths = list(filter(lambda x: re.search(pattern, x), data_paths))

        input_paths = [p for p in selected_paths if os.path.isfile(p) and p.endswith('.csv')]
        if len(input_paths) == 0:
            raise Exception("No .csv files found using pattern: '{}'".format(pattern))

        if self.n_proc > 1:
            # Load in parallel
            _n_proc = min(self.n_proc, len(input_paths))  # no more than file_names needed here
            logger.info("Loading {} datasets files using {} parallel processes ...".format(len(input_paths), _n_proc))
            with Pool(processes=_n_proc) as pool:
                all_df = pd.concat(pool.map(PMUData.load_single, input_paths))
        else:  # read 1 file at a time
            all_df = pd.concat(PMUData.load_single(path) for path in input_paths)

        return all_df

    @staticmethod
    def load_single(filepath):
        df = PMUData.read_data(filepath)
        # df = PMUData.select_columns(df)
        num_nan = df.isna().sum().sum()
        if num_nan > 0:
            logger.warning("{} nan values in {} will be replaced by 0".format(num_nan, filepath))
            df = df.fillna(0)

        return df

    @staticmethod
    def read_data(filepath):
        """Reads a single .csv, which typically contains a day of datasets of various weld sessions.
        """
        df = pd.read_csv(filepath)
        return df


class MddDataset(BaseData):
    """
    Dataset class for datasets included in:
        1) the Time Series Regression Archive (www.timeseriesregression.org), or
        2) the Time Series Classification Archive (www.timeseriesclassification.com)
    Attributes:
        all_df: (num_samples * seq_len, num_columns) dataframe indexed by integer indices, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: (num_samples * seq_len, feat_dim) dataframe; contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: (num_samples,) series of IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
        labels_df: (num_samples, num_labels) pd.DataFrame of label(s) for each sample
        max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
            (Moreover, script argument overrides this attribute)
    """

    def __init__(self, root_dir, file_list=None, pattern=None, n_proc=1, limit_size=None, length_size=None,
                 dim_size=None, config=None):

        # self.set_num_processes(n_proc=n_proc)

        print(root_dir)
        self.root_dir = root_dir
        self.config = config

        self.all_df, self.labels_df = self.load_all(root_dir, file_list=file_list, pattern=pattern,
                                                    label_str=config['label_str'])

        print('-------------all_df:')
        print(type(self.all_df))
        print(self.all_df.size)

        self.all_IDs = self.all_df.index.unique()  # all sample IDs (integer indices 0 ... num_samples-1)

        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]

        # use all features
        self.feature_names = self.all_df.columns
        self.feature_df = self.all_df

    def load_all(self, root_dir, file_list=None, pattern=None, label_str=None):
        """
        Loads datasets from csv files contained in `root_dir` into a dataframe, optionally choosing from `pattern`
        将 “root_dir” 中包含的 csv 文件中的数据集加载到数据帧中，可以选择 “pattern”`
        Args:
            root_dir: directory containing all individual .csv files 包含所有单个.csv文件的目录
            file_list: optionally, provide a list of file paths within `root_dir` to consider.
                Otherwise, entire `root_dir` contents will be used.
            pattern: optionally, apply regex string to select subset of files

        Returns:
            all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
                    单个（可能是串联的）数据帧，其中包含与指定文件对应的所有数据
            labels_df: dataframe containing label(s) for each sample
                        包含每个样本标签的数据帧

        """

        # Select paths for training and evaluation
        if file_list is None:
            data_paths = glob.glob(os.path.join(root_dir, '*'))  # list of all paths  root_dir/*

        else:
            data_paths = [os.path.join(root_dir, p) for p in file_list]

        print(data_paths)

        if len(data_paths) == 0:
            raise Exception('No files found using: {}'.format(os.path.join(root_dir, '*')))

        if pattern is None:
            # by default evaluate on
            selected_paths = data_paths
        else:
            selected_paths = list(filter(lambda x: re.search(pattern, x), data_paths))

        all_df, labels_df = self.load_single(data_paths[0], data_paths[1],
                                             label_str=label_str)  # a single file contains dataset 一个单个的.ts文件

        print("load_all ---------- ：")
        print(type(all_df))
        print(type(labels_df))

        return all_df, labels_df

    def load_single(self, hc_file_path, mdd_file_path, label_str=None):

        if self.config['mdd_selected']:
            print('-------- the data is recurrent mdd --------------')
            # MddData = load_mdd_data.selected_MddData(mdd_file_path, hc_file_path)

            if self.config['mdd_data_type'] == 'tc':
                print('--------- data is tc ------')
                # all_df, all_labels = MddData.__getallitems__()

            elif self.config['mdd_data_type'] == 'corr':
                print('---------- data is corr -------')
                if self.config['augmentation']:
                    print('----------- augmentation has been applied --------')
                    file = np.load(os.path.join(self.root_dir, 'selected_mdd_aug10_data.npz'))
                    all_df = file['corrs']
                    all_labels = file['labels']

                else:
                    file_name = np.load(os.path.join(self.root_dir, 'selected_data.npz'))
                    all_df = file_name['corrs']
                    all_labels = file_name['labels']

            elif self.config['mdd_data_type'] == 'cc200_corr':
                print('----------- data is cc200 corr ---------------')
                print('----------- data has been combated --------------')
                file = np.load(os.path.join(r'/home/studio-lab-user/sagemaker-studiolab-notebooks/DataSets/mdd_cc200_datasets/mdd_selected_fc_combat.npz'))
                all_df = file['corr']
                all_labels = file['labels']

            elif self.config['mdd_data_type'] == 'aal+cc200_corr':
                print('--------- data type is aal+cc200 corr ----------------')
                file_aal = np.load(r'/home/studio-lab-user/sagemaker-studiolab-notebooks/selected_mdd_dataset/selected_data.npz')
                all_df1 = file_aal['corrs']
                all_labels = file_aal['labels']
                file_cc200 = np.load(
                    r'/home/studio-lab-user/sagemaker-studiolab-notebooks/DataSets/mdd_cc200_datasets/mdd_selected_fc_combat.npz')
                all_df2 = file_cc200['corr']
                print('all_df1:{}'.format(all_df1.shape))
                print('all_df2:{}'.format(all_df2.shape))
                all_df3 = contact_array_in_dataframe(all_df1, all_df2)


        else:
            MddData = load_mdd_data.MddData(mdd_file_path, hc_file_path, label_str=label_str)

            if self.config['mdd_data_type'] == 'tc':
                all_df, all_labels = MddData.__getallitems__()

            elif self.config['mdd_data_type'] == 'corr':
                print('------------ data is corr -----------------')

                if self.config['slide_window'] == 'in_datasets':
                    print('apply slide window in datasets')

                    if self.config['is_combat']:
                        print('-------- data after combat -----------------')
                        filename = r'/home/studio-lab-user/sagemaker-studiolab-notebooks/mydata/mdd_corr_up_flatten_combat.npz'

                    else:
                        filename = r'/home/studio-lab-user/sagemaker-studiolab-notebooks/mydata/mdd_corr_up_flatten.npz'

                    all_df, all_labels = utils.apply_slide_window_in_flatten_matrix(filename,
                                                                                    self.config['slide_window_size'],
                                                                                    self.config['slide_window_stride'],
                                                                                    6670, MddData.total_subjects)

                else:
                    if self.config['is_combat']:
                        print('-------- data after combat -----------------')
                        corr_flatten_combat_file = np.load(
                            '/home/studio-lab-user/sagemaker-studiolab-notebooks/mydata/mdd_corr_up_flatten_combat.npz')
                        all_df = utils.get_flatten2matrix(corr_flatten_combat_file['corrs'], 116)
                        all_labels = corr_flatten_combat_file['labels']
                    else:
                        if self.config['augmentation']:
                            print('------- data augmentation has been applied -----------')
                            corr = np.load('/home/studio-lab-user/sagemaker-studiolab-notebooks/mydata/mdd_data_aug_time90_upto10.npz')
                            all_df = corr['corrs']
                            all_labels = corr['labels']
                        else:
                            corr = np.load('/home/studio-lab-user/sagemaker-studiolab-notebooks/mydata/mdd_all_corr.npz')
                            all_df = corr['tc']
                            print('-------------- normalization -------------------')
                            mean = np.mean(all_df, axis=0)
                            std = np.std(all_df, axis=0)
                            all_df = (all_df-mean)/std
                            all_labels = corr['labels']

            elif self.config['mdd_data_type'] == 'cc200_corr':
                print('--------------- data is cc200 corr -----------')
                if self.config['is_combat']:
                    print('------------ data after combat ------------')
                    file = np.load(r'/home/studio-lab-user/sagemaker-studiolab-notebooks/DataSets/mdd_cc200_datasets/mdd_cc200_combat_fc.npz')
                    all_df = file['corr']
                    all_labels = file['labels']
                else:
                    file = np.load(
                        r'/home/studio-lab-user/sagemaker-studiolab-notebooks/DataSets/mdd_cc200_datasets/mdd_cc200_fc.npz')
                    all_df = file['corr']
                    all_labels = file['labels']


            elif self.config['mdd_data_type'] == 'aal+cc200_corr':
                print('------------- data is aal and cc200 corr -------------')
                file_aal = np.load(r'/home/studio-lab-user/sagemaker-studiolab-notebooks/mydata/mdd_all_corr.npz')
                all_df1 = file_aal['tc']
                all_labels = file_aal['labels']
                file_cc200 = np.load(r'/home/studio-lab-user/sagemaker-studiolab-notebooks/DataSets/mdd_cc200_datasets/mdd_cc200_combat_fc.npz')
                all_df2 = file_cc200['corr']
                print('all_df1:{}'.format(all_df1.shape))
                print('all_df2:{}'.format(all_df2.shape))
                all_df3 = contact_array_in_dataframe(all_df1, all_df2)


            elif self.config['mdd_data_type'] == 'ec':
                print('-------------- data is ec ------------')

                if self.config['slide_window'] == 'in_datasets':
                    print('apply slide window in datasets')
                    file = np.load('/home/studio-lab-user/sagemaker-studiolab-notebooks/mydata/mdd_ec_data.npz')
                    filename = np.load('/home/studio-lab-user/sagemaker-studiolab-notebooks/mydata/mdd_ec_y2x_flatten.npz')
                    all_df, all_labels = utils.apply_slide_window_in_flatten_matrix(filename,
                                                                                    self.config['slide_window_size'],
                                                                                    self.config['slide_window_stride'],
                                                                                    6670, MddData.total_subjects)
                else:
                    file = r'/home/studio-lab-user/sagemaker-studiolab-notebooks/DataSets/mdd_ec_dataset/ECMatrix.npy'
                    data = load_mdd_data.Mdd_EC(file)
                    all_df, all_labels = data.__getallitems__()

            elif self.config['mdd_data_type'] == 'mixcoff':
                print('-------------- data is mixcoff ------------')
                dataset = load_mdd_data.MDD_mixcoff_data(matfile=r'/home/studio-lab-user/sagemaker-studiolab-notebooks/DataSets/zxydata/mixcoff_combat.mat',
                                                         labelfile=r'/home/studio-lab-user/sagemaker-studiolab-notebooks/DataSets/zxydata/mddNC_label.mat')
                all_df, all_labels = dataset.__getallitems__()

            elif self.config['mdd_data_type'] == 'svd_pca_100_tc':
                print('-------------- data is svd_100_tc ------------')
                svd_100_tc = np.load('/home/studio-lab-user/sagemaker-studiolab-notebooks/mydata/mddall_tc_SVD_100.npz')
                all_df = svd_100_tc['tc']
                all_labels = svd_100_tc['label']

            elif self.config['mdd_data_type'] == 'svd_np_100_tc_ur':
                print('-------------- data is svd_100_tc_ur ------------')
                svd_100_tc = np.load('/home/studio-lab-user/sagemaker-studiolab-notebooks/mydata/mddall_tc_npSVD_100.npz')
                all_df = svd_100_tc['tc_ur']
                all_labels = svd_100_tc['label']

            elif self.config['mdd_data_type'] == 'svd_np_100_tc_vr':
                print('-------------- data is svd_100_tc_vr ------------')
                svd_100_tc = np.load('/home/studio-lab-user/sagemaker-studiolab-notebooks/mydata/mddall_tc_npSVD_100.npz')
                all_df = svd_100_tc['tc_vr']
                all_labels = svd_100_tc['label']

            elif self.config['mdd_data_type'] == 'svd_pca_78_tc':
                print('-------------- data is svd_78_tc ------------')
                svd_100_tc = np.load('/home/studio-lab-user/sagemaker-studiolab-notebooks/mydata/mddall_tc_SVD_pca_78.npz')
                all_df = svd_100_tc['tc']
                all_labels = svd_100_tc['label']

            elif self.config['mdd_data_type'] == 'svd_pca_100_corr':
                print('-------------- data is svd_100_corr ------------')
                svd_100_corr = np.load(
                    '/home/studio-lab-user/sagemaker-studiolab-notebooks/mydata/mddall_corr_SVD_pca_100.npz')
                all_df = svd_100_corr['corrs']
                all_labels = svd_100_corr['labels']


            elif self.config['mdd_data_type'] == 'slide_window_60':
                print('-------------- data is slide_window_60 ------------')
                slide_window_60 = np.load(
                    '/home/studio-lab-user/sagemaker-studiolab-notebooks/mydata/mdd_slideWindow60_3222.npz')
                all_df = slide_window_60['tcs']
                all_labels = slide_window_60['labels']

            elif self.config['mdd_data_type'] == 'slide_window_60_corr':
                print('-------------- data is slide_window_60_corr ------------')
                slide_window_60_corr = np.load(
                    '/home/studio-lab-user/sagemaker-studiolab-notebooks/mydata/mdd_corr_slideWindow60_3222.npz')
                all_df = slide_window_60_corr['corrs']
                all_labels = slide_window_60_corr['labels']

            elif self.config['mdd_data_type'] == 'ics_30':
                print('-------------- data is ics_30 ------------')
                ics_30 = np.load('/home/studio-lab-user/sagemaker-studiolab-notebooks/mydata/mdd_ica_30x140.npz')
                all_df = ics_30['ics']
                all_labels = ics_30['labels']

            elif self.config['mdd_data_type'] == 'ics_50':
                print('-------------- data is ics_50 ------------')
                ics_50 = np.load('/home/studio-lab-user/sagemaker-studiolab-notebooks/mydata/mdd_ica_50x140.npz')
                all_df = ics_50['ics']
                all_labels = ics_50['labels']

            elif self.config['mdd_data_type'] == 'ics_50x116':
                print('-------------- data is ics_50x116 ------------')
                ics_50x116 = np.load('/home/studio-lab-user/sagemaker-studiolab-notebooks/mydata/mdd_ica_50x116.npz')
                all_df = ics_50x116['ics']
                all_labels = ics_50x116['labels']

            elif self.config['mdd_data_type'] == 'ics_100x116':
                print('-------------- data is ics_100x116 ------------')
                ics_100x116 = np.load('/home/studio-lab-user/sagemaker-studiolab-notebooks/mydata/mdd_ica_100x116.npz')
                all_df = ics_100x116['ics']
                all_labels = ics_100x116['labels']

        if self.config['select_rois']:
            print('----- rois are selected ------------')
            file = np.load(self.config['roi_file'])
            first_level = file['first_level']
            second_level = file['second_level']
            third_level = file['third_level']

            all_index = np.append(first_level, second_level, axis=0)
            all_index = np.append(all_index, third_level, axis=0).tolist()

            all_df_new = np.zeros((all_df.shape[0], all_df.shape[1], len(all_index)))
            for i in range(len(all_df)):
                all_df_new[i] = all_df[i][:, all_index]

        else:
            if self.config['mdd_data_type'] != 'aal+cc200_corr':
                all_df_new = all_df

        if self.config['mdd_data_type'] == 'aal+cc200_corr':
            df = all_df3
        elif self.config['mdd_data_type'] == 'mixcoff':
            all_df = all_df[:, np.newaxis]
            df = from_array_to_dataframe(all_df)

        else:
            df = from_array_to_dataframe(all_df_new)

        all_labels = np.squeeze(all_labels)
        print('all_labels.shape:{}'.format(all_labels.shape))

        labels = np.delete(all_labels, 1, 1).astype(str)
        labels_flatten = labels.flatten()

        labels_flatten = pd.Series(labels_flatten, dtype="category")
        self.class_names = labels_flatten.cat.categories
        labels_df = pd.DataFrame(labels_flatten.cat.codes,
                                 dtype=np.int8)  # int8-32 gives an error when using nn.CrossEntropyLoss

        print(' ------ labels in data.py load single ---------')
        print(labels.shape)  # 1 维的，没有问题
        print(f'df:{df}')

        lengths = df.applymap(
            lambda x: len(x)).values  # (num_samples, num_dimensions) array containing the length of each series
        horiz_diffs = np.abs(lengths - np.expand_dims(lengths[:, 0], -1))

        # most general check: len(np.unique(lengths.values)) > 1:  # returns array of unique lengths of sequences
        if np.sum(horiz_diffs) > 0:  # if any row (sample) has varying length across dimensions
            logger.warning(
                "Not all time series dimensions have same length - will attempt to fix by subsampling first dimension...")
            df = df.applymap(subsample)  # TODO: this addresses a very specific case (PPGDalia)

        if self.config['subsample_factor']:
            df = df.applymap(lambda x: subsample(x, limit=0, factor=self.config['subsample_factor']))

        lengths = df.applymap(lambda x: len(x)).values
        vert_diffs = np.abs(lengths - np.expand_dims(lengths[0, :], 0))
        if np.sum(vert_diffs) > 0:  # if any column (dimension) has varying length across samples
            self.max_seq_len = int(np.max(lengths[:, 0]))
            logger.warning("Not all samples have same length: maximum length set to {}".format(self.max_seq_len))
        else:
            self.max_seq_len = lengths[0, 0]

        print('self.max_seq_len:')
        print(self.max_seq_len)
        # First create a (seq_len, feat_dim) dataframe for each sample, indexed by a single integer ("ID" of the sample)
        # Then concatenate into a (num_samples * seq_len, feat_dim) dataframe, with multiple rows corresponding to the
        # sample index (i.e. the same scheme as all datasets in this project)
        print('the df is:')
        print(df)
        df = pd.concat((pd.DataFrame({col: df.loc[row, col] for col in df.columns}).reset_index(drop=True).set_index(
            pd.Series(lengths[row, 0] * [row])) for row in range(df.shape[0])), axis=0)

        # Replace NaN values
        grp = df.groupby(by=df.index)
        df = grp.transform(interpolate_missing)

        return df, labels_df


class ADHDDataset(BaseData):
    def __init__(self, root_dir, site_list=None,
                 pattern=None, n_proc=1, limit_size=None, length_size=None,
                 dim_size=None, config=None):

        # self.set_num_processes(n_proc=n_proc)

        print(root_dir)

        self.config = config

        self.all_df, self.labels_df = self.load_all(root_dir, site_list=site_list, pattern=pattern,
                                                    label_str=config['label_str'])

        print('-------------all_df:')
        print(type(self.all_df))
        print(self.all_df.size)

        self.all_IDs = self.all_df.index.unique()  # all sample IDs (integer indices 0 ... num_samples-1)

        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]

        # use all features
        self.feature_names = self.all_df.columns
        self.feature_df = self.all_df

    def load_all(self, root_dir, site_list=None, pattern=None, label_str=None):
        """
        Loads datasets from csv files contained in `root_dir` into a dataframe, optionally choosing from `pattern`
        将 “root_dir” 中包含的 csv 文件中的数据集加载到数据帧中，可以选择 “pattern”`
        Args:
            root_dir: directory containing all individual .csv files 包含所有单个.csv文件的目录
            file_list: optionally, provide a list of file paths within `root_dir` to consider.
                Otherwise, entire `root_dir` contents will be used.
            pattern: optionally, apply regex string to select subset of files

        Returns:
            all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
                    单个（可能是串联的）数据帧，其中包含与指定文件对应的所有数据
            labels_df: dataframe containing label(s) for each sample
                        包含每个样本标签的数据帧

        """

        # Select paths for training and evaluation
        if site_list is None:
            data_paths = glob.glob(os.path.join(root_dir, '*'))  # list of all paths  root_dir/*

        else:
            # data_paths = [os.path.join(root_dir, p) for p in file_list]
            data_paths = site_list

        print(data_paths)

        if len(data_paths) == 0:
            raise Exception('No files found using: {}'.format(os.path.join(root_dir, '*')))

        # if pattern is None:
        #     # by default evaluate on
        #     selected_paths = data_paths
        # else:
        selected_paths = list(filter(lambda x: re.search('.npz', x), data_paths))
        print('selected_paths:')
        print(selected_paths)

        all_df, labels_df = self.load_single(root_dir, site_name=selected_paths,
                                             label_str=label_str)  # a single file contains dataset 一个单个的.ts文件

        print("load_all ---------- ：")
        print(type(all_df))
        print(type(labels_df))

        return all_df, labels_df

    def load_single(self, alldata_path,
                    site_name=None,
                    time_length=232,
                    num_roi=116,
                    label_str=None):
        adhdData = load_mdd_data.ADHDData(alldata_path=alldata_path,
                                          site_name=site_name,
                                          time_length=time_length,
                                          num_roi=num_roi)
        all_tc, all_corrs, all_labels = adhdData.__getallitems__()

        if self.config['adhd_data_type'] == 'tc':
            print('----------- data type is tc ---------------')
            all_df = all_tc

        elif self.config['adhd_data_type'] == 'corr':
            print('----------- data type is corr ---------------')
            all_df = all_corrs

        df = from_array_to_dataframe(all_df)
        labels = np.delete(all_labels, 1, 1).astype(str)
        labels_flatten = labels.flatten()

        labels_flatten = pd.Series(labels_flatten, dtype="category")
        self.class_names = labels_flatten.cat.categories
        labels_df = pd.DataFrame(labels_flatten.cat.codes,
                                 dtype=np.int8)  # int8-32 gives an error when using nn.CrossEntropyLoss

        lengths = df.applymap(
            lambda x: len(x)).values  # (num_samples, num_dimensions) array containing the length of each series
        horiz_diffs = np.abs(lengths - np.expand_dims(lengths[:, 0], -1))

        # most general check: len(np.unique(lengths.values)) > 1:  # returns array of unique lengths of sequences
        if np.sum(horiz_diffs) > 0:  # if any row (sample) has varying length across dimensions
            logger.warning(
                "Not all time series dimensions have same length - will attempt to fix by subsampling first dimension...")
            df = df.applymap(subsample)  # TODO: this addresses a very specific case (PPGDalia)

        if self.config['subsample_factor']:
            df = df.applymap(lambda x: subsample(x, limit=0, factor=self.config['subsample_factor']))

        lengths = df.applymap(lambda x: len(x)).values
        vert_diffs = np.abs(lengths - np.expand_dims(lengths[0, :], 0))
        if np.sum(vert_diffs) > 0:  # if any column (dimension) has varying length across samples
            self.max_seq_len = int(np.max(lengths[:, 0]))
            logger.warning("Not all samples have same length: maximum length set to {}".format(self.max_seq_len))
        else:
            self.max_seq_len = lengths[0, 0]

        # First create a (seq_len, feat_dim) dataframe for each sample, indexed by a single integer ("ID" of the sample)
        # Then concatenate into a (num_samples * seq_len, feat_dim) dataframe, with multiple rows corresponding to the
        # sample index (i.e. the same scheme as all datasets in this project)
        df = pd.concat((pd.DataFrame({col: df.loc[row, col] for col in df.columns}).reset_index(drop=True).set_index(
            pd.Series(lengths[row, 0] * [row])) for row in range(df.shape[0])), axis=0)

        # Replace NaN values
        grp = df.groupby(by=df.index)
        df = grp.transform(interpolate_missing)

        return df, labels_df


class ABIDEDataset(BaseData):
    def __init__(self, root_dir,  file_list=None,
                 pattern=None, n_proc=1, limit_size=None, length_size=None,
                 dim_size=None, config=None):

        # self.set_num_processes(n_proc=n_proc)

        print(root_dir)

        self.config = config

        self.all_df, self.labels_df = self.load_all(root_dir, file_list=file_list, pattern=pattern,
                                                    label_str=config['label_str'])

        print('-------------all_df:')
        print(type(self.all_df))
        print(self.all_df.size)

        self.all_IDs = self.all_df.index.unique()  # all sample IDs (integer indices 0 ... num_samples-1)

        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]

        # use all features
        self.feature_names = self.all_df.columns
        self.feature_df = self.all_df

    def load_all(self, root_dir,  file_list=None, pattern=None, label_str=None):
        """
        Loads datasets from csv files contained in `root_dir` into a dataframe, optionally choosing from `pattern`
        将 “root_dir” 中包含的 csv 文件中的数据集加载到数据帧中，可以选择 “pattern”`
        Args:
            root_dir: directory containing all individual .csv files 包含所有单个.csv文件的目录
            file_list: optionally, provide a list of file paths within `root_dir` to consider.
                Otherwise, entire `root_dir` contents will be used.
            pattern: optionally, apply regex string to select subset of files

        Returns:
            all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
                    单个（可能是串联的）数据帧，其中包含与指定文件对应的所有数据
            labels_df: dataframe containing label(s) for each sample
                        包含每个样本标签的数据帧

        """

        # Select paths for training and evaluation
        if file_list is None:
            data_paths = glob.glob(os.path.join(root_dir, '*'))  # list of all paths  root_dir/*

        else:
            # data_paths = [os.path.join(root_dir, p) for p in file_list]
            data_paths = file_list

        print(data_paths)

        if len(data_paths) == 0:
            raise Exception('No files found using: {}'.format(os.path.join(root_dir, '*')))

        # if pattern is None:
        #     # by default evaluate on
        #     selected_paths = data_paths
        # else:
        selected_paths = list(filter(lambda x: re.search('.npz', x), data_paths))
        print('selected_paths:')
        print(selected_paths)

        all_df, labels_df = self.load_single(selected_paths[0],
                                             label_str=label_str)  # a single file contains dataset 一个单个的.ts文件

        print("load_all ---------- ：")
        print(type(all_df))
        print(type(labels_df))

        return all_df, labels_df

    def load_single(self, alldata_path,
                    time_length=78,
                    num_roi=116,
                    label_str=None):
        adhdData = load_mdd_data.ABIDE_Data(data_path=alldata_path,
                                            time_length=time_length,
                                            num_roi=num_roi)
        all_tc, all_corrs, all_labels = adhdData.__getallitems__()

        if self.config['abide_data_type'] == 'tc':
            print('----------- data type is tc ---------------')
            all_df = all_tc

        elif self.config['abide_data_type'] == 'corr':
            print('----------- data type is corr ---------------')
            all_df = all_corrs

        df = from_array_to_dataframe(all_df)
        labels = np.delete(all_labels, 1, 1).astype(str)
        labels_flatten = labels.flatten()

        labels_flatten = pd.Series(labels_flatten, dtype="category")
        self.class_names = labels_flatten.cat.categories
        labels_df = pd.DataFrame(labels_flatten.cat.codes,
                                 dtype=np.int8)  # int8-32 gives an error when using nn.CrossEntropyLoss

        lengths = df.applymap(
            lambda x: len(x)).values  # (num_samples, num_dimensions) array containing the length of each series
        horiz_diffs = np.abs(lengths - np.expand_dims(lengths[:, 0], -1))

        # most general check: len(np.unique(lengths.values)) > 1:  # returns array of unique lengths of sequences
        if np.sum(horiz_diffs) > 0:  # if any row (sample) has varying length across dimensions
            logger.warning(
                "Not all time series dimensions have same length - will attempt to fix by subsampling first dimension...")
            df = df.applymap(subsample)  # TODO: this addresses a very specific case (PPGDalia)

        if self.config['subsample_factor']:
            df = df.applymap(lambda x: subsample(x, limit=0, factor=self.config['subsample_factor']))

        lengths = df.applymap(lambda x: len(x)).values
        vert_diffs = np.abs(lengths - np.expand_dims(lengths[0, :], 0))
        if np.sum(vert_diffs) > 0:  # if any column (dimension) has varying length across samples
            self.max_seq_len = int(np.max(lengths[:, 0]))
            logger.warning("Not all samples have same length: maximum length set to {}".format(self.max_seq_len))
        else:
            self.max_seq_len = lengths[0, 0]

        # First create a (seq_len, feat_dim) dataframe for each sample, indexed by a single integer ("ID" of the sample)
        # Then concatenate into a (num_samples * seq_len, feat_dim) dataframe, with multiple rows corresponding to the
        # sample index (i.e. the same scheme as all datasets in this project)
        df = pd.concat((pd.DataFrame({col: df.loc[row, col] for col in df.columns}).reset_index(drop=True).set_index(
            pd.Series(lengths[row, 0] * [row])) for row in range(df.shape[0])), axis=0)

        # Replace NaN values
        grp = df.groupby(by=df.index)
        df = grp.transform(interpolate_missing)

        return df, labels_df

class HCDataset(BaseData):
    def __init__(self, root_dir,
                 pattern=None, n_proc=1, limit_size=None, length_size=None,
                 dim_size=None, config=None):

        # self.set_num_processes(n_proc=n_proc)

        print(root_dir)

        self.config = config

        self.all_df, self.labels_df = self.load_all()

        print('-------------all_df:')
        print(type(self.all_df))
        print(self.all_df.size)

        self.all_IDs = self.all_df.index.unique()  # all sample IDs (integer indices 0 ... num_samples-1)

        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]

        # use all features
        self.feature_names = self.all_df.columns
        self.feature_df = self.all_df

    def load_all(self):
        """
        Loads datasets from csv files contained in `root_dir` into a dataframe, optionally choosing from `pattern`
        将 “root_dir” 中包含的 csv 文件中的数据集加载到数据帧中，可以选择 “pattern”`
        Args:
            root_dir: directory containing all individual .csv files 包含所有单个.csv文件的目录
            file_list: optionally, provide a list of file paths within `root_dir` to consider.
                Otherwise, entire `root_dir` contents will be used.
            pattern: optionally, apply regex string to select subset of files

        Returns:
            all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
                    单个（可能是串联的）数据帧，其中包含与指定文件对应的所有数据
            labels_df: dataframe containing label(s) for each sample
                        包含每个样本标签的数据帧

        """

        # Select paths for training and evaluation
        # if file_list is None:
        #     data_paths = glob.glob(os.path.join(root_dir, '*'))  # list of all paths  root_dir/*
        #
        # else:
        #     # data_paths = [os.path.join(root_dir, p) for p in file_list]
        #     data_paths = file_list
        #
        # print(data_paths)
        #
        # if len(data_paths) == 0:
        #     raise Exception('No files found using: {}'.format(os.path.join(root_dir, '*')))
        #
        # # if pattern is None:
        # #     # by default evaluate on
        # #     selected_paths = data_paths
        # # else:
        # selected_paths = list(filter(lambda x: re.search('.npz', x), data_paths))
        # print('selected_paths:')
        # print(selected_paths)

        all_df, labels_df = self.load_single()  # a single file contains dataset 一个单个的.ts文件

        print("load_all ---------- ：")
        print(type(all_df))
        print(type(labels_df))

        return all_df, labels_df

    def load_single(self):

        abide_file_path = r'/home/studio-lab-user/sagemaker-studiolab-notebooks/ABIDEdataset/abide_aal.npz'
        mdd_file_path = r'/home/studio-lab-user/sagemaker-studiolab-notebooks/mydata/mdd_all_corr.npz'
        adhd_file_path = r'/home/studio-lab-user/sagemaker-studiolab-notebooks/ADHDdataset/AAL_all_data'
        HCData = load_mdd_data.ALL_HC_DATA(abide_file_path, mdd_file_path, adhd_file_path)
        all_corrs, all_labels = HCData.__getallitems__()

        # if self.config['hc_data_type'] == 'tc':
        #     print('----------- data type is tc ---------------')
        #     all_df = all_tc

        if self.config['hc_data_type'] == 'corr':
            print('----------- data type is corr ---------------')
            all_df = all_corrs

        df = from_array_to_dataframe(all_df)
        labels = np.delete(all_labels, 1, 1).astype(str)
        labels_flatten = labels.flatten()

        labels_flatten = pd.Series(labels_flatten, dtype="category")
        self.class_names = labels_flatten.cat.categories
        labels_df = pd.DataFrame(labels_flatten.cat.codes,
                                 dtype=np.int8)  # int8-32 gives an error when using nn.CrossEntropyLoss

        lengths = df.applymap(
            lambda x: len(x)).values  # (num_samples, num_dimensions) array containing the length of each series
        horiz_diffs = np.abs(lengths - np.expand_dims(lengths[:, 0], -1))

        # most general check: len(np.unique(lengths.values)) > 1:  # returns array of unique lengths of sequences
        if np.sum(horiz_diffs) > 0:  # if any row (sample) has varying length across dimensions
            logger.warning(
                "Not all time series dimensions have same length - will attempt to fix by subsampling first dimension...")
            df = df.applymap(subsample)  # TODO: this addresses a very specific case (PPGDalia)

        if self.config['subsample_factor']:
            df = df.applymap(lambda x: subsample(x, limit=0, factor=self.config['subsample_factor']))

        lengths = df.applymap(lambda x: len(x)).values
        vert_diffs = np.abs(lengths - np.expand_dims(lengths[0, :], 0))
        if np.sum(vert_diffs) > 0:  # if any column (dimension) has varying length across samples
            self.max_seq_len = int(np.max(lengths[:, 0]))
            logger.warning("Not all samples have same length: maximum length set to {}".format(self.max_seq_len))
        else:
            self.max_seq_len = lengths[0, 0]

        # First create a (seq_len, feat_dim) dataframe for each sample, indexed by a single integer ("ID" of the sample)
        # Then concatenate into a (num_samples * seq_len, feat_dim) dataframe, with multiple rows corresponding to the
        # sample index (i.e. the same scheme as all datasets in this project)
        df = pd.concat((pd.DataFrame({col: df.loc[row, col] for col in df.columns}).reset_index(drop=True).set_index(
            pd.Series(lengths[row, 0] * [row])) for row in range(df.shape[0])), axis=0)

        # Replace NaN values
        grp = df.groupby(by=df.index)
        df = grp.transform(interpolate_missing)

        return df, labels_df


data_factory = {'weld': WeldData,
                'tsra': TSRegressionArchive,
                'pmu': PMUData,
                'mdd_data': MddDataset,
                'adhd_data': ADHDDataset,
                'abide_data': ABIDEDataset,
                'hc_data': HCDataset}
