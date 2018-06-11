import bisect
import logging
import time

import numpy as np
import pandas as pd
import scipy.optimize as optimize
import scipy.stats as stats

import exrex

LOGGER = logging.getLogger(__name__)

ARGS_SEP = '@'
COV_SEP = '*'
RAW_EXT = '.raw.csv'
SYNTH_EXT = '.synth.csv'
TRANS_EXT = '.trans.csv'


class SDVException(Exception):
    pass


def add_noise(cov):
    '''Add noise to the covariance matrix by dividing all of
       the off-diagonal elements by 2.0. This means that
       they are less dependent of each other

    :param cov: the covariance matrix
    :type cov: array

    :returns: ndarray
    '''
    diagonal = np.diag(cov)
    cov = np.divide(cov, 2.0)
    np.fill_diagonal(cov, diagonal)
    return cov


def get_date_converter(col, missing, meta):
    '''Returns a converter that takes in an integer representing ms
       and turns it into a string date

    :param col: name of column
    :type col: str
    :param missing: true if column has NULL values
    :type missing: bool
    :param meta: type of column values
    :type meta: str

    :returns: function
    '''

    def safe_date(x):
        if missing and x['?' + col] == 0:
            return np.nan

        t = x[col]
        tmp = time.gmtime(float(t) / 1e9)
        return time.strftime(meta, tmp)

    return safe_date


def get_number_converter(col, missing, meta):
    '''Returns a converter that takes in a value and turns it into an
       integer, if necessary

    :param col: name of column
    :type col: str
    :param missing: true if column has NULL values
    :type missing: bool
    :param meta: type of column values
    :type meta: str

    :returns: function
    '''

    def safe_round(x):
        if missing and x['?' + col] == 0:
            return np.nan
        if meta == 'integer':
            return int(round(x[col]))
        return x[col]

    return safe_round


def get_many(ct, regex, unique_set=None):
    '''Synthesizing many new values based on the regex

    :param ct: length of the dataframe
    :type ct: int
    :param regex: type of column values
    :type regex: str

    :returns: list
    '''
    out = []

    for i in range(ct):
        val = exrex.getone(regex)
        if unique_set is not None:
            while val in unique_set:
                val = exrex.getone(regex)
        out.append(val)

    return out


def get_ll(X, covariance, cdfs, check):
    '''Given a vector X, covariance matrix, and cdfs for each element,
       return the log likelihood of X in that distribution

    :param X: a vector
    :type X: ndarray
    :param covariance: a covariance matrix
    :type covariance:
    :param cdfs: cdfs
    :type cdfs: list<Distribution>
    :param check: each check var represents whether or not to check for noise
    :type check: list<bool>

    '''

    try:
        X_cop = [get_normalize_fn(cdf, c)(x) for cdf, x, c
                 in zip(cdfs, X, check)]
    except Exception:
        return 0

    # if any elment is + or - infinity, it means that it is out of bounds for
    # the truncated distribution, so the log likelihood is 0
    if not np.isfinite(X_cop).all():
        LOGGER.warning('Infinite value found', X_cop)
        return 0.0

    try:
        return -stats.multivariate_normal.logpdf(
            X_cop, mean=[0] * len(covariance), cov=covariance)
    except ValueError:
        return -1.0
    except np.linalg.LinAlgError:
        return -1.0


def get_normalize_fn(cdf, check=False):
    '''Normalizing should be: \Phi^-1(F(x)) but because F(x) is sometimes 0 or 1,
       we fudge the extremeties a little.

    :arg cdf: cdf
    :type cdf: Distribution
    :arg check: whether or not to check for noise
    :type cdf: bool
    '''
    def normalize(x):
        stat = stats.norm.ppf(cdf(x, check))
        if np.isnan(stat):
            return 0

        if stat == float('Inf') or stat == float('-Inf'):
            LOGGER.debug('INF', x, cdf(x), stats.norm.ppf(cdf(x, check)))

        # FIXME this is probably unnecessary now
        if stat == float('Inf') and round:
            stat = 5
        elif stat == float('-Inf') and round:
            stat = -5
        return stat

    return normalize


def make_covariance_matrix(dim, triu_vals):
    '''Make a symmetric covariance matrix of shape (dim x dim)
       given an array of values that belong to the upper triangle.
       For example, if dim=3 and triu_vals=[1, 2, 3, 4, 5, 6] then
       the covariance is:

       [[1, 2, 3]
        [2, 4, 5]
        [3, 5, 6]]

    :param dim: matrix has shape (dim x dim)
    :type dim: int
    :param triu_vals: list of vals to make the covariance matrix from (see summary)
    :type triu_vals: list<int>

    :returns: ndarray -- symmetric covariance matrix
    '''
    covar = np.zeros((dim, dim))
    covar[np.triu_indices(dim)] = np.array(triu_vals)
    return covar + covar.T - np.diag(covar.diagonal())


class Distribution(object):

    def __init__(self, column=None, summary=None, categorical=False):
        '''Represents the distribution of a single field in a table.
           Either column or data must be provided.

        :param column: The raw values. Use these to infer the type of distribution
                     and its parameters.
        :type column: pandas.DataFrame
        :param summary: A dictionary with the distrib name and the distrib args.
                      If this is provided, there is no need to infer based on
                      the actual data
        :type summary: dict
        :raises: Exception -- 'Distribution expects either column or summary'
        '''

        # name of distribution
        self.name = None
        # if categorical, name of categories in increasing order of frequency
        self.cats = None
        # the parameters for the distribution
        self.args = None

        if categorical:
            self.name = 'categorical'

        if summary is None and column is not None:
            self._infer_distribution(column)
        elif summary is not None:
            if summary['values'] is None:
                self.name = 'kde'
                summary['values'] = self.estimate_args(column)
            self._recover_distribution(summary)
        else:
            raise Exception('Distribution expects either column or summary')

    def _recover_distribution(self, summary):
        '''Recover the distribution given the appropriate summary params.

        :param summary: the summary parameters
        :type summary:
           {
                'name': <name of distribution>,
                'values': <values of distribution args>,
                'cats': <if categorical, names of category headers in same order>
           }'''
        self.name = summary['name']

        if summary['name'] == 'categorical':
            self.cats = summary['cats']
        self.args = summary['values']
        self.cdf = self.get_cdf(self.args)
        self.ppf = self.get_ppf(self.args)

    def _infer_distribution(self, column):
        '''Infer the distribution given a column of actual datapoints.'''

        try:
            isnan = np.isnan(np.array(column)).all()
        except Exception:
            isnan = False

        if isnan:
            self.name = 'NaN'
            self.args = []

            def cdf(x, check=False):
                return np.random.rand(1)[0]
            self.cdf = cdf

            def ppf(u, check=False):
                return 'NaN'
            self.ppf = ppf
            return

        if self.name == 'categorical' or len(set(column)) == 1:
            self.name = 'categorical'
            self._set_categorical(column)
        elif self.name == 'kde':
            self._set_kde(column)
        else:
            self._find_and_set(column)

        # distributions cdf and inverse cdf functions
        self.cdf = self.get_cdf(self.args)
        self.ppf = self.get_ppf(self.args)

    def set_args(self, args):
        self.args = args
        self.cdf = self.get_cdf(args)
        self.ppf = self.get_ppf(args)

    def get_summary(self):
        '''Returns all the data necessary to recreate this object later.'''
        obj = {
            'name': self.name,
            'values': self.args,
        }
        if self.name == 'categorical':
            obj['cats'] = self.cats

        return obj

    def estimate_args(self, data):
        if self.name == 'categorical':
            args = []

            for category in self.cats:
                p = len([True for i in data if i == category or
                         str(i) == category]) / float(len(data))
                args.append(p)
            return args

        data = np.array(data)
        data = data[~np.isnan(data)]

        # all the values were nan, means that there will be another binary
        # column taking care of missing data, so these args can be nan
        if len(data) == 0:
            return self.args

        lower = np.min(data)
        upper = np.max(data)

        mu = np.mean(data)
        sigma = np.var(data)**0.5

        if self.name == 'norm':
            return ((lower - mu) / sigma, (upper - mu) / sigma, mu, sigma)

        elif self.name == 'uniform':
            return (lower, upper - lower)
        elif self.name == 'kde':
            return stats.gaussian_kde(data)

    def fix_args(self, args):
        '''Fixes the args so that they are valid for the distribution this
           is supposed to represent.

        :raises: Exception -- Unsupported distribution:

        '''

        # all params of categorical distribution must be between 0 and 1
        if self.name == 'categorical':
            # zero out the negatives
            args = [max(0.0, arg) for arg in args]

            # N-1 degrees of freedom, so add the last one
            if len(args) == len(self.args) - 1:
                leftover = max(0.0, 1 - sum(args))
                args.append(leftover)

            # normalize
            sum_args = sum(args)
            args = [arg / sum_args for arg in args]

        # uniform distribution is between args[0] and args[0]+args[1]
        # which means that args[1] must be greater than or equal to 0
        elif self.name == 'uniform':
            if args[1] <= 0:
                args[1] = self.args[1]

        # normal distribution has mean args[0] and variance args[1]
        # which means that args[1] must be greater than 0
        elif self.name == 'norm':
            if args[1] <= 0:
                args[1] = self.args[1]
        # kernel density estimation has kernel type args[0] and bandwidth args[1]
        # which means that args[1] must be greater than 0
        elif self.name == 'kde':
            if args[0] is None:
                args[0] = self.args[0]

        else:
            raise Exception('Unsupported distribution: ' + str(self.name))

        return args

    def get_cdf(self, args):
        if self.name == 'NaN':
            def cdf(x, care=False):
                return np.random.rand(1)[0]
            return cdf

        # FIXME rewrite
        if self.name == 'norm':
            mu = args[2]
            sigma = args[3]
            low = (args[0] * sigma) + mu
            high = (args[1] * sigma) + mu

            # FIXME is there a better way to fix for noise?
            tolerance = (high - low) / 1000.0

            def cdf(x, care=True):
                if (x < low - tolerance or x > high + tolerance) and care:
                    raise Exception('Input ' + str(x) + ' is not in bounds: ' +
                                    str(low) + ' to ' + str(high))
                v = stats.truncnorm.cdf(x, *args)
                if v == 0.0:
                    return np.finfo(type(v)).eps
                elif v == 1.0:
                    return 1 - np.finfo(type(v)).eps
                return v
            return cdf

        elif self.name == 'uniform':
            low = args[0]
            high = low + args[1]

            # FIXME better way to fix for noise?
            tolerance = (high - low) / 1000.0

            def cdf(x, care=True):
                if (x < low - tolerance or x > high + tolerance) and care:
                    raise Exception('Input ' + str(x) + ' is not in bounds: ' +
                                    str(low) + ' to ' + str(high))

                if high == low:
                    return np.random.rand(1)[0]

                v = stats.uniform.cdf(x, *args)

                if v == 0.0:
                    return np.finfo(type(v)).eps
                elif v == 1.0:
                    return 1 - np.finfo(type(v)).eps
                return v
            return cdf
        elif self.name == 'kde':
            # fix this
            low_bounds = -10000
            kde = self.args
            # h = args[0]
            # kernel = args[1]

            def cdf(x, u=0, care=True):
                return kde.integrate_box(low_bounds, x) - u
            return cdf

        else:
            running_tot = np.cumsum(args)

            def cdf(x, care=True):
                # noise the CDF of x by sampling from a gaussian distribution where
                # the mean is the center for that category and the std deviation
                # so that 99% of the samples values fall in the range (+- 3 sigma)
                if str(x) == 'nan':
                    x = 'nan'

                i = self.cats.index(x)
                high = running_tot[i]

                if i == 0:
                    low = 0
                else:
                    low = running_tot[i - 1]

                if high == low:
                    return np.random.rand(1)[0]

                mu = (high + low) / 2.0
                sigma = (high - low) / 6.0
                a = (low - mu) / sigma + np.finfo(float).eps
                b = (high - mu) / sigma - np.finfo(float).eps
                return stats.truncnorm.rvs(a, b, mu, sigma)

            return cdf

    def get_ppf(self, args):
        if self.name == 'NaN':
            def ppf(u):
                return 'NaN'
            return ppf

        if self.name == 'norm':
            def ppf(u):
                return stats.truncnorm.ppf(u, *args)
            return ppf

        elif self.name == 'uniform':
            def ppf(u):
                return stats.uniform.ppf(u, *args)
            return ppf
        elif self.name == 'kde':
            def ppf(u):
                x = optimize.brentq(self.cdf, -100.0, 100.0, args=(u))
                return x
            return ppf
        else:
            running_tot = np.cumsum(args)

            def ppf(u):
                i = bisect.bisect_left(running_tot, u)
                out = self.cats[i]

                if out == "nan":
                    return np.nan

                return out

            return ppf

    def _set_categorical(self, column):
        orig_length = float(len(column))

        num_nan = len([i for i in column if str(i) == 'nan'])

        unique = set(column)
        freq_dict = {}

        for val in unique:
            freq = len([True for i in column if i == val]) / orig_length
            if freq > 0.0:
                freq_dict[val] = freq

        if num_nan > 0:
            freq_dict['nan'] = num_nan / orig_length

        freqs = freq_dict.items()
        freqs = sorted(freqs, key=lambda i: i[1])

        self.cats = [i for (i, j) in freqs]
        self.args = [j for (i, j) in freqs]

        self.cdf = self.get_cdf(self.args)
        self.ppf = self.get_ppf(self.args)

    def _set_kde(self, column):
        self.args = stats.gaussian_kde(column)

    def _find_and_set(self, column):
        column = np.array(column)
        column = column[~pd.isnull(column)]

        lower = np.min(column)
        upper = np.max(column)
        mu = np.mean(column)
        sigma = np.var(column)**0.5

        args_truncnorm = ((lower - mu) / sigma, (upper - mu) / sigma, mu, sigma)
        _, p_truncnorm = stats.kstest(column, 'truncnorm', args_truncnorm)

        args_uniform = (lower, upper - lower)
        _, p_uniform = stats.kstest(column, 'uniform', args_uniform)

        if p_truncnorm > p_uniform:
            self.name = 'norm'
            self.args = args_truncnorm
        else:
            self.name = 'uniform'
            self.args = args_uniform


def generate_samples(covariance, ppfs, N, means=None):
    '''Use a Gaussian Copula along with the given quantile functions to generate
    N samples whose elements are appropriately correlated'''

    # http://stackoverflow.com/questions/27727762/scipy-generate-random-variables-with-correlations
    if len(covariance) == 0 and len(ppfs) == 0:
        return [[] for i in range(N)]

    if means is None:
        means = [0.0] * len(covariance)

    samples = np.random.multivariate_normal(means, covariance, N)
    samples = stats.norm.cdf(samples)

    def fn(vector, ppfs):
        return [ppf(v) for v, ppf in zip(vector, ppfs)]

    return [fn(i, ppfs) for i in samples]


def update(obs, obs_indices, covariance, cdfs, obs_care):
    '''Perform inference to update the covariance and the means based on the
    observed values. Returns the updated (normalized) mean and covariance matrix.

    :param obs: the observations made on original table
    :type obs: list
    :param obs_indices: the indices of those observations (column indices)
    :type obs_indicies: list<int>
    :param covariance: the full covariance matrix of table
    :type covariance: ndarray
    :param cdfs: list of all the cdfs for each column in the table
    :type cdfs: list<Distribution>

   '''

    # http://stats.stackexchange.com/questions/30588/deriving-the-conditional-distributions-of-a-multivariate-normal-distribution

    # Follow formula to break covariance matrix into quadrants and calculate
    # new covariance for just the unobserved
    unobs_indices = [i for i in range(len(cdfs)) if i not in obs_indices]
    sigma_11 = covariance[unobs_indices, :][:, unobs_indices]
    sigma_12 = covariance[unobs_indices, :][:, obs_indices]
    sigma_21 = covariance[obs_indices, :][:, unobs_indices]
    sigma_22 = covariance[obs_indices, :][:, obs_indices]

    new_sigma = sigma_11 - np.dot(np.dot(sigma_12, np.linalg.inv(sigma_22)),
                                  sigma_21)

    # Convert the observations into the appropriate quantiles
    obs_cdfs = [cdfs[i] for i in range(len(cdfs)) if i in obs_indices]
    # cdf of each
    converted_obs = [cdf(o, care=c) for o, cdf, c in zip(obs, obs_cdfs, obs_care)]
    converted_obs = stats.norm.ppf(converted_obs)  # inverse normal of that

    new_means = np.dot(np.dot(sigma_12, np.linalg.inv(sigma_22)),
                       converted_obs)

    return new_means, new_sigma


class NonVariable(object):

    def __init__(self, column, is_key=False, regex=None):
        '''Represents a column that is not a variable we are modeling. This
           is because the column is either a primary key or foreign key, or
           it is not relevant to the problem we're trying to solve.'''
        self.regex = regex
        self.is_key = is_key

        data = np.array(column.as_matrix()).flatten()
        if is_key:
            self.unique = set(data)
        else:
            self.unique = None

        self.conv = type(data[0])

    def generate_new(self):
        '''Create a new value that could belong in the column using the regex.'''

        newone = exrex.getone(self.regex)

        if self.unique is not None:
            while newone in self.unique:
                newone = exrex.getone(self.regex)

        try:
            data = self.conv(newone)
        except ValueError:
            data = np.nan

        if self.unique is not None:
            self.unique.add(data)
        return newone
