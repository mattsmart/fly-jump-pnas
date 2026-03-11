import csv
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys
from scipy.stats import fisher_exact
from matplotlib.gridspec import GridSpec

# Adds fly-jump to sys.path
ROOT_FLYJUMP = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_FLYJUMP)


DIR_INPUT = ROOT_FLYJUMP + os.sep + 'data' + os.sep + 'input_for_plotting'

chamber_grp_A = list([i for i in range(1,9)])  # 4 groups of 8
chamber_grp_B = list([i for i in range(9,17)])  # 4 groups of 8
chamber_grp_C = list([i for i in range(17,25)])  # 4 groups of 8
chamber_grp_D = list([i for i in range(25,33)])  # 4 groups of 8

chamber_idx_to_group = {i: ((i-1)//8) for i in range(1, 33)}


def wrapper_read_data_into_dict(fpath, data_version, verbose=False):
    """
    data_version
        'v1': from Anton Nov 2023
        'v2': from Anton Jan 2023
        'v3': from Anton April 2024 (dropbox label)
    """
    assert data_version in ['v1', 'v2', 'v3']

    if verbose:
        print('using read_data_into_dict_%s(...)' % data_version)

    if data_version == 'v1':
        outs = read_data_into_dict_v1(fpath)
    elif data_version == 'v2':
        outs = read_data_into_dict_v2(fpath)
    else:
        assert data_version == 'v3'
        outs = read_data_into_dict_v3(fpath)
    return outs


def read_data_into_dict_v1(fpath):
    """
    fpath
    Returns:
        data_dict = {'metadata': ...,
                     'arr': ...}
    """
    print('In read_data_into_dict_v1(...)')
    # TODO: make pandas friendly version of this, prob will be much shorter and faster
    data_dict = dict()  # this is what we will build and return
    data_row = 32  # TODO generalize
    data_col = 100  # TODO generalize
    with open(fpath) as datafile:
        header_row_count = None  # will be updated within file loop
        header_index = -1  # will be updated within file loop
        for idx, ln in enumerate(datafile):
            ln = ln.replace('\n', '')  # remove newlines everywhere (specifically end of each line)
            sep_by_space = ln.split(' ')
            # case 1: header lines
            # e.g. '#data_KK_day2, 1486mb, 13759, 2023-09-15, Var#2 KK control, 4 days old, habituation test 1'
            if len(sep_by_space) > 2:
                sep_by_comma = ln.split(', ')
                assert len(sep_by_comma) == 7
                label_0 = sep_by_comma[0]
                label_1 = sep_by_comma[1]
                label_2 = sep_by_comma[2]
                label_3 = sep_by_comma[3]
                label_4 = sep_by_comma[4]
                label_5 = sep_by_comma[5]
                label_6 = sep_by_comma[6]

                header_row_count = 0
                header_index += 1

                data_dict[header_index] = {
                    'metadata': {
                        'fullheader': ln,
                        'label_0': label_0,
                        'label_1': label_1,
                        'label_2': label_2,
                        'label_date': label_3,
                        'label_4': label_4,
                        'label_age': label_5,
                        'label_6': label_6,
                    },
                    'arr': []
                }

            # case 2: data lines
            # e.g. '03: 1010010001011010100111111110010100001111101100000100001010001111001100000000001001010110111110000111'
            else:
                assert len(sep_by_space) == 2
                header_row_count += 1
                assert len(sep_by_space) == 2
                assert sep_by_space[0][-1] == ':'
                datarow_int_str = sep_by_space[0][:-1]
                datarow_as_list = [int(a) for a in sep_by_space[1]]
                assert header_row_count == int(datarow_int_str)
                # now fill in the data row
                data_dict[header_index]['arr'].append(datarow_as_list)

        # post-process the data dict by
        #   1) turning the arr objects into numpy arrays from list of list
        #   2) validating that those arrays are all the same size, and returning the size (this part may be not necessary)
        # TODO handle this differently, very clunky
        num_headers_seen = header_index + 1
        unique_nrows = []
        unique_ncols = []
        for idx in range(num_headers_seen):
            data_listoflist = data_dict[idx]['arr']
            nrows = len(data_listoflist)
            ncols_list = [len(a) for a in data_listoflist]
            if nrows not in unique_nrows:
                unique_nrows.append(nrows)
            for ncols in ncols_list:
                if ncols not in unique_ncols:
                    unique_ncols.append(ncols)

            data_dict[idx]['arr'] = np.array(data_dict[idx]['arr'], dtype=int)

        # print(unique_nrows, unique_ncols)
        assert len(unique_nrows) == 1
        assert len(unique_ncols) == 1
        nrows_per_header = unique_nrows[0]
        nobs_per_expt = unique_ncols[0]
        return data_dict, num_headers_seen, nrows_per_header, nobs_per_expt


def read_data_into_dict_v2(fpath):
    """
    fpath
    Returns:
        data_dict = {'metadata': ...,
                     'arr': ...}
    """
    # TODO: make pandas friendly version of this, prob will be much shorter and faster
    data_dict = dict()  # this is what we will build and return

    with open(fpath) as datafile:
        for idx, ln in enumerate(datafile):
            ln = ln.replace('\n', '')  # remove newlines everywhere (specifically end of each line)
            ln = ln.replace(' ', '')  # remove all spaces
            sep_by_comma = ln.split(',')
            # Case 1: header lines
            # e.g. '#data_KK_day2, 1486mb, 13759, 2023-09-15, Var#2 KK control, 4 days old, habituation test 1'
            if len(sep_by_comma) > 3:
                assert len(sep_by_comma) in [5, 6]  # there are two cases: reactivity (5 items) or responses (6 items)

                # Case 1.1: headers in "responses_foo.txt"
                #   e.g. gene_bgr=KK, age=3, var=1, date=2023-09-07, testnum=2, exp_name=1427mb
                if len(sep_by_comma) == 6:
                    exp_name = sep_by_comma[5].split('=')[
                        1]  # should be unique row identifier for dataframe (i.e. unique dict key)
                    data_dict[exp_name] = {
                        'metadata': {
                            'exp_name': exp_name,
                            'gene_bgr': sep_by_comma[0].split('=')[1],
                            'age': int(sep_by_comma[1].split('=')[1]),
                            'var': int(sep_by_comma[2].split('=')[1]),
                            'date': sep_by_comma[3].split('=')[1],
                            'testnum': int(sep_by_comma[4].split('=')[1]),
                            'interval_sec': 1.0,
                        },
                        'arr_jumps': [],
                        'arr_int_fly': [],
                        'arr_int_chamber': []
                    }
                else:
                    # Case 1.2: headers in "reactivity_foo.txt"
                    #   e.g. gene_bgr=KK, age=3, var=1, date=2023-09-07, exp_name=1431mbSRA
                    exp_name = sep_by_comma[4].split('=')[
                        1]  # should be unique row identifier for dataframe (i.e. unique dict key)
                    data_dict[exp_name] = {
                        'metadata': {
                            'exp_name': exp_name,
                            'gene_bgr': sep_by_comma[0].split('=')[1],
                            'age': int(sep_by_comma[1].split('=')[1]),
                            'var': int(sep_by_comma[2].split('=')[1]),
                            'date': sep_by_comma[3].split('=')[1],
                            'testnum': 99,  # instead of 4 (1,2,3 responses then 4th is reactivity - denote by 99)
                            'interval_sec': 5.0,
                        },
                        'arr_jumps': [],
                        'arr_int_fly': [],
                        'arr_int_chamber': []
                    }

            # Case 2: up to 32 data lines corresponding to an experiment header
            # e.g. ' 1, 3: 1010010001011010100111111110010100001111101100000100001010001111001100000000001001010110111110000111'
            # e.g. '11,12: 1010010001011010100111111110010100001111101100000100001010001111001100000000001001010110111110000111'
            else:
                toplevel_key = exp_name  # assumes text file is being read in order
                assert len(sep_by_comma) == 2

                int_chamber, int_fly = [int(a) for a in ln.split(':')[0].split(',')]
                assert 1 <= int_chamber <= 32
                assert 1 <= int_fly <= 64

                datarow_int_str = ln.split(':')[1]
                datarow_as_list = [int(a) for a in datarow_int_str]

                # now fill in the data row
                data_dict[toplevel_key]['arr_jumps'].append(datarow_int_str)
                data_dict[toplevel_key]['arr_int_chamber'].append(int_chamber)
                data_dict[toplevel_key]['arr_int_fly'].append(int_fly)

        # post-process the data dict by counting the number of rows in each experiment, adding to metadata
        for k in data_dict.keys():
            subdict = data_dict[k]
            flies_remaining = len(subdict['arr_int_fly'])  # a number between 0 and 32
            data_dict[k]['metadata']['flies_remaining'] = flies_remaining

        return data_dict


def read_data_into_dict_v3(fpath):
    """
    fpath
    Returns:
        data_dict = {'metadata': ...,
                     'arr': ...}
    """
    # TODO: make pandas friendly version of this, prob will be much shorter and faster
    data_dict = dict()  # this is what we will build and return

    with open(fpath) as datafile:
        for idx, ln in enumerate(datafile):
            ln = ln.replace('\n', '')  # remove newlines everywhere (specifically end of each line)
            if ln == '#':
                continue

            # Case 1: header lines
            # e.g. genbgr=GD age=14 var=1 date=2024-03-12 exp_name=1921mbSRA
            elif ln[:6] == 'genbgr':
                sep_by_space = ln.split(' ')

                assert len(sep_by_space) in [5, 6]  # there are two cases: reactivity (5 items) or responses (6 items)

                # Case 1.1: headers in "responses_foo.txt"
                #   e.g. gene_bgr=KK, age=3, var=1, date=2023-09-07, testnum=2, exp_name=1427mb
                if len(sep_by_space) == 6:
                    exp_name = sep_by_space[5].split('=')[
                        1]  # should be unique row identifier for dataframe (i.e. unique dict key)
                    data_dict[exp_name] = {
                        'metadata': {
                            'exp_name': exp_name,
                            'gene_bgr': sep_by_space[0].split('=')[1],
                            'age': int(sep_by_space[1].split('=')[1]),
                            'var': int(sep_by_space[2].split('=')[1]),
                            'date': sep_by_space[3].split('=')[1],
                            'testnum': int(sep_by_space[4].split('=')[1]),
                            'interval_sec': 1.0,
                        },
                        'arr_jumps': [],
                        'arr_int_fly': [],
                        'arr_int_chamber': []
                    }
                else:
                    # Case 1.2: headers in "reactivity_foo.txt"
                    #   e.g. gene_bgr=KK, age=3, var=1, date=2023-09-07, exp_name=1431mbSRA
                    exp_name = sep_by_space[4].split('=')[
                        1]  # should be unique row identifier for dataframe (i.e. unique dict key)
                    data_dict[exp_name] = {
                        'metadata': {
                            'exp_name': exp_name,
                            'gene_bgr': sep_by_space[0].split('=')[1],
                            'age': int(sep_by_space[1].split('=')[1]),
                            'var': int(sep_by_space[2].split('=')[1]),
                            'date': sep_by_space[3].split('=')[1],
                            'testnum': 99,  # instead of 6 (1,2,3,4,5 responses then 4th is reactivity - denote by 99)
                            'interval_sec': 5.0,
                        },
                        'arr_jumps': [],
                        'arr_int_fly': [],
                        'arr_int_chamber': []
                    }

            # Case 2: up to 32 data lines corresponding to an experiment header
            # e.g. ' 1, 3:1010010001011010100111111110010100001111101100000100001010001111001100000000001001010110111110000111'
            # e.g. '11,12:1010010001011010100111111110010100001111101100000100001010001111001100000000001001010110111110000111'
            else:
                ln = ln.replace(' ', '')  # remove all spaces

                toplevel_key = exp_name  # assumes text file is being read in order

                sep_by_comma = ln.split(',')
                assert len(sep_by_comma) == 2

                int_chamber, int_fly = [int(a) for a in ln.split(':')[0].split(',')]
                assert 1 <= int_chamber <= 32
                assert 1 <= int_fly <= 64

                datarow_int_str = ln.split(':')[1]
                #datarow_as_list = [int(a) for a in datarow_int_str]

                # now fill in the data row
                data_dict[toplevel_key]['arr_jumps'].append(datarow_int_str)
                data_dict[toplevel_key]['arr_int_chamber'].append(int_chamber)
                data_dict[toplevel_key]['arr_int_fly'].append(int_fly)

        # post-process the data dict by counting the number of rows in each experiment, adding to metadata
        for k in data_dict.keys():
            subdict = data_dict[k]
            flies_remaining = len(subdict['arr_int_fly'])  # a number between 0 and 32
            data_dict[k]['metadata']['flies_remaining'] = flies_remaining

        return data_dict


def get_TTC_canonical(jumpdata_str, num_non_jump=5):
    """
    returns no-jump criterion (trials to criterion, TTC)
    - e.g. for '1100000' return 2 (index of first 0 in the sequence)
    """
    nn = len(jumpdata_str)

    target_non_jump_sequence = '0' * num_non_jump
    target_loc = jumpdata_str.find(target_non_jump_sequence)

    if target_loc == -1:
        ttc = nn
    else:
        ttc = target_loc
    return ttc


def habituation_time_fisherexact(jumpdata_vector):
    """
    Identify the time point with the maximum difference in response frequency before and after it using Fisher's exact test.
    Input: response_vector: A binary vector (length 200) where 1 = response, 0 = no response.
    Returns: The time step with the most significant drop in response probability.
    """
    best_p_value = 1  # initialize with the highest possible p-value
    ttc = 0

    for t in range(1, len(jumpdata_vector)):
        before = jumpdata_vector[:t]
        after = jumpdata_vector[t:]

        # Construct contingency table
        table = [[sum(before), len(before) - sum(before)],  # responses vs. no responses before t
                 [sum(after), len(after) - sum(after)]]  # responses vs. no responses after t

        # Apply Fisher's exact test
        _, p_value = fisher_exact(table, alternative='two-sided')

        # Update best step if this test yields a lower p-value (stronger difference)
        if p_value < best_p_value:
            best_p_value = p_value
            ttc = t

    return ttc


def build_main_dataframe(data_dict, df=None, verbose=False):
    """
    Function to assemble giant pandas DataFrame from dictionary read from one textfile
    """
    columns_header = ['exp_name', 'gene_bgr', 'age', 'date', 'var', 'testnum', 'interval_sec', 'flies_remaining']
    columns_full = columns_header + ['idx_fly', 'idx_chamber', 'jumpdata_str']
    if df is None:
        df = pd.DataFrame(columns=columns_full)
    else:
        assert all(item in df.columns for item in columns_full)

    for key, vals in data_dict.items():
        if verbose:
            print('adding key=%s...' % key)
        row_as_dict_template = {r: vals['metadata'][r] for r in columns_header}
        row_as_dict_template['exp_name'] = key

        num_rows = vals['metadata']['flies_remaining']
        list_of_row_dicts = [row_as_dict_template.copy() for _ in range(num_rows)]

        for i in range(num_rows):
            list_of_row_dicts[i]['idx_fly'] = vals['arr_int_fly'][i]
            list_of_row_dicts[i]['idx_chamber'] = vals['arr_int_chamber'][i]
            list_of_row_dicts[i]['jumpdata_str'] = vals['arr_jumps'][i]

        df_expt = pd.DataFrame.from_records(list_of_row_dicts)
        df = pd.concat([df, df_expt])

    # set datatypes
    for cname in ['age', 'var', 'testnum', 'flies_remaining', 'idx_fly', 'idx_chamber']:
        df[cname] = df[cname].astype(int)
    # for cname in ['exp_name', 'gene_bgr', 'date']:
    #    df[cname] = df[cname].astype(str)

    # additional post-processing:
    # - add column for chamber group (the 32 chambers are arranged into 4 groups of 8)
    df['chamber_group'] = df['idx_chamber'].map(lambda x: chamber_idx_to_group[x])

    # - add column for TTC canonical
    df['ttc_can'] = df['jumpdata_str'].map(lambda x: get_TTC_canonical(x))
    df['ttc_can_1to10'] = df['jumpdata_str'].map(lambda x: [get_TTC_canonical(x, num_non_jump=a) for a in range(1, 11)])

    # set unique row indices via column Index for downstream analyses
    df.reset_index(inplace=True, drop=True)

    return df


def build_dataframe_from_data(data_version='v3', verbose=False):
    """
    Load all datafiles into single pd DataFrame
    """

    DATA_FILES = dict(
        v2=[DIR_INPUT + os.sep + '2024-01-29_reactivity_GD.txt',
               DIR_INPUT + os.sep + '2024-01-29_reactivity_KK.txt',
               DIR_INPUT + os.sep + '2024-01-29_responses_GD.txt',
               DIR_INPUT + os.sep + '2024-01-29_responses_KK.txt'],

        v3=[DIR_INPUT + os.sep + '2024-04-01_reactivity.txt',
            DIR_INPUT + os.sep + '2024-04-01_responses.txt'],
    )

    data_files_list = DATA_FILES[data_version]

    # TODO we could just concatenate into one big df over all versions, and add version as a key?
    df_full = None
    for fpath in data_files_list:
        dict_one_file = wrapper_read_data_into_dict(fpath, data_version, verbose=verbose)
        df_full = build_main_dataframe(dict_one_file, df=df_full, verbose=verbose)

    return df_full


def build_mergetrials_dataframe(df_full_unmerged, data_version='v3'):
    """
    Function to create merge350 variant of DataFrame output <-- build_dataframe_from_data()
    - v2 is 350 (3x100 + 50)
    - v3 is 1050 (5x200 + 50)

    Notes:
        - 'ttc_can' goes from an int to a tuple of int in this representation (called 'ttc_can' still)
    """
    version_to_expected_length = dict(v2=350, v3=1050)
    expected_length = version_to_expected_length[data_version]
    if data_version == 'v2':
        version_to_exp_options_list = lambda exp_int: [exp_int - 2, exp_int - 1, exp_int]
    else:
        assert data_version == 'v3'
        version_to_exp_options_list = lambda exp_int: [exp_int - 4, exp_int - 3, exp_int - 2, exp_int - 1, exp_int]

    all_exp_name_groups = sorted(set(df_full_unmerged['exp_name'][df_full_unmerged['interval_sec'].isin([5.0])]))

    columns_header = ['exp_name', 'gene_bgr', 'age', 'date', 'var', 'flies_remaining']
    columns_full = columns_header + ['idx_fly', 'idx_chamber', 'jumpdata_str']

    # jumpdata_str_samples = []
    # col_gene_bgr = []

    df_mergetrials = pd.DataFrame(columns=columns_full)
    df_full_unmerged = df_full_unmerged.sort_values(by=['exp_name'])  # need this line to properly concatenate jump strings below

    for exp_name_SRA in all_exp_name_groups:
        exp_int = int(exp_name_SRA.split('mb')[0])

        merging_exp_options = [str(a) + 'mb' for a in
                               version_to_exp_options_list(exp_int)
                               ] + [exp_name_SRA]

        df_subset = df_full_unmerged[df_full_unmerged['exp_name'].isin(merging_exp_options)]

        # define how to aggregate various fields
        merge_ttc_func = lambda x: tuple(x)

        agg_functions = {'idx_fly': 'first',
                         'idx_chamber': 'first',
                         'exp_name': 'first',
                         'gene_bgr': 'first',
                         'date': 'first',
                         'var': 'first',
                         'flies_remaining': 'first',
                         'age': 'first',
                         'chamber_group': 'first',
                         'jumpdata_str': 'sum',
                         'ttc_can': merge_ttc_func,
                         'ttc_can_1to10': merge_ttc_func
                         }

        # create small merged df (based on aggregating data for a given fly over the K + 1 (HAB then SRA) experiments)
        df_subset_merge = df_subset.groupby(df_subset['idx_fly']).aggregate(agg_functions)

        # append to larger growing df over the loop
        df_mergetrials = pd.concat([df_mergetrials, df_subset_merge])

    # set datatypes
    for cname in ['age', 'var', 'flies_remaining', 'idx_fly', 'idx_chamber', 'chamber_group']:
        df_mergetrials[cname] = df_mergetrials[cname].astype(int)
    # for cname in ['exp_name', 'gene_bgr', 'date']:
    #    df[cname] = df[cname].astype(str)

    # additional post-processing:
    # - drop rows where jumpdata_str len != expected_length
    df_mergetrials = df_mergetrials[df_mergetrials['jumpdata_str'].str.len() == expected_length]

    # set unique row indices via column Index for downstream analyses
    df_mergetrials.reset_index(inplace=True, drop=True)

    return df_mergetrials


def filter_df_by_filterdict(df, dict_of_filters):
    """
    Input takes the form column key -> list of acceptable values (applies AND across all, but OR within each)
    Sample usage:
        dict_of_filters = {'exp_name': ['1422mb', '1425mb'],
                        'idx_chamber': [1,2,3]}
        df_filtered = filter_df_by_dict_of_lists(dict_of_filters)
        print(df_filtered)
    """
    df_selection = df
    for k, vals in dict_of_filters.items():
        df_selection = df_selection[df_selection[k].isin(vals)]
    return df_selection


def df_to_arr_jumps(df, jump_col='jumpdata_str'):
    jumps_series = df[jump_col].apply(lambda x: [int(a) for a in x])
    arr_jumps = np.array(jumps_series.tolist())
    '''
    arr_age =  df['age'].to_numpy()
    arr_testnum = df['test_num'].to_numpy()
    arr_testnum = df['test_num'].to_numpy()
    arr_testnum = df['test_num'].to_numpy()'''
    return arr_jumps


if __name__ == '__main__':
    data_version = 'v3'  # try either v2 (given Jan 2024) or v3: (given April 2024)

    if data_version == 'v2':
        heatmap_title = 'Merged jump data (%s - each row is 3 HAB + 1 SRA experiments)' % data_version
        heatmap_aspect = 0.1
    else:
        assert data_version == 'v3'
        heatmap_title = 'Merged jump data (%s - each row is 5 HAB + 1 SRA experiments)' % data_version
        heatmap_aspect = 0.75

    df_full = build_dataframe_from_data(data_version=data_version)
    print(df_full)
    print(df_full.index.is_unique)

    df_full_mergetrials = build_mergetrials_dataframe(df_full, data_version=data_version)  # v2 is 350; v3 is 1050
    print(df_full_mergetrials)

    plt.figure(figsize=(8, 7))
    plt.imshow(df_to_arr_jumps(df_full_mergetrials), aspect=heatmap_aspect, interpolation='None')
    plt.title(heatmap_title, fontsize=12)
    plt.show()

    plt.hist(df_full_mergetrials['ttc_can'])
    plt.title('histogram of ttc_can tuples')
    plt.show()

    # plot where the x axis is time index from 0 to 200 and y axis is cumulative number of jumps
    # (for each fly, each chamber, each experiment)
    # - this will be a line plot with many lines
    # - each line will be a different color/fly
    cumulative_jumps_exp1 = df_to_arr_jumps(df_full_mergetrials)[:, 0:200].cumsum(axis=1)
    cumulative_jumps_exp1_mean = cumulative_jumps_exp1.mean(axis=0)
    plt.figure(figsize=(8, 7))
    plt.plot(cumulative_jumps_exp1.T, alpha=0.5)
    plt.plot(cumulative_jumps_exp1_mean, color='black', linewidth=2, label='mean')
    plt.title('cumulative jumps per fly per chamber per experiment')
    plt.show()
