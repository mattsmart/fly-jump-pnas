import csv
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys

from matplotlib.gridspec import GridSpec

from data_tools import build_dataframe_from_data, build_mergetrials_dataframe
from python.settings import DIR_DATA_EXPT

"""
Background: 
- The separate data files Anton gave Bob do not have the chamber ID, but do have the fly ID.
- Anton data has chamber ID, and fly ID column that differs slightly from Bob's.
- In Bob's output, the fly ID and raw data (jump string) should appear in final output fly-stability-days.csv. 

This script outputs revised data files with fly_ID and chamber_ID matching the original data files from Anton. 
"""


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

    # add a column called idx_fly_128 which is: idx_fly + 64 * (var-1)
    df_full_mergetrials['idx_fly_128'] = df_full_mergetrials['idx_fly'] + 64 * (df_full_mergetrials['var'] - 1)

    # Rearrange the columns in the DataFrame
    new_column_order = ['gene_bgr', 'age', 'idx_fly_128', 'idx_chamber', 'idx_fly', 'var', 'exp_name', 'date', 'flies_remaining']
    # Reorder the DataFrame columns
    df_full_mergetrials = df_full_mergetrials[new_column_order + [col for col in df_full_mergetrials.columns if col not in new_column_order]]
    # Assuming df_full_mergetrials is your DataFrame
    df_full_mergetrials = df_full_mergetrials.drop(columns=['chamber_group', 'ttc_can', 'ttc_can_1to10'])
    # Rename the columns to match the format of the data files used by Bob
    df_full_mergetrials = df_full_mergetrials.rename(columns={
        'gene_bgr': 'genotype',
        'age': 'day',
        'idx_fly_128': 'fly_id',
        'idx_fly': 'fly_id_64',
        'idx_chamber': 'chamber_id',
        'jumpdata_str': 'jumpdata',
    })

    # Assuming df_full_mergetrials is already defined and contains the necessary data
    # Define the genotypes and days
    genotypes = ['KK', 'GD']
    days = [7, 14, 21]

    # Loop through each combination of genotype and day - save to file
    for genotype in genotypes:
        for day in days:
            # Filter the dataframe
            df_filtered = df_full_mergetrials[
                (df_full_mergetrials['genotype'] == genotype) & (df_full_mergetrials['day'] == day)]

            # Define the output file name
            output_file = DIR_DATA_EXPT + os.sep + f"{genotype}_{day}days_with_details_unfiltered.csv"

            # Save to CSV
            df_filtered.to_csv(output_file, index=False)

    def local_filter_to_common_flies(filenames, output_filenames):
        """
        Filter data to include only flies that are present in all files.
        - mimics the function in fit_experimental_data.py
        """
        # Filter data to include only flies that are present in all files.
        fly_data = [pd.read_csv(file) for file in filenames]
        fly_ids = [set(df["fly_id"]) for df in fly_data]

        # Find the intersection of fly IDs present in all files
        common_flies = set.intersection(*fly_ids)

        # Filter each file for the common flies and write to new files
        for df, output_file in zip(fly_data, output_filenames):
            filtered_df = df[df["fly_id"].isin(common_flies)]
            filtered_df.to_csv(output_file, index=False)


    local_filter_to_common_flies(
        [DIR_DATA_EXPT + os.sep + f"KK_{day}days_with_details_unfiltered.csv" for day in days],
        [DIR_DATA_EXPT + os.sep + f"KK_{day}days_with_details_filtered.csv" for day in days]
    )
    local_filter_to_common_flies(
        [DIR_DATA_EXPT + os.sep + f"GD_{day}days_with_details_unfiltered.csv" for day in days],
        [DIR_DATA_EXPT + os.sep + f"GD_{day}days_with_details_filtered.csv" for day in days]
    )

    # Filter the data to include only flies that are present in all files
    '''
    # some of the flies disappear from day 7->14->21 so we need to remove the rows that are not there for all 3 days
    # for GD, identify the flies that are not present in all 3 days
    df_GD_7days = pd.read_csv(DIR_DATA_EXPT + os.sep + 'GD_7days_with_details.csv')
    df_GD_14days = pd.read_csv(DIR_DATA_EXPT + os.sep + 'GD_14days_with_details.csv')
    df_GD_21days = pd.read_csv(DIR_DATA_EXPT + os.sep + 'GD_21days_with_details.csv')
    # get a list of the flies not present in all 3 days, and print the list
    idx_fly_GD_7days = set(df_GD_7days['idx_fly_128'])
    idx_fly_GD_14days = set(df_GD_14days['idx_fly'])
    idx_fly_GD_21days = set(df_GD_21days['idx_fly_128'])
    idx_fly_GD_not_in_all_3_days = idx_fly_GD_7days ^ idx_fly_GD_14days ^ idx_fly_GD_21days
    common_flies = set.intersection(idx_fly_GD_7days, idx_fly_GD_14days, idx_fly_GD_21days)

    print('common_flies')
    print(common_flies)

    # repeat but for KK
    df_KK_7days = pd.read_csv(DIR_DATA_EXPT + os.sep + 'KK_7days_with_details.csv')
    df_KK_14days = pd.read_csv(DIR_DATA_EXPT + os.sep + 'KK_14days_with_details.csv')
    df_KK_21days = pd.read_csv(DIR_DATA_EXPT + os.sep + 'KK_21days_with_details.csv')
    # get a list of the flies not present in all 3 days, and print the list
    idx_fly_KK_7days = set(df_KK_7days['idx_fly'])
    idx_fly_KK_14days = set(df_KK_14days['idx_fly'])
    idx_fly_KK_21days = set(df_KK_21days['idx_fly'])
    idx_fly_KK_not_in_all_3_days = idx_fly_KK_7days ^ idx_fly_KK_14days ^ idx_fly_KK_21days
    common_flies = set.intersection(idx_fly_KK_7days, idx_fly_KK_14days, idx_fly_KK_21days)

    print('common_flies')
    print(common_flies)

    #output_file = DIR_DATA_EXPT + os.sep + f"{genotype}_{day}days_with_details_filtered.csv"
    '''