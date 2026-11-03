import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def get_statistics(df, step):
    """
    Print descriptive statistics (df.describe()) in blocks of columns.

    Printing in blocks keeps the output readable when the DataFrame
    has many columns.

    Parameters
    ----------
    df : pd.DataFrame
        Data to describe.
    step : int
        Number of columns shown in each block.
    """
    for i in range(0, len(df.columns.to_list()), step):
        print(f'Statistics for columns {i} to {i+step}')
        print(df.iloc[:, i:i+step].describe())


def get_data_proportion(df):
    """
    Print the proportion of each distinct value in every column.

    Missing values are counted as their own category (dropna=False),
    and the values are sorted for easier reading.

    Parameters
    ----------
    df : pd.DataFrame
        Data to inspect.
    """
    print("-"*150)
    print("Proportion of values in each column")
    for column in df.columns.to_list():
        print(f'Column {column}:\n {df[column].value_counts(dropna=False, normalize=True).sort_index()}')


def quantile_analysis(df):
    """
    Print the upper quantiles (90%, 95%, 99.5% and 99.9%) of every column.

    Useful to spot extreme values in the right tail of each variable.

    Parameters
    ----------
    df : pd.DataFrame
        Data to inspect (numeric columns).
    """
    print("-"*100)
    print("Quantile analysis by column\n")
    for column in df.columns:
        print(f'Column {column}:')
        print(df[column].quantile([0.9, 0.95, 0.995, 0.999]).to_string())


def defaulter_statistics(df):
    """
    Print descriptive statistics for the defaulters subset.

    Calls get_statistics with blocks of 5 columns.

    Parameters
    ----------
    df : pd.DataFrame
        Data already filtered to defaulters (Default == 1).
    """
    print("-"*100)
    print("Defaulter's statistics:")
    get_statistics(df, step=5)


def correlation_analysis(df):
    """
    Print and plot the Spearman correlation between all columns.

    Spearman is used because it captures monotonic relationships and is
    robust to outliers and skewed distributions.

    The function:
      1. prints the full correlation matrix;
      2. lists the pairs whose absolute correlation is greater than 0.4
         (strong positive or negative relationships);
      3. plots a heatmap of the lower triangle, labelled with column names.

    Parameters
    ----------
    df : pd.DataFrame
        Numeric data (the target can be included to see its correlations).
    """
    # Prints the Spearman's correlation of columns
    corr = df.corr(method='spearman')
    print("-"*150)
    print("Correlation analysis: (Spearman's method)")
    print(corr)

    # Identifying pairs with absolute correlation greater than 0.4
    # (abs() also catches strong negative correlations)
    print("-"*150)
    print("Pairs with absolute Spearman's correlation greater than 0.4:")
    for i in corr.columns:
        for j in corr.columns:
            if (i > j and abs(corr.loc[i, j]) > 0.4):
                print(f"Correlation between {i} and {j}: {corr.loc[i, j]:.2f}")

    # Mask for the heatmap: hides the upper triangle (it mirrors the lower one)
    mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)

    # Plots the heatmap, using the column names as axis labels
    plt.figure(figsize=(12, 10))
    plt.title("Heatmap for Spearman's correlation")
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', annot_kws={'size': 7},
                cmap='coolwarm', vmin=-1, vmax=1)
    plt.tight_layout()


def bin_table(df, column, bins, labels, target='Default'):
    """
    Build a table, by bin, with volume, default rate, WoE and IV of a variable.

    The variable is cut into ranges with pd.cut (include_lowest=True, so the
    first bin includes its lower limit). Missing values go to their own
    'Missing' bin. Values outside the bins are left out, with a warning.

    WoE convention: WoE = ln(% goods / % bads), so a negative WoE means a
    riskier bin. To avoid ln(0) when a bin has no goods or no bads, 0.5 is
    added to the counts only inside the WoE/IV calculation.

    Parameters
    ----------
    df : pd.DataFrame
        Data with the variable and the target.
    column : str
        Name of the variable to analyse.
    bins : list
        Bin edges passed to pd.cut (e.g. [-1, 0, 1, 2, float('inf')]).
    labels : list
        One label per bin (len(labels) == len(bins) - 1).
    target : str, default 'Default'
        Name of the binary target column (1 = default).

    Returns
    -------
    pd.DataFrame
        One row per bin with: n_clients, n_default, default_rate,
        pct_good, pct_bad, woe, iv. The variable's total IV is
        table['iv'].sum().
    """
    binned = pd.cut(df[column], bins=bins, labels=labels, include_lowest=True)

    # Values that exist but fall outside the bins (pd.cut turns them into NaN)
    outside = binned.isna() & df[column].notna()
    if outside.sum() > 0:
        print(f"Warning: {outside.sum()} values of {column} are outside the bins "
              f"and were left out of the table.")

    # Missing values get their own bin
    missing = df[column].isna()
    if missing.sum() > 0:
        binned = binned.cat.add_categories('Missing')
        binned[missing] = 'Missing'

    # Counts by bin (observed=False keeps empty bins in the table)
    table = df.groupby(binned, observed=False)[target].agg(
        n_clients='count', n_default='sum')
    n_good = table['n_clients'] - table['n_default']

    table['default_rate'] = table['n_default'] / table['n_clients']

    # Share of all goods / all bads that falls in each bin (real values)
    table['pct_good'] = n_good / n_good.sum()
    table['pct_bad'] = table['n_default'] / table['n_default'].sum()

    # WoE and IV, with +0.5 in the counts to avoid ln(0)
    good_dist = (n_good + 0.5) / n_good.sum()
    bad_dist = (table['n_default'] + 0.5) / table['n_default'].sum()
    table['woe'] = np.log(good_dist / bad_dist)
    table['iv'] = (good_dist - bad_dist) * table['woe']

    # Empty bins carry no information: no WoE and zero IV
    empty = table['n_clients'] == 0
    table.loc[empty, 'woe'] = np.nan
    table.loc[empty, 'iv'] = 0.0

    return table


def plot_bin_table(table, column):
    """
    Plot a table from bin_table: client volume (bars) and default rate (line).

    The bars use the left axis (number of clients) and the line uses the
    right axis (default rate, in %). The dashed line is the overall default
    rate, to show which bins are above or below the average. The title
    shows the variable's total IV.

    Parameters
    ----------
    table : pd.DataFrame
        Output of bin_table.
    column : str
        Name of the variable (used in the title and x label).
    """
    x = table.index.astype(str)
    overall_rate = table['n_default'].sum() / table['n_clients'].sum()

    fig, ax1 = plt.subplots(figsize=(9, 4))

    # Bars: number of clients by bin
    ax1.bar(x, table['n_clients'], color='lightgray')
    ax1.set_ylabel('No. of clients')
    ax1.set_xlabel(column)
    ax1.tick_params(axis='x', rotation=45)

    # Line: default rate by bin, on a second axis
    ax2 = ax1.twinx()
    ax2.plot(x, table['default_rate'] * 100, marker='o', color='firebrick')
    ax2.axhline(overall_rate * 100, color='firebrick', linestyle='--', linewidth=1)
    ax2.set_ylabel('Default rate (%)')
    ax2.set_ylim(bottom=0)

    plt.title(f"{column}: IV = {table['iv'].sum():.3f}")
    fig.tight_layout()


def missing_values_statistics(df, target='Default'):
    """
    Compare the default rate of rows with and without missing values.

    Only columns that have missing values are analysed. For each one, the
    function prints the number and share of missing rows and the default
    rate in the missing and non-missing groups. A clear difference means
    that missingness itself carries information about risk.

    Parameters
    ----------
    df : pd.DataFrame
        Data with the target column (before any imputation).
    target : str, default 'Default'
        Name of the binary target column (1 = default).
    """
    print("-"*150)
    print("Statistical analysis of missing values by column")

    # Only the columns that actually have missing values
    columns = df.columns[df.isna().any()].to_list()

    if len(columns) == 0:
        print("No missing values found.")
        return

    for column in columns:
        mask = df[column].isna()

        print(f"\nColumn {column}:")
        print(f"  Missing rows: {mask.sum()} ({mask.mean()*100:.1f}%)")
        print(f"  Default rate (non-missing): {df.loc[~mask, target].mean()*100:.1f}%")
        print(f"  Default rate (missing):     {df.loc[mask, target].mean()*100:.1f}%")


def iv_summary(df, bins_dict, target='Default'):
    """
    Compute the total IV of several variables and rank them.

    Runs bin_table for each variable in bins_dict and adds the usual IV
    strength label:
        < 0.02       not useful
        0.02 - 0.10  weak
        0.10 - 0.30  medium
        0.30 - 0.50  strong
        > 0.50       suspicious (check for leakage or overly fine bins)

    Parameters
    ----------
    df : pd.DataFrame
        Data with the variables and the target.
    bins_dict : dict
        {column: {'bins': [...], 'labels': [...]}}, the same dictionary
        used for the default-rate plots.
    target : str, default 'Default'
        Name of the binary target column (1 = default).

    Returns
    -------
    pd.DataFrame
        One row per variable with its IV and strength, sorted by IV
        (highest first).
    """
    rows = []
    for column, spec in bins_dict.items():
        table = bin_table(df, column, spec['bins'], spec['labels'], target=target)
        rows.append({'variable': column, 'iv': table['iv'].sum()})

    summary = pd.DataFrame(rows).sort_values('iv', ascending=False)

    # Strength label based on the usual IV ranges
    summary['strength'] = pd.cut(
        summary['iv'],
        bins=[-float('inf'), 0.02, 0.1, 0.3, 0.5, float('inf')],
        labels=['not useful', 'weak', 'medium', 'strong', 'suspicious'])

    return summary.reset_index(drop=True)