import pandas as pd
import pathlib as plib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def get_statistics(df, step):
     for i in range(0, len(df.columns.to_list()), step):
          print(f'Statistics for columns {i} to {i+step}')
          print(df.iloc[:, i:i+step].describe())

#Verifica a proporcao de valores por coluna
def get_data_proportion(df):
    print("-"*150)
    print("Proportion of values in each column")
    for column in df.columns.to_list():
         print(f'Column {column}:\n {df[column].value_counts(dropna=False, normalize=True).sort_index()}')

def bivariate_analysis(df):
    print("-"*20)
    print("Bivariate analysis")
    for column in df.columns:
         print (df.groupby(column).size())

#bivariate_analysis(defaulter)

def quantile_analysis(df):
     print("-"*100)
     print("Quantile analysis by column\n")
     for column in df.columns:
          print(f'Column {column}:')
          print(df[column].quantile([0.9, 0.95, 0.995, 0.999]).to_string())

def defaulter_statistics(df):
     print("-"*100)
     print("Defaulter's statistics:")
     get_statistics(df, step=5)

def bar_graph(df, column_name, bins, labels, xlabel):
    
     df_plot = df.copy()
     df_plot[column_name] = pd.cut(df[column_name], include_lowest=True, bins=bins, labels=labels)

     plt.figure()
     df_plot[column_name].value_counts(sort=False).plot(kind='bar')
     plt.subplots_adjust(bottom=0.25)
     plt.ylabel('No. of defaulters')
     plt.xlabel(xlabel)

def plot_graph(defaulter):
     #Debt bar graph --- debt/income ratio
     bar_graph(defaulter, 'Debt', bins = [0, 0.1, 0.2, 0.5, 0.75, 1, 1.5, 2, 5, 10, float('inf')],
                                   labels = [
                                        "0-10%",
                                        "10-20%",
                                        "20-50%",
                                        "50-75%",
                                        "75-100%",
                                        "100-150%",
                                        "150-200%",
                                        "200-500%",
                                        "500-1000%",
                                        ">1000%"
                                        ],
                                   xlabel='Debt/Income Ratio')

     #Late30_59 bar graph
     bar_graph(defaulter, 'Late30_59', bins=[-1, 0, 1, 2, 3, 4, float('inf')], labels=['0', '1', '2', '3', '4', '4+'], xlabel='Late30_59')

     #Late60_89 bar graph
     bar_graph(defaulter, 'Late60_89', bins=[-1, 0, 1, 2, 3, 4, float('inf')], labels=['0', '1', '2', '3', '4', '4+'], xlabel='Late60_89')

     #Late90 bar graph
     bar_graph(defaulter, 'Late90', bins=[-1, 0, 1, 2, 3, 4, float('inf')], labels=['0', '1', '2', '3', '4', '4+'], xlabel='Late90')

     #OpenCred bar graph
     bar_graph(defaulter, 'OpenCred', bins=[0, 5, 10, 15, 20, 25, float('inf')], labels=['0-5', '6-10', '11-15', '16-20', '21-25', '25+'], xlabel='OpenCred')

     #RealEstate bar graph
     bar_graph(defaulter, 'RealEstate', bins=[-1, 0, 1, 2, 3, float('inf')], labels=['0', '1', '2', '3', '3+'], xlabel='RealEstate')

     #No.Dep bar graph
     bar_graph(defaulter, 'No.Dep', bins=[-1, 0, 1, 2, 3, 4, 5, float('inf')], labels=['0', '1', '2', '3', '4', '5', '5+'], xlabel='No.Dep')

     #Age bar graph
     bar_graph(defaulter, 'Age', bins=[0, 17, 22, 25, 30, 40, 50, 60, 70, float('inf')], labels=['0-17', '18-22', '23-25', '26-30', '31-40', '41-50', '51-60', '61-70', '70+'], xlabel='Age')

     #Util bar graph
     bar_graph(defaulter, 'Util', labels= [
                                                       "0-10%",
                                                       "10-20%",
                                                       "20-50%",
                                                       "50-75%",
                                                       "75-95%",
                                                       "95-100%",
                                                       "100-150%",
                                                       "150-200%",
                                                       "200-500%",
                                                       ">500%"
                                                       ],
                                   bins=[0, 0.1, 0.2, 0.5, 0.75, 0.95, 1, 1.5, 2, 5, float('inf')], xlabel='Util')

     #Income bar graph
     bar_graph(defaulter, 'Income', bins=[0, 1e3, 2e3, 4e3, 6e3, 8e3, 10e3, 15e3, 20e3, 50e3, float('inf')],
                                        labels=[
                                             '0-1k',
                                             '1-2k',
                                             '2-4k',
                                             '4-6k',
                                             '6-8k',
                                             '8-10k',
                                             '10-15k',
                                             '15-20k',
                                             '20-50k',
                                             '50k+'],
                                   xlabel='Income')

def correlation_analysis(df):
     #Prints the Spearman's correlation of columns
     corr = df.corr(method='spearman')
     print("-"*150)
     print("Correlation analysis: (Spearman's method)")
     print(corr)

     #Identifying pairs with correlation greather than 0.4
     print("-"*150)
     print("Pairs with Spearman's correlation greater than 0.4:")
     for i in corr.columns:
          for j in corr.columns:
               if (i > j and corr.loc[i,j] > 0.4):
                    print(f"Correlation between {i} and {j}: {corr.loc[i,j]}")

     #Plots the heatmap
     corr_plot = corr.copy()
     corr_plot.columns = range(1, len(corr.columns.to_list()) + 1)
     corr_plot.index = range(1, len(corr.columns.to_list()) + 1)

     #Mask for the heatmap
     mask = np.triu(np.ones(corr_plot.shape, dtype=bool), k=1)

     plt.figure(figsize=(20,20))
     plt.title('Heatmap for Spearmans\'s correlation')
     sns.heatmap(corr_plot, mask=mask, annot=True, fmt='.2f', annot_kws={'size':7})

     legenda = """
          1  = Default
          2  = Util
          3  = Age
          4  = Late30_59
          5  = Debt
          6  = Income
          7  = OpenCred
          8  = Late90
          9  = RealEstate
          10 = Late60_89
          11 = No.Dep
          """

     plt.figtext(
          0.85,      # posição horizontal
          0.5,       # posição vertical
          legenda,
          fontsize=9,
          va='center'
     )

# This function includes the least element of the first range (include_lowest=True)
def default_rate_by_range(df, column_name, bins, labels, xlabel):

     defaulters_by_range = []

     for i in range(0, len(bins) - 1):
          if (i != 0):
               clients = df[(df[column_name] > bins[i]) & (df[column_name] <= bins[i+1])]
          else:
               clients = df[(df[column_name] >= bins[0]) & (df[column_name] <= bins[1])]

          defaulters = clients[clients['Default'] == 1]

          if (clients.shape[0] != 0):
               defaulters_by_range.append(defaulters.shape[0]/clients.shape[0]*100)
          else:
               defaulters_by_range.append(0)

          print(f'{labels[i]}:{defaulters_by_range[i]:.1f}')

     plt.figure()
     plt.ylabel('% of defaulters by range')
     plt.xlabel(column_name)
     plt.bar(labels, defaulters_by_range)


     #credit_by_range = pd.cut(df[column_name], bins=bins, labels=labels)
     #defaulters_by_range = pd.cut(df[df['Default'] == 1][column_name], bins=bins, labels=labels)
     
#######################################################################

ROOT = plib.Path(__file__).resolve().parent

credit_data = pd.read_csv(ROOT / 'data' / 'cs-training.csv')

colunas_desc = """
ID
    Unique identifier for each borrower record

Default
    Loan default indicator (1 = default, 0 = no default)

Util
    Credit utilization ratio (revolving balance / credit limit)

Age
    Age of the borrower in years

Late30_59
    Number of times borrower was 30–59 days past due

Debt
    Debt-to-income ratio

Income
    Monthly income of the borrower

OpenCred
    Number of open credit lines and loans

Late90
    Number of times borrower was 90+ days past due

RealEstate
    Number of mortgage/real estate loans

Late60_89
    Number of times borrower was 60–89 days past due

No.Dep
    Number of dependents in the borrower's household
"""

credit_data.columns = [
    'ID',
    'Default',
    'Util',
    'Age',
    'Late30_59',
    'Debt',
    'Income',
    'OpenCred',
    'Late90',
    'RealEstate',
    'Late60_89',
    'No.Dep'
]

credit_data = credit_data.drop(columns='ID')

print(f'Data shape: (n_rows, n_columns): {credit_data.shape}')
print(f'Data columns and types: \n{credit_data.dtypes}')
print(f'First 5 data:\n {credit_data.head()}')
print(credit_data.info())

get_statistics(credit_data, step=5)

correlation_analysis(credit_data)

defaulter = credit_data[credit_data['Default']==1]

defaulter_statistics(defaulter)
quantile_analysis(defaulter)
plot_graph(defaulter)

default_rate_by_range(credit_data, 'Age', bins=[0, 17, 22, 25, 30, 40, 50, 60, 70, float('inf')], labels=['0-17', '18-22', '23-25', '26-30', '31-40', '41-50', '51-60', '61-70', '70+'], xlabel='Age')

default_rate_by_range(
    credit_data,
    'Util',
    bins=[0, 10, 20, 50, 75, 95, 100, 150, 200, 500, float('inf')],
    labels=[
        '0–10%',
        '10–20%',
        '20–50%',
        '50–75%',
        '75–95%',
        '95–100%',
        '100–150%',
        '150–200%',
        '200–500%',
        '>500%'
    ],
     xlabel = 'Util'
)

default_rate_by_range(
     credit_data,
     'Late30_59',
     bins=[0, 1, 2, 3, 4, float('inf')],
     labels=['1', '2', '3', '4', '4+'],
     xlabel='Late30_59'
)

default_rate_by_range(
     credit_data,
     'Late60_89',
     bins=[0, 1, 2, 3, 4, float('inf')],
     labels=['1', '2', '3', '4', '4+'],
     xlabel='Late60_89'
)

default_rate_by_range(
     credit_data,
     'Late90',
     bins=[0, 1, 2, 3, 4, float('inf')],
     labels=['1', '2', '3', '4', '4+'],
     xlabel='Late90'
)

default_rate_by_range(
     credit_data,
     'OpenCred',
     bins=[0, 5, 10, 15, 20, 25, float('inf')],
     labels=['0-5', '6-10', '11-15', '16-20', '21-25', '25+'],
     xlabel='OpenCred'
)

default_rate_by_range(
     credit_data,
     'Income',
     bins=[0, 1, 2, 4, 6, 8, 10, 15, 20, 50, float('inf')],
     labels=['0-1k', '1-2k', '2-4k', '4-6k', '6-8k', '8-10k', '10-15k', '15-20k', '20-50k', '50+'],
     xlabel='Income'
)

default_rate_by_range(
     credit_data,
     'No.Dep',
     bins=[0, 1, 2, 3, 4, 5, float('inf')],
     labels=['1', '2', '3', '4', '5', '5+'],
     xlabel='No.Dep'
)

default_rate_by_range(
    credit_data,
    'Debt',
    bins=[0, 10, 20, 50, 75, 100, 150, 200, 500, 1000, float('inf')],
    labels=[
        '0–10%',
        '10–20%',
        '20–50%',
        '50–75%',
        '75–100%',
        '100–150%',
        '150–200%',
        '200–500%',
        '500–1000%',
        '>1000%'
    ],
    xlabel='Debt'
)

default_rate_by_range(
    credit_data,
    'RealEstate',
    bins=[0, 1, 2, 3, float('inf')],
    labels=[
        '1',
        '2',
        '3',
        '3+'
    ],
    xlabel='RealEstate'
)

plt.show()
