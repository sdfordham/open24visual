import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
import plotly.express as px

_DATE_FORMAT = '%d %b %y'
_CATEGORY = 'Category'
_SUBCATEGORY = 'Subcategory'
_OTHER = 'Other'
_ROLLING_WINDOWS = ['30d', '60d']
_MONTH = 'Month'


class Open24Visual:
    def __init__(self, path: str = None):
        self.data = None if path is None else self.load(path)
        if self.data is not None:
            for att in ('description', 'in', 'out', 'balance', ):
                setattr(self, att, self._get_column_name(self.data, att))

    def load(self, path: str) -> pd.DataFrame:
        # Load the Open24 xls
        data = pd.read_csv(path, encoding='utf-16', sep='\t', thousands=',')
        date_col = self._get_column_name(data, 'date')
        data[date_col] = pd.to_datetime(data[date_col], format=_DATE_FORMAT)
        desc_col = self._get_column_name(data, 'description')
        data[desc_col] = data[desc_col].astype('string')
        data.set_index(date_col, inplace=True)
        data.sort_index(inplace=True)
        return data

    @staticmethod
    def _get_column_name(data: pd.DataFrame, column: str) -> str:
        # Need this due to column names changing
        for c in data.columns:
            if c.lower().find(column.lower()) > -1:
                return c
        else:
            raise ValueError(f'Cannot match {column} with a data column')

    def categorise_data(self, yaml_path: str):
        # Adds a category & subcategory column by grep'ing from description column using YAML dictionary
        if self.data is None:
            raise ValueError('No data loaded')
        with open(yaml_path, 'r') as Y:
            category_dict = yaml.load(Y, Loader=yaml.Loader)
        self.data[_CATEGORY] = self.data[self.description].apply(lambda x: self._categorise(x, category_dict)[0])
        self.data[_CATEGORY] = self.data[_CATEGORY].astype('category')
        self.data[_SUBCATEGORY] = self.data[self.description].apply(lambda x: self._categorise(x, category_dict)[1])
        self.data[_SUBCATEGORY] = self.data[_SUBCATEGORY].astype('category')

    @staticmethod
    def _categorise(x: str, category_dict: dict) -> tuple:
        for k, v in category_dict.items():
            for el in v:
                if x.lower().find(el.lower()) > -1:
                    return k, el
        return _OTHER, _OTHER

    def plot_balance(self):
        # Plot balance over time with rolling averages using Matplotlib
        if self.data is None:
            raise ValueError('No data available')
        data_no_zero = self.data[self.data[self.balance] != 0.0]
        balance_data = data_no_zero[self.balance]
        for p in _ROLLING_WINDOWS:
            r = balance_data.rolling(p).mean()
            plt.plot(r, label=f'{p} rolling average')
        plt.plot(balance_data, label=balance_data.name)
        plt.legend()
        plt.show()

    def plot_category_totals(self):
        # Plot category totals per month using Plotly
        if self.data is None:
            raise ValueError('No data available')
        if _CATEGORY not in self.data.columns:
            raise ValueError('Categorise data first')
        _data = self.data.copy()
        _data[_MONTH] = _data.index.strftime('%Y-%m')
        stats = pd.pivot_table(_data, values=self.out, index=_MONTH, columns=_CATEGORY, aggfunc=np.sum)
        fig = px.bar(stats, barmode='group')
        fig.show()

    def plot_subcategory_totals(self, category: str):
        # Plot subcategory totals per month for a given category using Plotly
        if self.data is None:
            raise ValueError('No data available')
        if _SUBCATEGORY not in self.data.columns:
            raise ValueError('Categorise data first')
        if category not in self.data[_CATEGORY].unique():
            raise ValueError('Not a valid category')
        _data = self.data.copy()
        _data[_MONTH] = _data.index.strftime('%Y-%m')
        _data = _data[_data[_CATEGORY] == category]
        stats = pd.pivot_table(_data, values=self.out, index=_MONTH, columns=_SUBCATEGORY, aggfunc=np.sum)
        fig = px.bar(stats, barmode='group')
        fig.show()


if __name__ == '__main__':
    DATA_PATH = r'permanent tsb - Transaction list.xls'
    YAML_PATH = r'categories.yaml'

    o = Open24Visual(DATA_PATH)
    o.categorise_data(YAML_PATH)
    o.plot_balance()
    o.plot_category_totals()
    # o.plot_subcategory_totals('groceries')
