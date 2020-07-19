import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
import plotly.graph_objects as go

_DATE_FORMAT = '%d %b %y'
_MONTH_FORMAT = '%Y-%m'
_CATEGORY = 'Category'
_IDENTIFIER = 'Tag'
_OTHER = 'Other'
_ROLLING_WINDOWS = ['30d', '60d']
_MONTH = 'Month'


class Open24Visual:
    def __init__(self, path: str = None):
        self.data = None if path is None else self.load(path)
        if self.data is not None:
            for attr in ('description', 'in', 'out', 'balance', ):
                setattr(self, attr, self._get_column_name(self.data, attr))

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
        # Adds a category & identifier column by grep'ing from description column using YAML dictionary
        if self.data is None:
            raise ValueError('No data loaded')
        with open(yaml_path, 'r') as Y:
            category_dict = yaml.load(Y, Loader=yaml.Loader)
        self.data[_CATEGORY] = self.data[self.description].apply(
            lambda x: self._categorise(x, category_dict)[0]
        )
        self.data[_CATEGORY] = self.data[_CATEGORY].astype('category')
        self.data[_IDENTIFIER] = self.data[self.description].apply(
            lambda x: self._categorise(x, category_dict)[1]
        )
        self.data[_IDENTIFIER] = self.data[_IDENTIFIER].astype('category')

    @staticmethod
    def _categorise(x: str, category_dict: dict) -> tuple:
        for category, contents in category_dict.items():
            for pattern, tag in contents.items():
                if x.lower().find(pattern.lower()) > -1:
                    return category, tag
        return _OTHER, _OTHER

    def balance_with_trends(self):
        # Balance over time with rolling averages
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

    def monthly_totals(self, category: str = None):
        # Monthly totals for all categories or a given category
        if self.data is None:
            raise ValueError('No data available')
        if _IDENTIFIER not in self.data.columns:
            raise ValueError('Categorise data first')

        _data = self.data.copy()
        _data[_MONTH] = _data.index.strftime(_MONTH_FORMAT)

        if category is None or category.lower() is 'all':
            return pd.pivot_table(_data, values=self.out, index=_MONTH, columns=_IDENTIFIER, aggfunc=np.sum)
        else:
            if category not in self.data[_CATEGORY].unique():
                raise ValueError('Not a valid category')
            _data = _data[_data[_CATEGORY] == category]
            _data.dropna(subset=[self.out], inplace=True)
            return pd.pivot_table(_data, values=self.out, index=_MONTH, columns=_IDENTIFIER, aggfunc=np.sum)


def show_monthly_graphs(o24v: Open24Visual):
    fig = go.Figure()
    location_idx = list()
    for cat in o24v.data[_CATEGORY].unique():
        totals_data = o24v.monthly_totals(cat)
        for col in totals_data.columns:
            fig.add_trace(go.Bar(x=totals_data.index, y=totals_data[col], name=col))
            location_idx.append(cat)

    tick_range = pd.date_range(o24v.data.index.min(), o24v.data.index.max(), freq='MS')
    tick_vals = [d.strftime('%Y-%m') for d in tick_range]

    fig.update_layout(
        updatemenus=[
            go.layout.Updatemenu(
                active=1,
                showactive=True,
                buttons=list(
                    [dict(label=cat,
                          method='update',
                          args=[{'visible': [cat == loc for loc in location_idx]},
                                {'title': f'Monthly spend for category: {cat}',
                                 'showlegend': True}]) for cat in set(location_idx)]
                )
            )
        ],
        xaxis=dict(tickmode='array', tickvals=tick_vals, ticktext=tick_vals)
    )
    fig.show()


if __name__ == '__main__':
    DATA_PATH = r'permanent tsb - Transaction list.xls'
    YAML_PATH = r'categories.yaml'

    o = Open24Visual(DATA_PATH)
    o.categorise_data(YAML_PATH)

    show_monthly_graphs(o)
