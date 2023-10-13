"""Visualization utilities for the Copulas library."""

import pandas as pd

from copulas.utils2 import PlotConfig
import plotly.express as px
from pandas.api.types import is_datetime64_dtype


def _generate_column_bar_plot(real_data, synthetic_data, plot_kwargs={}):
    """Generate a bar plot of the real and synthetic data.

    Args:
        real_column (pandas.Series):
            The real data for the desired column.
        synthetic_column (pandas.Series):
            The synthetic data for the desired column.
        plot_kwargs (dict, optional):
            Dictionary of keyword arguments to pass to px.histogram. Keyword arguments
            provided this way will overwrite defaults.

    Returns:
        plotly.graph_objects._figure.Figure
    """
    all_data = pd.concat([real_data, synthetic_data], axis=0, ignore_index=True)
    histogram_kwargs = {
        'x': 'values',
        'barmode': 'group',
        'color_discrete_sequence': [PlotConfig.DATACEBO_DARK, PlotConfig.DATACEBO_GREEN],
        'pattern_shape': 'Data',
        'pattern_shape_sequence': ['', '/'],
        'histnorm': 'probability density',
    }
    histogram_kwargs.update(plot_kwargs)
    fig = px.histogram(
        all_data,
        **histogram_kwargs
    )

    return fig

def _generate_scatter_plot(all_data, columns):
    """Generate a scatter plot for column pair plot.

    Args:
        all_data (pandas.DataFrame):
            The real and synthetic data for the desired column pair containing a
            ``Data`` column that indicates whether is real or synthetic.
        columns (list):
            A list of the columns being plotted.

    Returns:
        plotly.graph_objects._figure.Figure
    """
    fig = px.scatter(
        all_data,
        x=columns[0],
        y=columns[1],
        color='Data',
        color_discrete_map={
            'Real': PlotConfig.DATACEBO_DARK,
            'Synthetic': PlotConfig.DATACEBO_GREEN
        },
        symbol='Data'
    )

    fig.update_layout(
        title=f"Real vs. Synthetic Data for columns '{columns[0]}' and '{columns[1]}'",
        plot_bgcolor=PlotConfig.BACKGROUND_COLOR,
        font={'size': PlotConfig.FONT_SIZE},
    )

    return fig

def _generate_column_plot(real_column,
                          synthetic_column,
                          plot_kwargs={},
                          plot_title=None,
                          x_label=None):
    """Generate a plot of the real and synthetic data.

    Args:
        real_column (pandas.Series):
            The real data for the desired column.
        synthetic_column (pandas.Series):
            The synthetic data for the desired column.
        hist_kwargs (dict, optional):
            Dictionary of keyword arguments to pass to px.histogram. Keyword arguments
            provided this way will overwrite defaults.
        plot_title (str, optional):
            Title to use for the plot. Defaults to 'Real vs. Synthetic Data for column {column}'
        x_label (str, optional):
            Label to use for x-axis. Defaults to 'Category'.

    Returns:
        plotly.graph_objects._figure.Figure
    """
    column_name = real_column.name if hasattr(real_column, 'name') else ''

    real_data = pd.DataFrame({'values': real_column.copy().dropna()})
    real_data['Data'] = 'Real'
    synthetic_data = pd.DataFrame({'values': synthetic_column.copy().dropna()})
    synthetic_data['Data'] = 'Synthetic'

    is_datetime_sdtype = False
    if is_datetime64_dtype(real_column.dtype):
        is_datetime_sdtype = True
        real_data['values'] = real_data['values'].astype('int64')
        synthetic_data['values'] = synthetic_data['values'].astype('int64')

    trace_args = {}

    fig = _generate_column_bar_plot(real_data, synthetic_data, plot_kwargs)

    for i, name in enumerate(['Real', 'Synthetic']):
        fig.update_traces(
            x=pd.to_datetime(fig.data[i].x) if is_datetime_sdtype else fig.data[i].x,
            hovertemplate=f'<b>{name}</b><br>Frequency: %{{y}}<extra></extra>',
            selector={'name': name},
            **trace_args
        )

    if not plot_title:
        plot_title = f"Real vs. Synthetic Data for column '{column_name}'"

    if not x_label:
        x_label = 'Category'

    fig.update_layout(
        title=plot_title,
        xaxis_title=x_label,
        yaxis_title='Frequency',
        plot_bgcolor=PlotConfig.BACKGROUND_COLOR,
        annotations=[],
        font={'size': PlotConfig.FONT_SIZE},
    )
    return fig

def hist_1d(data, title=None, bins=20, label=None):
    """Plot 1 dimensional data in a histogram.
    
    Args:
        data (pd.DataFrame):
            The table data.
        title (str):
            The title of the plot.
        bins (int):
            The number of bins to use for the histogram.
        label (str):
            The label of the plot.
    
    Returns:
        plotly.graph_objects._figure.Figure
    """
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    if len(data.columns) > 1:
        raise ValueError('Only 1 column can be plotted')

    fig = px.histogram(
        data_frame=data,
        barmode='group',
        color_discrete_sequence=[PlotConfig.DATACEBO_DARK],
        histnorm='probability density',
        title=title,
        nbins=bins,
    )
    fig.update_layout(
        xaxis_title='',
        yaxis_title='',
        legend_title=label,
        showlegend=True if label else False
    )

    return fig

def compare_1d(real, synth):
    """Return a plot of the real and synthetic data for a given column.

    Args:
        real (pandas.DataFrame):
            The real table data.
        synth (pandas.DataFrame):
            The synthetic table data.

    Returns:
        plotly.graph_objects._figure.Figure
    """
    if not isinstance(real, pd.Series):
        real = pd.Series(real)
    if not isinstance(synth, pd.Series):
        synth = pd.Series(synth)

    return _generate_column_plot(real, synth, plot_type='bar')

def scatter_2d(data, columns=None):
    """Plot 2 dimensional data in a scatter plot.

    Args:
        data (pandas.DataFrame):
            The table data.
        columns (list[string]):
            The names of the two columns to plot.

    Returns:
        plotly.graph_objects._figure.Figure
    """
    data = data[columns]
    columns = list(data.columns)
    data['Data'] = 'Real'

    return _generate_scatter_plot(data, columns)

def compare_2d_(real, synth, columns=None):
    """Return a plot of the real and synthetic data for a given column pair.

    Args:
        real (pandas.DataFrame):
            The real table data.
        synth (pandas.Dataframe):
            The synthetic table data.
        columns (list[string]):
            The names of the two columns to plot.

    Returns:
        plotly.graph_objects._figure.Figure
    """
    real_data = real_data[columns]
    synthetic_data = synthetic_data[columns]
    columns = list(real_data.columns)
    real_data['Data'] = 'Real'
    synthetic_data['Data'] = 'Synthetic'
    all_data = pd.concat([real_data, synthetic_data], axis=0, ignore_index=True)

    return _generate_scatter_plot(all_data, columns)

def scatter_3d_plotly(data, columns=None):
    """Return a 3D scatter plot of the data.

    Args:
        data (pandas.DataFrame):
            The table data. Must have at least 3 columns.
        column_names (list[string]):
            The names of the three columns to plot. If not passed,
            the first three columns of the data will be used.

    Returns:
        plotly.graph_objects._figure.Figure
    """
    fig = px.scatter(
        data,
        x=columns[0],
        y=columns[1],
        z=columns[2],
        color='Data',
        color_discrete_map={
            'Real': PlotConfig.DATACEBO_DARK,
            'Synthetic': PlotConfig.DATACEBO_GREEN
        },
        symbol='Data'
    )

    fig.update_layout(
        title=f"Data for columns '{columns[0]}', '{columns[1]}' and '{columns[2]}'",
        plot_bgcolor=PlotConfig.BACKGROUND_COLOR,
        font={'size': PlotConfig.FONT_SIZE},
    )

    return fig

def compare_3d(real, synth, columns=None):
    """Generate a 3d scatter plot comparing real/synthetic data.

    Args:
        real (pd.DataFrame):
            The real data.
        synth (pd.DataFrame):
            The synthetic data.
        columns (list):
            The name of the columns to plot.
    """
    columns = columns or real.columns

    fig = scatter_3d_plotly(real[columns])
    fig = scatter_3d_plotly(synth[columns], fig=fig)

    return fig
