"""Visualization utilities for the Copulas library."""

import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff


class PlotConfig:
    """Custom plot settings for visualizations."""

    DATACEBO_DARK = '#000036'
    DATACEBO_GREEN = '#01E0C9'
    BACKGROUND_COLOR = '#F5F5F8'
    FONT_SIZE = 18


def _generate_1d_plot(data, title, labels, colors):
    """Generate a density plot of an array-like structure.

    Args:
        data (array-like structure):
            The data to plot.
        title (str):
            The title of the plot.
        labels (list[str]):
            The labels of the data.
        colors (list[str]):
            The colors of the data.

    Returns:
        plotly.graph_objects._figure.Figure
    """
    fig = ff.create_distplot(
        hist_data=data, group_labels=labels, show_hist=False, show_rug=False, colors=colors
    )

    for i, name in enumerate(labels):
        fig.update_traces(
            x=fig.data[i].x,
            hovertemplate=f'<b>{name}</b><br>Frequency: %{{y}}<extra></extra>',
            selector={'name': name},
            fill='tozeroy',
        )

    fig.update_layout(
        title=title,
        plot_bgcolor=PlotConfig.BACKGROUND_COLOR,
        font={'size': PlotConfig.FONT_SIZE},
        showlegend=True if labels[0] else False,
        xaxis_title='value',
        yaxis_title='frequency',
    )

    return fig


def dist_1d(data, title=None, label=None):
    """Plot the 1 dimensional data.

    Args:
        data (array_like structure):
            The table data.
        title (str):
            The title of the plot.
        label (str):
            The label of the plot.

    Returns:
        plotly.graph_objects._figure.Figure
    """
    if not title:
        title = 'Data'
        if isinstance(data, pd.DataFrame):
            title += f" for column '{data.columns[0]}'"
        elif isinstance(data, pd.Series) and data.name:
            title += f" for column '{data.name}'"

    return _generate_1d_plot(
        data=[data], title=title, labels=[label], colors=[PlotConfig.DATACEBO_DARK]
    )


def compare_1d(real, synth, title=None):
    """Plot the comparison between real and synthetic data.

    Args:
        real (array_like):
            The real data.
        synth (array_like):
            The synthetic data.
        title (str):
            The title of the plot.

    Returns:
        plotly.graph_objects._figure.Figure
    """
    if not title:
        title = 'Real vs. Synthetic Data'
        if isinstance(real, pd.DataFrame):
            title += f" for column '{real.columns[0]}'"
        elif isinstance(real, pd.Series) and real.name:
            title += f" for column '{real.name}'"

    return _generate_1d_plot(
        data=[real, synth],
        title=title,
        labels=['Real', 'Synthetic'],
        colors=[PlotConfig.DATACEBO_DARK, PlotConfig.DATACEBO_GREEN],
    )


def _generate_scatter_2d_plot(data, columns, color_discrete_map, title):
    """Generate a scatter plot for a pair of columns.

    Args:
        data (pandas.DataFrame):
            The data for the desired column pair containing a
            ``Data`` column indicating whether it is real or synthetic.
        columns (list):
            A list of the columns being plotted.
        color_discrete_map (dict):
            A dictionary mapping the values of the ``Data`` column to the colors
            used to plot them.
        title (str):
            The title of the plot.

    Returns:
        plotly.graph_objects._figure.Figure
    """
    if columns:
        columns.append('Data')
    else:
        columns = data.columns

    if len(columns) != 3:  # includes the 'Data' column
        raise ValueError('Only 2 columns can be plotted')

    fig = px.scatter(
        data,
        x=columns[0],
        y=columns[1],
        color='Data',
        color_discrete_map=color_discrete_map,
        symbol='Data',
    )

    fig.update_layout(
        title=title,
        plot_bgcolor=PlotConfig.BACKGROUND_COLOR,
        font={'size': PlotConfig.FONT_SIZE},
        showlegend=False if len(color_discrete_map) == 1 else True,
    )

    return fig


def scatter_2d(data, columns=None, title=None):
    """Plot 2 dimensional data in a scatter plot.

    Args:
        data (pandas.DataFrame):
            The table data.
        columns (list[string]):
            The names of the two columns to plot.
        title (str):
            The title of the plot.

    Returns:
        plotly.graph_objects._figure.Figure
    """
    data = data.copy()
    data['Data'] = 'Real'

    if not title:
        title = 'Data'
        if columns:
            title += f" for columns '{columns[0]}' and '{columns[1]}'"
        elif isinstance(data, pd.DataFrame):
            title += f" for columns '{data.columns[0]}' and '{data.columns[1]}'"

    return _generate_scatter_2d_plot(
        data=data,
        columns=columns,
        color_discrete_map={'Real': PlotConfig.DATACEBO_DARK},
        title=title,
    )


def compare_2d(real, synth, columns=None, title=None):
    """Plot the comparison between real and synthetic data for a given column pair.

    Args:
        real (pandas.DataFrame):
            The real table data.
        synth (pandas.Dataframe):
            The synthetic table data.
        columns (list[string]):
            The names of the two columns to plot.
        title (str):
            The title of the plot.

    Returns:
        plotly.graph_objects._figure.Figure
    """
    real, synth = real.copy(), synth.copy()
    real['Data'] = 'Real'
    synth['Data'] = 'Synthetic'
    data = pd.concat([real, synth], axis=0, ignore_index=True)

    if not title:
        title = 'Real vs. Synthetic Data'
        if columns:
            title += f" for columns '{columns[0]}' and '{columns[1]}'"
        elif isinstance(data, pd.DataFrame):
            title += f" for columns '{data.columns[0]}' and '{data.columns[1]}'"

    return _generate_scatter_2d_plot(
        data=data,
        columns=columns,
        color_discrete_map={
            'Real': PlotConfig.DATACEBO_DARK,
            'Synthetic': PlotConfig.DATACEBO_GREEN,
        },
        title=title,
    )


def _generate_scatter_3d_plot(data, columns, color_discrete_map, title):
    """Generate a scatter plot for column pair plot.

    Args:
        data (pandas.DataFrame):
            The data for the desired three columns containing a
            ``Data`` column that indicates whether it is real or synthetic.
        columns (list):
            A list of the columns being plotted.
        color_discrete_map (dict):
            A dictionary mapping the values of the ``Data`` column to the colors
            used to plot them.
        title (str):
            The title of the plot.

    Returns:
        plotly.graph_objects._figure.Figure
    """
    if columns:
        columns.append('Data')
    else:
        columns = data.columns

    if len(columns) != 4:  # includes the 'Data' column
        raise ValueError('Only 3 columns can be plotted')

    fig = px.scatter_3d(
        data,
        x=columns[0],
        y=columns[1],
        z=columns[2],
        color='Data',
        color_discrete_map=color_discrete_map,
        symbol='Data',
    )

    fig.update_traces(marker={'size': 5})

    fig.update_layout(
        title=title,
        plot_bgcolor=PlotConfig.BACKGROUND_COLOR,
        font={'size': PlotConfig.FONT_SIZE},
        showlegend=False if len(color_discrete_map) == 1 else True,
    )

    return fig


def scatter_3d(data, columns=None, title=None):
    """Plot 3 dimensional data in a scatter plot.

    Args:
        data (pandas.DataFrame):
            The table data. Must have at least 3 columns.
        columns (list[string]):
            The names of the three columns to plot.
        title (str):
            The title of the plot.

    Returns:
        plotly.graph_objects._figure.Figure
    """
    data = data.copy()
    data['Data'] = 'Real'

    if not title:
        title = 'Data'
        if columns:
            title += f" for columns '{columns[0]}', '{columns[1]}' and '{columns[2]}'"
        elif isinstance(data, pd.DataFrame):
            title += (
                f" for columns '{data.columns[0]}', '{data.columns[1]}' and '{data.columns[2]}'"
            )

    return _generate_scatter_3d_plot(
        data=data,
        columns=columns,
        color_discrete_map={'Real': PlotConfig.DATACEBO_DARK},
        title=title,
    )


def compare_3d(real, synth, columns=None, title=None):
    """Plot the comparison between real and synthetic data for a given column triplet.

    Args:
        real (pd.DataFrame):
            The real data.
        synth (pd.DataFrame):
            The synthetic data.
        columns (list):
            The name of the columns to plot.
        title (str):
            The title of the plot.
    """
    real, synth = real.copy(), synth.copy()
    real['Data'] = 'Real'
    synth['Data'] = 'Synthetic'
    data = pd.concat([real, synth], axis=0, ignore_index=True)

    if not title:
        title = 'Real vs. Synthetic Data'
        if columns:
            title += f" for columns '{columns[0]}', '{columns[1]}' and '{columns[2]}'"
        elif isinstance(data, pd.DataFrame):
            title += (
                f" for columns '{data.columns[0]}', '{data.columns[1]}' and '{data.columns[2]}'"
            )

    return _generate_scatter_3d_plot(
        data=data,
        columns=columns,
        color_discrete_map={
            'Real': PlotConfig.DATACEBO_DARK,
            'Synthetic': PlotConfig.DATACEBO_GREEN,
        },
        title=title,
    )
