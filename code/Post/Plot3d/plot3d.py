import plotly.graph_objs as go
from plotly.subplots import make_subplots
import os
import pandas as pd

def create_and_save_3d_scatter_plot_plotly(subsets_directory, filename):

    color_dict = {
        'Charges': 'red',
        'Inner Domain': 'lightgreen',
        'Interface': 'purple',
        'Outer Domain': 'lightblue',
        'Outer Border': 'orange',
        'Experimental': 'cyan',
        'test': 'red'
    }

    csv_files = [file for file in os.listdir(subsets_directory) if file.endswith('.csv')]
    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])
    for i, csv_file in enumerate(csv_files):

        name = csv_file.replace('.csv','')
        data = pd.read_csv(os.path.join(subsets_directory, csv_file))
        trace = go.Scatter3d(
            x=data['X'],
            y=data['Y'],
            z=data['Z'],
            mode='markers',
            marker=dict(size=4, opacity=0.7, color=color_dict[name]),
            name=name
        )
        fig.add_trace(trace)

    fig.update_layout(title='Dominio 3D', scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))
    path_files = os.path.join(os.getcwd(), 'code', 'Post', 'Plot3d')
    fig.write_html(os.path.join(path_files, filename))

path_files = os.path.join(os.getcwd(),'code','Post','Plot3d','data')
filename = '3d_plot.html'

create_and_save_3d_scatter_plot_plotly(path_files, filename)
