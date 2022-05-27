import sys

import pandas as pd
import plotly.graph_objs as go
from plotly.offline import iplot

from datasets.cont_smr import *


def plot_3dSurface_and_heatmap(eeg_data, clab):
    temp_df = pd.DataFrame(data=eeg_data, index=clab)
    channel = np.arange(62)
    sensor_positions = clab
    data = [go.Surface(z=temp_df, colorscale='Bluered')]
    layout = go.Layout(
        title='<br>3d Surface and Heatmap of Sensor Values ',
        width=800,
        height=900,
        autosize=False,
        margin=dict(t=0, b=0, l=0, r=0),
        scene=dict(
            xaxis=dict(
                title='Time (sample num)',
                gridcolor='rgb(255, 255, 255)',
                #             erolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            ),
            yaxis=dict(
                title='Channel',
                tickvals=channel,
                ticktext=sensor_positions,
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230, 230)'
            ),
            zaxis=dict(
                title='Sensor Value',
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            ),
            aspectratio=dict(x=1, y=1, z=0.5),
            aspectmode='manual'
        )
    )
    updatemenus = list([
        dict(
            buttons=list([
                dict(
                    args=['type', 'surface'],
                    label='3D Surface',
                    method='restyle'
                ),
                dict(
                    args=['type', 'heatmap'],
                    label='Heatmap',
                    method='restyle'
                )
            ]),
            direction='left',
            pad={'r': 10, 't': 10},
            showactive=True,
            type='buttons',
            x=0.1,
            xanchor='left',
            y=1.1,
            yanchor='top'
        ),
    ])

    annotations = list([
        dict(text='Trace type:', x=0, y=1.085, yref='paper', align='left', showarrow=False)
    ])
    layout['updatemenus'] = updatemenus
    layout['annotations'] = annotations

    fig = dict(data=data, layout=layout)
    iplot(fig)


if __name__ == "__main__":

    module_path = os.path.abspath(os.path.join('..'))
    if module_path not in sys.path:
        sys.path.append(module_path)

    DATA_PATH = os.path.join(module_path, "data/SMR/raw")

    data, fs, clab, mnt, mrk_class, mrk_className, task_type, task_typeName, timepoints, trial_artifact = \
        load_matlab_data_complete(subjectno=1, sessionno=1, data_path=DATA_PATH, outfile=False)

    idx_class_0 = np.squeeze(np.argwhere(task_type == 0))
    d = data[idx_class_0][0].tolist()
    index_clab = clab

    df = pd.DataFrame(data=d, index=index_clab)
    channel = np.arange(62)
    sensor_positions = clab

    # list_of_pairs = []
    # j = 0
    # for column in sample_corr_df.columns:
    #     j += 1
    #     for i in range(j, len(sample_corr_df)):
    #         if column != sample_corr_df.index[i]:
    #             temp_pair = [column + '-' + sample_corr_df.index[i]]
    #             list_of_pairs.append(temp_pair)
    #
    # corr_pairs_dict = {}
    # for i in range(len(list_of_pairs)):
    #     temp_corr_pair = dict(zip(list_of_pairs[i], [0]))
    #     corr_pairs_dict.update(temp_corr_pair)
    #
    # j = 0
    # for column in correlation_df.columns:
    #     j += 1
    #     for i in range(j, len(correlation_df)):
    #         if ((correlation_df[column][i] >= threshold) & (column != correlation_df.index[i])):
    #             corr_pairs_dict[column + '-' + correlation_df.index[i]] += 1
    #
    # corr_count = pd.DataFrame(corr_pairs_dict, index=['count']).T.reset_index(drop=False).rename(
    #     columns={'index': 'channel_pair'})
    # print('Channel pairs that have correlation value >= ' + str(threshold) + ' (' + group + ' group):')
    # print(corr_count['channel_pair'][corr_count['count'] > 0].tolist())
