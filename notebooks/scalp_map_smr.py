from pathlib import Path

import numpy as np
import pyrootutils
import scipy as sp
from bokeh.io import curdoc
from bokeh.layouts import row
from bokeh.models import BasicTicker
from bokeh.models import ColorBar, LinearColorMapper
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.smr_datamodule import SMR_Data, normalize

def bokeh_scalpmap(mnt, v, clim='minmax', cb_label='', plot_width=250, plot_height=250):
    '''
    Usage:
        bokeh_scalpmap(mnt, v, clim='minmax', cb_label='')
    Parameters:
        ... (same as before)
    Returns:
        Bokeh figure object with the scalpmap plotted.
    '''

    # interpolate between channels
    xi, yi = np.linspace(-1, 1, 100), np.linspace(-1, 1, 100)
    xi, yi = np.meshgrid(xi, yi)
    rbf = sp.interpolate.Rbf(mnt[:, 0], mnt[:, 1], v, function='linear')
    zi = rbf(xi, yi)

    # mask area outside of the scalp
    a, b, n, r = 50, 50, 100, 50
    mask_y, mask_x = np.ogrid[-a:n - a, -b:n - b]
    mask = mask_x * mask_x + mask_y * mask_y >= r * r
    zi[mask] = np.nan

    if clim == 'minmax':
        vmin = v.min()
        vmax = v.max()
    elif clim == 'sym':
        vmin = -np.absolute(v).max()
        vmax = np.absolute(v).max()
    else:
        vmin = clim[0]
        vmax = clim[1]

    mapper = LinearColorMapper(palette="Viridis256", low=vmin, high=vmax)

    p = figure(width=plot_width, height=plot_height, x_range=(-1, 1), y_range=(-1, 1), tools="")
    p.image(image=[zi], x=-1, y=-1, dw=2, dh=2, color_mapper=mapper)
    p.circle(mnt[:, 0], mnt[:, 1], color='black', size=5)

    # Add a color bar
    color_bar = ColorBar(color_mapper=mapper, ticker=BasicTicker(),
                         label_standoff=12, border_line_color=None, location=(0, 0))
    p.add_layout(color_bar, 'right')

    p.axis.visible = False
    p.grid.visible = False

    return p

def normalize_positions(mnt):
    # Normalize x and y coordinates to fit within [-1, 1]
    mnt[:, 0] = 2 * (mnt[:, 0] - mnt[:, 0].min()) / (mnt[:, 0].max() - mnt[:, 0].min()) - 1
    mnt[:, 1] = 2 * (mnt[:, 1] - mnt[:, 1].min()) / (mnt[:, 1].max() - mnt[:, 1].min()) - 1
    return mnt


if __name__ == "__main__":
    # Using a raw string
    data_dir = Path("D:\\SMR\\")
    task_name = "2D"
    subject_sessions_dict = {"S4": "all"}
    loading_data_mode = "within_subject"
    ival = "2s:10s:1ms"
    bands = [8, 13]
    chans = "*"
    fallback_neighbors = 4
    transform = None
    normalize_dict = {"norm_type": "std", "norm_axis": 0}

    smr_datamodule_nn = SMR_Data(data_dir=data_dir,
                                 task_name=task_name,
                                 subject_sessions_dict=subject_sessions_dict,
                                 loading_data_mode=loading_data_mode,
                                 ival=ival,
                                 bands=bands,
                                 chans=chans,
                                 fallback_neighbors=fallback_neighbors,
                                 transform=transform,
                                 normalize=normalize_dict,
                                 process_noisy_channels=False)

    subjects_sessions_path_dict = smr_datamodule_nn.collect_subject_sessions(subject_sessions_dict)
    subject_data_dict, subjects_info_dict = smr_datamodule_nn.load_subjects_sessions(subjects_sessions_path_dict)

    subject_name = list(subject_data_dict.keys())[0]
    loaded_subject_sessions = subject_data_dict[subject_name]
    loaded_subject_sessions_info = subjects_info_dict[subject_name]["sessions_info"]

    # append the sessions (FIXME : forced trials are not used)
    valid_trials = smr_datamodule_nn.append_sessions(loaded_subject_sessions,
                                                     loaded_subject_sessions_info)

    mrk_classname = valid_trials.className

    valid_trials_norm, norm_params_valid = normalize(valid_trials,
                                                     norm_type=normalize_dict["norm_type"],
                                                     axis=normalize_dict["norm_axis"])

    epo = valid_trials_norm
    normalized_mnt = normalize_positions(epo.chans.mnt[:, :2])

    # Generate intervals with 100ms windows
    start_times = np.arange(3000, 7300, 40)
    ival = [[start, start + 100] for start in start_times]

    # Set up a row of plots for each class
    plots = [bokeh_scalpmap(epo.chans.mnt[:, :2], np.zeros(epo.chans.mnt.shape[0])) for _ in mrk_classname]
    plot_sources = [ColumnDataSource(data={'image': [np.random.random((100, 100))]}) for _ in plots]

    for p, source in zip(plots, plot_sources):
        p.renderers[0].data_source = source

    # The function to update the plots
    current_interval = 0


    def update():
        global current_interval
        for klass_idx, klass_name in enumerate(mrk_classname):
            start, end = ival[current_interval]
            indices = (epo.t >= start) & (epo.t <= end)
            mean = np.mean(epo[:, :, indices][epo.y == klass_idx, :, :], axis=(0, 2))

            # Interpolate between channels (similar to scalpmap function)
            xi, yi = np.linspace(-1, 1, 100), np.linspace(-1, 1, 100)
            xi, yi = np.meshgrid(xi, yi)
            rbf = sp.interpolate.Rbf(epo.chans.mnt[:, 0], epo.chans.mnt[:, 1], mean, function='linear')
            zi = rbf(xi, yi)

            plot_sources[klass_idx].data.update(image=[zi])

        current_interval = (current_interval + 1) % len(ival)


    # Add the plots to the current document
    curdoc().add_root(row(*plots))
    curdoc().add_periodic_callback(update, 500)
