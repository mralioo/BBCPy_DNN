import copy
from collections import defaultdict

import numpy as np
from bbcpy.load.srm_eeg import *
from bokeh.io import output_notebook
from bokeh.models import ColumnDataSource, LabelSet
from bokeh.models import FactorRange
from bokeh.palettes import Spectral6
from bokeh.plotting import figure, show
from bokeh.transform import factor_cmap


def sort_subject_by_key(subject_group_path, key="MBSRsubject"):
    """ Sort the subject_group_path dictionary by the key."""
    sorted_subject_group_path = {}
    if key == "MBSRsubject":
        sorted_subject_group_path["MBSRsubject"] = {}
        sorted_subject_group_path["no_MBSRsubject"] = {}

        for subject_id, subj_dict in subject_group_path.items():
            for session_id, sess_path in subj_dict.items():
                tmp = load_session_metadata(sess_path)
                if tmp[subject_id]["MBSRsubject"] == True:
                    sorted_subject_group_path["MBSRsubject"][subject_id] = copy.deepcopy(subj_dict)
                if tmp[subject_id]["MBSRsubject"] == False:
                    sorted_subject_group_path["no_MBSRsubject"][subject_id] = copy.deepcopy(subj_dict)
    else:
        raise NotImplementedError("The key is not implemented yet.")

    return sorted_subject_group_path


def get_trial_stats(subject_group_path, key="subject_result"):
    """ Get the trial stats from the trial_info_dict."""
    trial_stat_dict = {}
    sessions_ids = []
    subject_ids = []
    num_subjects = len(subject_group_path.keys())
    if key == "subject_result":
        trial_stat_dict["targethitnumber"] = {}
        trial_stat_dict["targethitnumber"]["L"] = [[] for _ in range(num_subjects)]
        trial_stat_dict["targethitnumber"]["R"] = [[] for _ in range(num_subjects)]
        trial_stat_dict["targethitnumber"]["count"] = {"L": [], "R": []}

        trial_stat_dict["result"] = {}
        trial_stat_dict["result"]["failed"] = [[] for _ in range(num_subjects)]
        trial_stat_dict["result"]["succeed"] = [[] for _ in range(num_subjects)]
        trial_stat_dict["result"]["NT"] = [[] for _ in range(num_subjects)]
        trial_stat_dict["result"]["count"] = {"failed": [], "succeed": [], "NT": []}

        trial_stat_dict["artifact"] = {}
        trial_stat_dict["artifact"]["yes"] = [[] for _ in range(num_subjects)]
        trial_stat_dict["artifact"]["no"] = [[] for _ in range(num_subjects)]
        trial_stat_dict["artifact"]["count"] = {"yes": [], "no": []}

    sorted_subject_group_path = sorted(subject_group_path.keys(), key=lambda x: int(x[1:]))

    for i, subject_id in enumerate(sorted_subject_group_path):
        subject_ids.append(subject_id)

        for session_id, sess_path in subject_group_path[subject_id].items():
            tmp = load_session_metadata(sess_path)
            sess_id = session_id.split("_")[-1]
            sessions_ids.append(f"{subject_id}-{sess_id}")
            if key == "subject_result":
                trial_stat_dict["targethitnumber"]["L"][i].append(tmp["targethitnumber"]["L"])
                trial_stat_dict["targethitnumber"]["R"][i].append(tmp["targethitnumber"]["R"])

                trial_stat_dict["result"]["failed"][i].append(tmp["result"]["failed"])
                trial_stat_dict["result"]["succeed"][i].append(tmp["result"]["succeed"])
                trial_stat_dict["result"]["NT"][i].append(tmp["result"]["NT"])

                # artifact trials
                arti_dict = tmp["artifact"]
                if "yes" in arti_dict:
                    trial_stat_dict["artifact"]["yes"][i].append(arti_dict["yes"])
                else:
                    trial_stat_dict["artifact"]["yes"][i].append(0)
                trial_stat_dict["artifact"]["no"][i].append(arti_dict["no"])

        trial_stat_dict["targethitnumber"]["count"]["L"].append(
            np.sum(np.array(trial_stat_dict["targethitnumber"]["L"][i])))
        trial_stat_dict["targethitnumber"]["count"]["R"].append(
            np.sum(np.array(trial_stat_dict["targethitnumber"]["R"][i])))

        trial_stat_dict["result"]["count"]["failed"].append(np.sum(np.array(trial_stat_dict["result"]["failed"][i])))
        trial_stat_dict["result"]["count"]["succeed"].append(np.sum(np.array(trial_stat_dict["result"]["succeed"][i])))
        trial_stat_dict["result"]["count"]["NT"].append(np.sum(np.array(trial_stat_dict["result"]["NT"][i])))

        trial_stat_dict["artifact"]["count"]["yes"].append(np.sum(np.array(trial_stat_dict["artifact"]["yes"][i])))
        trial_stat_dict["artifact"]["count"]["no"].append(np.sum(np.array(trial_stat_dict["artifact"]["no"][i])))

    trial_stat_dict["subject_ids"] = subject_ids
    trial_stat_dict["sessions_ids"] = sessions_ids

    return trial_stat_dict


def plot_histogram_subject(trial_stat_dict, subject_group="MBSRsubject", key="targethitnumber"):
    """ Plot the histogram of the trial stats."""

    subject_ids = trial_stat_dict["subject_ids"]
    # Set the desired width and height of the figure
    width = 1800
    height = 600

    if key == "targethitnumber":
        left_counts = list(trial_stat_dict["targethitnumber"]["count"]["L"])
        right_counts = list(trial_stat_dict["targethitnumber"]["count"]["R"])
        tasks = ["L", "R"]

        plot_title = f"{subject_group} group - Hit Target task count per subject"

        data_dict = {'left': left_counts, 'right': right_counts, 'sessions': subject_ids}
        x = [(session_id, task) for session_id in subject_ids for task in tasks]
        counts = sum(zip(data_dict['left'], data_dict['right']), ())

        source = ColumnDataSource(data=dict(x=x, counts=counts))

    if key == "result":
        failed_counts = list(trial_stat_dict["result"]["count"]["failed"])
        succeed_counts = list(trial_stat_dict["result"]["count"]["succeed"])
        NT_counts = list(trial_stat_dict["result"]["count"]["NT"])
        tasks = ["failed", "succeed", "NT"]

        plot_title = f"{subject_group} group - Trial result count per subject"

        data_dict = {'failed': failed_counts, 'succeed': succeed_counts, 'NT': NT_counts, 'sessions': subject_ids}
        x = [(session_id, task) for session_id in subject_ids for task in tasks]
        counts = sum(zip(data_dict['failed'], data_dict['succeed'], data_dict['NT']), ())

        source = ColumnDataSource(data=dict(x=x, counts=counts))
    if key == "artifact":
        yes_counts = list(trial_stat_dict["artifact"]["count"]["yes"])
        no_counts = list(trial_stat_dict["artifact"]["count"]["no"])
        tasks = ["yes", "no"]

        plot_title = f"{subject_group} group - Artifact count per subject"

        data_dict = {'yes': yes_counts, 'no': no_counts, 'sessions': subject_ids}
        x = [(session_id, task) for session_id in subject_ids for task in tasks]
        counts = sum(zip(data_dict['yes'], data_dict['no']), ())

        source = ColumnDataSource(data=dict(x=x, counts=counts))

    # Set up the figure
    pp = figure(title=plot_title,
                x_range=FactorRange(*x),
                width=width, height=height,)
                # tools="pan,wheel_zoom,box_zoom,reset")

    pp.vbar(x='x', top='counts', width=0.9, source=source, line_color="white",
            fill_color=factor_cmap('x', palette=Spectral6, factors=tasks, start=1, end=2))

    pp.x_range.range_padding = 0.1
    pp.xaxis.major_label_orientation = 1
    pp.xgrid.grid_line_color = None
    pp.background_fill_color = None
    pp.border_fill_color = None

    # Show the plot
    output_notebook()
    show(pp)


def plot_scatter_subject(trial_stat_dict, subject_group="MBSRsubject", key="targethitnumber"):
    """ Plot the scatter of the trial stats."""

    subject_ids = trial_stat_dict["subject_ids"]
    # sessions_ids = trial_stat_dict["sessions_ids"]

    # Set the desired width and height of the figure
    width = 1800
    height = 800

    if key == "targethitnumber":

        x = list(trial_stat_dict["targethitnumber"]["count"]["L"])
        y = list(trial_stat_dict["targethitnumber"]["count"]["R"])

        scatter_plot_title = f"{subject_group} group - Hit Target task count per subject"
        x_axis_label = "Left"
        y_axis_label = "Right"

        shared_points = defaultdict(list)
        # Iterate over the sessions and populate the shared points dictionary
        for i, subject_id in enumerate(subject_ids):
            point = (x[i], y[i])
            shared_points[point].append(subject_id)

        # Convert shared_points dictionary to lists for ColumnDataSource
        x_data = []
        y_data = []
        labels = []
        for point, sessions in shared_points.items():
            x_data.append(point[0])
            y_data.append(point[1])
            label_text = "\n".join(sessions)
            labels.append(label_text)

        # Create a ColumnDataSource object to store the data
        source = ColumnDataSource(data=dict(
            x=x_data,
            y=y_data,
            labels=labels
        ))

    if key == "result":
        pass
    if key == "artifact":
        pass

    # Set up the figure
    p = figure(title=scatter_plot_title,
               x_axis_label=x_axis_label,
               y_axis_label=y_axis_label,
               width=width, height=height,)
               # tools='pan,wheel_zoom,box_zoom,reset')

    # Create the scatter plot
    p.scatter(x='x', y='y', color='blue', source=source)

    labels = LabelSet(x='x', y='y', text='labels', text_color='black',
                      x_offset=0, y_offset=0, source=source)
    p.add_layout(labels)

    # Add a legend
    p.legend.location = "top_right"

    # Show the plot
    output_notebook()  # If using a Jupyter Notebook, otherwise omit this line
    show(p)


if __name__ == "__main__":
    pass
