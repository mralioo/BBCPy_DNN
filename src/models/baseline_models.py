from bbcpy.pipeline import make_pipeline
import bbcpy.functions.helpers as helpers
from bbcpy.functions.base import ImportFunc
from bbcpy.functions.spatial import CSP, MBCSP
from bbcpy.functions.artireject import AverageVariance
from bbcpy.functions.statistics import cov
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from pyriemann.utils import covariance
import numpy as np


# def csp_pipeline(steps_config):
#     """Creates a pipeline with all the preprocessing steps specified in `steps_config`, ordered in a sequential manner
#
#     Args:
#         steps_config (DictConfig): the config containing the instructions for
#                                     creating the feature selectors or transformers
#
#     Returns:
#         [sklearn.pipeline.Pipeline]: a pipeline with all the preprocessing steps, in a sequential manner
#     """
#     steps = []
#
#     for step_config in steps_config:
#         # retrieve the name and parameter dictionary of the current steps
#
#         for k,v in step_config.items():
#             steps.append(v)
#         # step_name, step_params = step_config.items()[0]
#
#         # instantiate the pipeline step, and append to the list of steps
#
#
#
#     return make_pipeline(*steps)


def csp_pipeline(steps_config):

    for step_dict in steps_config:
        for step_name,step_config in step_dict.items():
            if step_name == 'CSP':
                csp_step = CSP(excllev=step_config.excllev,
                               estimator=step_config.estimator,
                               scoring=helpers.evscoring_medvar,
                               select=helpers.evselect_directorscut)
                # steps.append(csp_step)
            elif step_name == 'var':
                var_step = ImportFunc(np.var, axis=step_config.axis)
                # steps.append(var_step)
    log_step = np.log
    lda_step = LDA()

    steps = [csp_step, var_step, log_step, lda_step]
    return make_pipeline(*steps)

def mbcsp_pipeline(steps_config):

    for step_dict in steps_config:
        for step_name,step_config in step_dict.items():
            if step_name == 'CSP':
                csp_step = CSP(excllev=step_config.excllev,
                               estimator=step_config.estimator,
                               scoring=helpers.evscoring_medvar,
                               select=helpers.evselect_directorscut)
                # steps.append(csp_step)
            elif step_name == 'var':
                var_step = ImportFunc(np.var, axis=step_config.axis)
                # steps.append(var_step)
    log_step = np.log
    lda_step = LDA()

    steps = [csp_step, var_step, log_step, lda_step]
    return make_pipeline(*steps)


def pyriemann_pipeline(steps_config):

    for step_dict in steps_config:
        for step_name,step_config in step_dict.items():
            if step_name == 'CSP':
                csp_step = CSP(excllev=step_config.excllev,
                               estimator=step_config.estimator,
                               scoring=helpers.evscoring_medvar,
                               select=helpers.evselect_directorscut)
                # steps.append(csp_step)
            elif step_name == 'var':
                var_step = ImportFunc(np.var, axis=step_config.axis)
                # steps.append(var_step)
    log_step = np.log
    lda_step = LDA()

    steps = [csp_step, var_step, log_step, lda_step]
    return make_pipeline(*steps)