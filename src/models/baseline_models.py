# from bbcpy.pipeline import make_pipeline
import numpy as np
# from bbcpy.functions.base import ImportFunc
# from bbcpy.functions.spatial import CSP, MBCSP
# from bbcpy.functions.artireject import AverageVariance
# from bbcpy.functions.statistics import cov
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from bbcpy.functions.statistics import cov
import sklearn
import pyriemann
import bbcpy


# from bbcpy import functions


def classifier_pipeline(steps_config):
    steps = []
    # for step_dict in steps_config: # if list
    for step_name, step_config in steps_config.items():
        if step_name == "FeatureExtraction":
            for algo_name, algo_config in step_config.items():

                if algo_name == 'CSP' and algo_config.applied == True:
                    csp_step = bbcpy.functions.spatial.CSP(n_cmps=algo_config.n_cmps,
                                                           excllev=algo_config.excllev,
                                                           estimator=algo_config.estimator,
                                                           scoring=bbcpy.functions.helpers.evscoring_medvar,
                                                           select=bbcpy.functions.helpers.evselect_directorscut)

                    steps.append(csp_step)

                if algo_name == 'MBCSP' and algo_config.applied == True:
                    mbcsp_step = bbcpy.functions.spatial.MBCSP(n_cmps=algo_config.n_cmps,
                                                               excllev=algo_config.excllev,
                                                               estimator=algo_config.estimator,
                                                               scoring=bbcpy.functions.helpers.evscoring_medvar,
                                                               select=bbcpy.functions.helpers.evselect_directorscut)
                    steps.append(mbcsp_step)

                if algo_name == 'AverageVariance' and algo_config.applied == True:
                    avg_var_step = bbcpy.functions.artireject.AverageVariance(excllev=algo_config.excllev,
                                                                              estimator=algo_config.estimator)
                    steps.append(avg_var_step)

                if algo_name == 'Covariance' and algo_config.applied == True:
                    cov_step = bbcpy.functions.base.ImportFunc(cov, estimator=algo_config.estimator)
                    steps.append(cov_step)

        elif step_name == 'Transformation':
            for algo_name, algo_config in step_config.items():
                if algo_name == 'log' and algo_config.applied == True:
                    log_step = np.log
                    steps.append(log_step)

                elif algo_name == 'var' and algo_config.applied == True:
                    var_step = bbcpy.functions.base.ImportFunc(np.var, axis=algo_config.axis)
                    steps.append(var_step)

                elif algo_name == 'tangent_space' and algo_config.applied == True:
                    tangent_space_step = pyriemann.tangentspace.TangentSpace()
                    steps.append(tangent_space_step)

        elif step_name == 'Classification':
            for algo_name, algo_config in step_config.items():
                if algo_name == 'LDA' and algo_config.applied == True:
                    lda_step = LDA(solver=algo_config.solver)
                    steps.append(lda_step)
                elif algo_name == 'SVC-pyriemann' and algo_config.applied == True:
                    svc_step = pyriemann.classification.SVC(metric='logeuclid')
                    steps.append(svc_step)

                elif algo_name == 'SVC-sklearn' and algo_config.applied == True:
                    svc_step = sklearn.svm.SVC
                    steps.append(svc_step)

    return bbcpy.pipeline.make_pipeline(*steps)
