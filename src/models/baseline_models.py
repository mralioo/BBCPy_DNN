import numpy as np
import pyriemann
import sklearn
from mne.decoding import CSP as CSP_MNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

import bbcpy.functions.helpers as helpers
from bbcpy.functions.artireject import AverageVariance
from bbcpy.functions.base import ImportFunc
from bbcpy.functions.spatial import CSP, MBCSP
from bbcpy.functions.statistics import cov
from bbcpy.pipeline import make_pipeline


def classifier_pipeline(steps_config):
    steps = []
    # for step_dict in steps_config: # if list
    for step_name, step_config in steps_config.items():
        if step_name == "FeatureExtraction":
            for algo_name, algo_config in step_config.items():

                if algo_name == 'CSP' and algo_config.applied == True:

                    csp_step = CSP(n_cmps=algo_config.n_cmps,
                                   excllev=algo_config.excllev,
                                   estimator=algo_config.estimator,
                                   scoring=helpers.evscoring_medvar,
                                   select=helpers.evselect_directorscut)

                    steps.append(csp_step)

                if algo_name == 'CSP-MNE' and algo_config.applied == True:
                    csp_mne_step = CSP_MNE(n_components=algo_config.n_cmps,
                                           reg=algo_config.reg,
                                           log=algo_config.log,
                                           cov_est=algo_config.cov_est,
                                           transform_into=algo_config.transform_into,
                                           norm_trace=algo_config.norm_trace,
                                           cov_method_params=algo_config.cov_method_params,
                                           rank=algo_config.rank,
                                           component_order=algo_config.component_order)
                    steps.append(csp_mne_step)

                if algo_name == 'MBCSP' and algo_config.applied == True:
                    mbcsp_step = MBCSP(n_cmps=algo_config.n_cmps,
                                       excllev=algo_config.excllev,
                                       estimator=algo_config.estimator,
                                       scoring=helpers.evscoring_medvar,
                                       select=helpers.evselect_directorscut)
                    steps.append(mbcsp_step)

                if algo_name == 'AverageVariance' and algo_config.applied == True:
                    avg_var_step = AverageVariance(excllev=algo_config.excllev,
                                                   estimator=algo_config.estimator)
                    steps.append(avg_var_step)

                if algo_name == 'Covariance' and algo_config.applied == True:
                    cov_step = ImportFunc(cov, estimator=algo_config.estimator, axis=algo_config.axis)
                    steps.append(cov_step)

        elif step_name == 'Transformation':
            for algo_name, algo_config in step_config.items():

                if algo_name == 'log' and algo_config.applied == True:
                    log_step = np.log
                    steps.append(log_step)

                elif algo_name == 'var' and algo_config.applied == True:
                    var_step = ImportFunc(np.var, axis=algo_config.axis)
                    steps.append(var_step)

                elif algo_name == 'tangent_space' and algo_config.applied == True:
                    tangent_space_step = pyriemann.tangentspace.TangentSpace(metric=algo_config.metric)
                    steps.append(tangent_space_step)

        elif step_name == 'Classification':
            for algo_name, algo_config in step_config.items():

                if algo_name == 'LDA' and algo_config.applied == True:
                    lda_step = LDA(solver=algo_config.solver,
                                   shrinkage=algo_config.shrinkage,
                                   priors=algo_config.priors,
                                   n_components=algo_config.n_components,
                                   store_covariance=algo_config.store_covariance,
                                   tol=algo_config.tol)
                    steps.append(lda_step)

                elif algo_name == 'SVC-pyriemann' and algo_config.applied == True:

                    svc_step = pyriemann.classification.SVC(metric=algo_config.metric,
                                                            class_weight=algo_config.class_weight,
                                                            probability=algo_config.probability,
                                                            kernel_fct=algo_config.kernel_fct,
                                                            Cref=algo_config.Cref,
                                                            C=algo_config.C,
                                                            shrinking=algo_config.shrinking,
                                                            tol=algo_config.tol,
                                                            cache_size=algo_config.cache_size,
                                                            verbose=algo_config.verbose,
                                                            max_iter=algo_config.max_iter,
                                                            decision_function_shape=algo_config.decision_function_shape,
                                                            break_ties=algo_config.break_ties,
                                                            random_state=algo_config.random_state,
                                                            )
                    steps.append(svc_step)

                elif algo_name == 'SVC-sklearn' and algo_config.applied == True:

                    svc_step = sklearn.svm.SVC(class_weight=algo_config.class_weight,
                                               C=algo_config.C,
                                               kernel=algo_config.kernel,
                                               degree=algo_config.degree,
                                               gamma=algo_config.gamma,
                                               coef0=algo_config.coef0,
                                               shrinking=algo_config.shrinking,
                                               probability=algo_config.probability,
                                               tol=algo_config.tol,
                                               cache_size=algo_config.cache_size,
                                               verbose=algo_config.verbose,
                                               max_iter=algo_config.max_iter,
                                               decision_function_shape=algo_config.decision_function_shape,
                                               break_ties=algo_config.break_ties,
                                               random_state=algo_config.random_state,
                                               )
                    steps.append(svc_step)

                elif algo_name == 'MDM' and algo_config.applied == True:
                    mdm_step = pyriemann.classification.MDM(metric=algo_config.metric)
                    steps.append(mdm_step)

    return make_pipeline(*steps)
