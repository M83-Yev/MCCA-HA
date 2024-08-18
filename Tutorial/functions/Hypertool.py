import numpy as np


def procrustes(source, target):
    def fit(source, target):
        datas = (source, target)

        ##-------------- STEP 1: Normalize ---------------##
        ssqs = [np.sum(d**2, axis=0) for d in datas]
        norms = [ np.sqrt(np.sum(ssq)) for ssq in ssqs ]
        normed = [ data/norm for (data, norm) in zip(datas, norms) ]
        source, target = normed

        ##------ STEP 2: Calculate optimal rotation ------##
        U, s, Vh = np.linalg.svd(np.dot(target.T, source),
                                 full_matrices=False)
        T = np.dot(Vh.T, U.T)

        ##---------------- STEP 3: Scaling ---------------##
        ss = sum(s)
        scale = ss * norms[1] / norms[0]
        proj = scale * T

        return proj

    ##------------- STEP 4: Transformation -----------##
    proj = fit(source, target)
    return np.dot(source, proj)


def HyperAlign(data):

    ##----------- STEP 1: MAKE TEMPLATE -----------##
    # make preliminary template from subject 1 and adjust it
    template = np.copy(data[0])
    for x in range(1, len(data)):
        next = procrustes(data[x], template/(x))
        template += next
    template /= len(data)

    ##-------- STEP 2: NEW COMMON TEMPLATE --------##
    # align each subj to the template from STEP 1
    new_template = np.zeros(template.shape)
    for x in range(0, len(data)):
        next = procrustes(data[x], template)
        new_template += next
    new_template /= len(data)

    ##---- STEP 3: HYPER-ALIGN TO NEW TEMPLATE ----##
    # align each subj to the Final template from STEP 2
    aligned = [np.zeros(new_template.shape)] * len(data)
    for x in range(0, len(data)):
        next = procrustes(data[x], new_template)
        aligned[x] = next

    # data > Template > aligned data
    return aligned, new_template

