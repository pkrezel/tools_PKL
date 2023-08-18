#! /usr/bin/env python
# -*- coding: UTF-8 -*-
import logging
logger = logging.getLogger(__name__)
# =============================================================================


def fuse_LM_pickle(
        Lfile,
        fout=None,  # if fout is given, the result is saved
        axis=0,
    ):
    """
    Get a numpy array from the fusion of a list of pickle containing numpy arrays of the same size along an axis.
    """
    if type(Lfile) is str:
        Lfile = get_Lfile(Lfile)
    M = load_var(Lfile[0])
    for file in Lfile[1:]:
        M2 = load_var(file)
        M = np.append(M, M2, axis=axis)
    if fout is None:
        return M
    else:
        dump_var(M, fout)
        del(M)
