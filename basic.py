#! /usr/bin/env python
# -*- coding: UTF-8 -*-
import logging
logger = logging.getLogger(__name__)
# ==============================================================================
import os
import glob
from copy import deepcopy
from distutils.dir_util import copy_tree
from datetime import date, datetime, timedelta
import gc
from glob import glob
import numpy as np
from numba import njit
import shutil
import sys
import traceback
import pickle
pickle.HIGHEST_PROTOCOL = 5
import pandas as pd
import random
import re
import subprocess
from subprocess import call
import inspect
import math
import threading
from copy import copy
from difflib import Differ
from collections import deque, Counter, OrderedDict
import itertools
from itertools import permutations, combinations, product, chain
from shutil import copyfile
import collections
import json
#import xlsxwriter
from io import BytesIO
import urllib
import time
import string
import gzip
#from pdfminer.high_level import extract_pages
#from pdfminer.layout import LTTextBoxHorizontal
# ==============================================================================
import urllib
import parmap
from collections import OrderedDict
# ==============================================================================
try:
    import thread
except ImportError:
    import _thread as thread
# ==============================================================================


def get_Ls_rand(
        nLs=10, # number of strings
        ns=3,  # number of caracters in each string
        sep="_",
        method="v1",
        nmin=0,
        nmax=100,
        n0=10,
    ):
    logger.info("get_Ls_rand")
    
    def get_Ls_rand_v1():
        """
        Gets a random list of strings.
        """
        logger.info("get_Ls_rand_v1")
        Ls = []
        for _ in range(nLs):
            Ls.append(sep.join(random.choices(Lalpha, k=ns)))
        return Ls
        
    
    def get_Ls_rand_v2():
        """
        Generates a list of random string based on the function get_Lrand.
        """
        logger.info("get_Ls_rand_v2")
        LL = get_LLrand(
                        nLL=nLs,
                        nmin=nmin,
                        nmax=nmax,
                        nL=n)
        return [sep.join([f"{n0:0>6}" for n in T[0]]) for T in zip(LL)]
    
    return eval(f"get_Ls_rand_{method}()")
        
                
    
def get_LLrand(
        nLL=4,
        nmin=0,
        nmax=100,
        nL=10,
    ):
    """
    Generates a list of list of random number based on get_Lrand.
    """
    LL =  []
    for _ in range(nLL):
        LL.append(get_Lrand(nL,  nmin=nmin, nmax=nmax))
    return LL
    

def get_txt_without_duplicates(
        txt,
        fout=None,
    ):
    """
    Get a file txt without duplicates of lines.
    """
    bash(f"sort -us  {txt}")



def get_LL_sorted_by_size(
        LL,
        reverse=False,
    ):
    return sorted(LL, key=lambda x:len(x), reverse=reverse)

def split_txt_in_Ltxt(
    txt, # file to split
    n=1000, # number of lines
    ):
    """
    split a txt file in Ltxt file of n lines.
    """
    txt2 = txt.split(".")[0]
    return bashr(f"split -l {n} {txt} {txt2}_x_")


def create_txt_without_duplicates_on_zone(
        txt,
        ideb=0,
        iend=6,
    ):
    """
    """
    #bash(f"awk '!seen[$1]++' {txt}")
    #bash(f"awk -F:'!seen[^.{{{ideb},{iend}}}]' {txt}", info=True)
    #AFINIR
    pass
    



def create_csv_without_duplicate(
        csv,
        i=0, # index not to duplicate
        with_header_in=None,
        with_header_out=None,
        sep=",",
    ):
    """
    Creates a new csv without duplicates at i position
    """

    df = load_df_from_csv(
                            csv, 
                            header=with_header_in,
                            sep=sep,
                            )
    df.drop_duplicates(i, inplace=True)
    remove_file(csv)
    create_csv_from_df(
                            df,
                            csv, 
                            with_header=with_header_out,
                    )


def create_txt_without_duplicate(
        txt,
    ):
    """
    Creates a new csv without duplicates at i position
    """

    Ls =load_Ls_from_txt(txt)
    df = get_df_from_L(Ls)
    df.drop_duplicates("x", inplace=True)
    remove_file(txt)
    create_txt_from_L(df["x"], txt)



def get_nLs_from_txt(
        txt,
    ):
    """
    gives the number of lines ia txt file
    """
    return int(bashr(f"wc -l {txt}").split(" ")[0])
    

def extract_txt_from_txt(
        txt=None,
        txt_new=None,
        LT=None,
        sep=",",
    ):
    logger.info("extract_txt_from_txt")
    with open(txt, "r") as fr:
        with open(txt_new, "w") as fw:
            for line in fr:
                Ls2 = line[:-1].split(sep)
                ya = True
                for T in LT:
                    if Ls2[T[0]] != T[1]:
                        ya = False
                        break
                if ya:
                    fw.write(line)


def get_L_without_None(
        L,
    ):
    return [x for x in L if x is not None]

def get_Tn_from_s(
        s,
        sep="_",
    ):
    """
    Converts a string s in a tuple of int.
    1_2_3_4 => (1,2,3,4)
    """
    return tuple([int(s2) for s2 in s.split(sep)])



def get_nNone_from_L(
        L,
    ):
    return len([x for x in L if x is None])


def get_s_from_L(
        L,
        sep="_",
    ):
    """
    Converts a list in a string
    Ex:(1,2,3) => 1_2_3
    """
    Ls = [str(x) for x in L]
    return sep.join(Ls)

def get_LTcombination(
        L,
        k=None,
        kmax=None,
    ):
    if type(L) is int:
        L = list(range(L))
    if k is not None:
        return list(combinations(L,k))
    if kmax is None:
        kmax= len(L)
    LT = []
    for k2 in range(1,kmax+1):
        LT += list(combinations(L,k2))
    return LT

def get_L_without_repetition(
        L,
    ):
    x2 = L[0]
    L2 = [x2]
    for x in L[1:]:
        if x == x2:
            continue
        else:
            L2.append(x)
            x2 = x

    return L2
    

def split_df_in_Ldf(
        df,
        nLdf=2,
    ):
    """
    split a dataframe (df) in a list of df.
    """
    return [df.loc[Lid1] for Lid1 in np.array_split(df.index,nLdf)]


def remove_Lsfile(
        s,
        path=".",
    ):
    """
    remove all the subfiles in path that have s in their name.
    ex: find pdb_prepared/ -name dirDLmol_F2D* -exec rm -rf {} \;
    """
    bashr(f"find {path}/ -name {s} -exec rm -f {{}} \;", info=True)



    
    
# def get_df_from_LD(
#     LD, #list of dictionnaries
#     Lcolumn=None,
#     ):
#     logger.info("get_df_from_LD")
#     LL = []
#     for D in LD:
#         L = []
#         for name in Lcolumn:
#             L.append(eval(f"D[{name}]"))
#         LL.append(L)
#     return get_df_from_LL(
            
def get_D_from_df(
        df,
        Lcolumn=None, # list of column to use, the first will be the key,
        ordered=False,
        type_of_values=None,
    ):
    if Lcolumn:
        if len(Lcolumn) == 2:
            if type_of_values:
                Lvalues = [eval(f"{type_of_values}([x])") for x in  df[Lcolumn[1]]]
            else:
                Lvalues = df[Lcolumn[1]]
            return get_D_from_2L(
                                    df[Lcolumn[0]], 
                                    Lvalues, 
                                    ordered=ordered,
                                )
        else:
            logger.warning("Code to do")
            return
    else:
        return df.to_dict('index')


def get_df_from_L(
            L,
            name="x",
            ):
    return pd.DataFrame(L, columns=[name])


def load_df_from_json(
    file,
    nmax_line=None,
    ):
    """
    get a dataframe from a json file
    """
    L = []
    if nmax_line :
        n_line = 0
        with open(file, 'r') as fr:
            for line in fr:
                L.append(json.loads(line))
                n_line += 1
                if n_line == nmax_line: 
                    break 
    else:
        with open(file, 'r') as fr:
            for line in fr:
                L.append(json.loads(line)) 
    return pd.DataFrame(L)

get_df_from_json = load_df_from_json


def pop_s_from_file(
    file,
    ):
    """
    removes the first line of a file and returns it.
    """

    with open(file, 'r+') as f: # open file in read / write mode
        firstLine = f.readline() # read the first line and throw it out
        data = f.read() # read the rest
        f.seek(0) # set the cursor to the top of the file
        f.write(data) # write the data back
        f.truncate() # set the file size to the current size
        return firstLine[:-1]


def get_Ls_upper(Ls):
    return [s.upper() for s in Ls]

def get_Ls_lower(Ls):
    return [s.lower() for s in Ls]

def get_diff_bw_L(
    L1,
    L2,
    ):
    """
    L1 = [10, 15, 20, 25, 30, 35, 40, 25]
    L2 = [25, 40, 35]

    > [10, 15, 20, 25, 30]
    """
    counter1 = Counter(L1)
    counter2 = Counter(L2)
    counter3 = counter1 - counter2
    return list(counter3.elements())
    

def fuse_ST(
        ST,
    ):
    LT = set(ST)
    return set(fuse_LL(LT))



def get_df_sorted(
        df,
        Lic=None, # list of index of columns
        Lir=None, # list of index of rows
        axis=0,
        ascending=True,
    ):
    logger.info("get_df_sorted")
    if Lic:
        return df.sort_values(
                                by=Lic, 
                                axis=0,
                                ascending=ascending,
                                )
    elif Lir:
        return df.sort_values(
                                by=Lir, 
                                axis=1,
                                ascending=ascending,
                                )


@njit
def get_Vrand_with_sum(
        n,
        sum,
    ):
    V = np.zeros(n)
    for i in range(sum):
        i2 = np.random.randint(n)
        V[i2] += 1
    return V.astype(np.int8)

@njit
def get_Mrand_with_sum(
        size=(10,5),
        sum=20,
    ):
    n,m = size
    M = np.zeros(size)
    for j in range(m):
        for i in range(sum):
            i2 = np.random.randint(n)
            M[i2,j] += 1
    return M.astype(np.int8)


def get_Jaccard_sim_bw_S(
        S1,
        S2,
        niv_log=1,
    ):
    logger.debug("\t"*niv_log + "get_Jaccard_sim_bw_S")
    # --------------------------------------------------
    return len(S1 & S2)/ len(S1 | S2)


def add_to_L(
        var,
        fname,
        niv_log=1,
    ):
    logger.debug("\t"*niv_log + "add_to_L")
    # --------------------------------------------------
    if yapa(fname):
        L = [var]
        dump_var(L, fname)
    else:
        L = load_var(fname)
        L.append(var)
        dump_var(L, fname)
        
def add_to_S(
        var,
        fname,
    ):
    if yapa(fname):
        S = set(var)
        dump_var(L, fname)
    else:
        S = load_var(fname)
        S.add(var)
        dump_var(S, fname)    

def add_to_txt(
    txt,
    s, #string to add
    ):
    with open(txt, "a") as fw:
        fw.write(s+"\n")

def get_Ldir_alphabet(
        repeat=2,
        lower_case=False,
    ):
    La = get_La_alphabet(lower_case=lower_case)
    LT = list(product(La, repeat=repeat))
    Ldir = ["".join(T) for T in LT]
    return Ldir


def get_La_alphabet(
        lower_case=False
    ):
    s = string.ascii_lowercase
    if lower_case:
        return list(s)
    else:
        return list(s.upper())
    

def get_date():
    x = str(datetime.now())
    return x.split(" ")[0]



def split_Ls_in_LLs(
        Ls,
        na=6,
    ):
    """
    Convert a list of string in a list of list of string that begin all by the same na caracters.
    """
    Ls2 = sorted(Ls)
    LLs = []
    L = [Ls2[0]]
    ref1 = Ls2[0][:na]
    for s in Ls2:
        ref2 = s[:na]
        if ref2 != ref1:
            LLs.append(L)
            L = [s]
            ref1 = ref2
        else:
            L.append(s)
    LLs.append(L)
    return LLs



def get_nL_from_LL(
        LL
    ):
    nL = 0
    for L in LL:
        nL += len(L)
    return nL


def get_Lfile_modifed_after(
        fname,
        nmax_old=1,
        unit_time="days",
    ):
    Lfile = get_Lfile(fname)
    time_before = get_time_before(
                                        nmax_old=nmax_old,
                                        unit_time=unit_time,
                                        )
    Lfile2 = [file for file in Lfile if is_file_modified_after(file, time_before=time_before)]
    return Lfile


              
def get_time_before(
        nmax_old=1,
        unit_time="days",  # weeks / days / hours / minutes / seconds / microseconds/
        ):
    dtime = exec(f"timedelta({unit_time} = {nmax_old})")
    return datetime.now() - dtime


def is_file_modified_after(
        file,
        nmax_old=1,
        unit_time="days",
        time_before=None,
        ):
    logger.debug("is_file_read_after")
    if time_before is None:
        time_before = get_time_before(
                                        nmax_old=nmax_old,
                                        unit_time=unit_time,
                                        )
    logger.debug(f"\t time_before:{time_before}")
    time_file_last_modif = datetime.fromtimestamp(os.path.getmtime(file))
    if time_file_last_modif <= time_before:
        return False
    return True

def is_file_read_after(
        fname,
        nmax_old=1,
        unit_time="days",
        time_before=None,
        ):
    logger.debug("is_file_read_after")
    if time_before is None:
        time_before = get_time_before(
                                        nmax_old=nmax_old,
                                        unit_time=unit_time,
                                        )
    logger.debug(f"\t time_before:{time_before}")
    time_file_last_read = datetime.fromtimestamp(os.stat(Lfile[0]).st_atime)
    if time_file_last_read <= time_before:
        return False
    return True



def extract_s_from_s_with_regex(
        s,
        regx,
        n=0,
    ):
    """
    r">(.*)<" : to extract s bw > <, n=1 
    """
    logger.debug("extract_s_from_s_with_regx")
    res = re.search(regx, s)
    if res is not None:
        return res.group(n)
    else:
        logger.debug("\t no string that match the regx")
        return ""


def get_Li_overlap_from_Ln(
        Ln,
        size=15,
        overlap=5,
    ):
    nLn = len(Ln)
    Li = []
    Vi = np.arange(overlap)
    Vi += (size - overlap)
    Li =  list(Vi)
    while True:
        Vi += (size - overlap)
        if Vi[-1] >= nLn:
            break
        Li += list(Vi)
    return Li


def get_Li_unoverlap_from_Ln(
        Ln,
        size=15,
        overlap=5,
    ):
    """
    """
    nLn = len(Ln)
    Li = get_Li_overlap_from_Ln(
                                Ln,
                                size=size,
                                overlap=overlap,
                                )
    return sorted(list(set(range(nLn))-set(Li)))

def get_LLn_from_Ln_by_overlap(
        Ln,
        size=15,
        overlap=5,
    ):
    """
    Gets list of list of integer from Ln that overlap
    """
    nLn = len(Ln)
    LLn = []
    i1=0
    i2 = size
    LLn.append(Ln[i1:i2])
    while True:
        i1 = i2 - overlap
        if i1 >= nLn:
            break
        i2 = i1 + size
        if i2 > nLn:
            LLn.append(Ln[i1:nLn])
            break
        LLn.append(Ln[i1:i2])
    return LLn


def get_LLn_from_Ln_by_random_overlap(
        Ln,
        size_min=6,
        size_max=18,
        overlap_min=0,
        overlap_max=0,
    ):
    nLn = len(Ln)
    LLn = []
    i1=0
    size1 = random.choice(list(range(size_min, size_max)))
    i2 = size1
    LLn.append(Ln[i1:i2])
    while True:
        if overlap_max != 0:
            overlap = random.choice(list(range(overlap_min, overlap_max)))
        else:
            overlap = 0
        size2 = random.choice(list(range(size_min, size_max)))
        i1 = i2 - overlap
        if i1 >= nLn:
            break
        i2 = i1 + size2
        if i2 > nLn:
            LLn.append(Ln[i1:nLn])
            break
        LLn.append(Ln[i1:i2])
    return LLn
        
        
    


def get_s_without_r_at_extremity(
        s,
        r
    ):
    logger.debug("get_s_without_r_at_extremity")
    i1 = 0
    ns= len(s)
    while s[i1] == r:
        i1 += 1
        if i1 == ns:
            return ""
    i2 = len(s) -1
    while s[i2] == r:
        i2 -= 1
    logger.debug(f"\t i1={i1}, i2={i2}")
    return s[i1:i2+1]


def get_Le_from_LLn(
        LLn,
        with_cycle=False,
    ):
    Se = set()
    for Ln in LLn:
        Le = get_Le_from_Ln(
                                Ln,
                                with_cycle=with_cycle,
                            )
        Se |= set(Le)
    return list(Se)
    

def get_Le_from_Ln(
        Ln,
        with_cycle=False,
    ):
    """
    Get a list of edges from a list of index
    """
    logger.debug("get_Le_from_Ln")
    Le = []
    nLn = len(Ln) -1
    i = 0
    while True:
        Le.append((Ln[i], Ln[i+1]))
        i += 1
        if i == nLn:
            break
    if with_cycle:
        Le.append((Ln[-1], Ln[0]))
    return Le

def get_LLrand_from_L(
        L,
        n=20,
        di=10,
        ):
    nL = len(L)
    Li = list(range(nL-di))
    LL = []
    for _ in range(n):
        i = random.choice(Li)
        LL.append(L[i: i+di])
    return LL


def get_Ln_with_random_0_from_Ln(
        Ln,
        n_0=2,
    ):
    Ln2 = Ln.copy()
    nLn = len(Ln)
    Li = random.sample(list(range(nLn)), n_0)
    for i in Li:
        Ln2[i] = 0
    return Ln2


def get_s_from_Ln(
        Ln,
        Ln_zero=None,
        sep="_"
        ):
    if Ln_zero is None:
        Lr = [str(n) for n in Ln]
    else:
        Lr  = [str(n)  if n not in Ln_zero else "_" for n in Ln]
    return sep.join(Lr)


def get_s_from_Lv(
        Ln,
        n_min=50,
        n_max=None,
        r_low="x",
        r_OK=" ",
        r_high=" ",
        ):
    """
    """
    Lr = []
    if (n_min is not None) and (n_max is None):
        for n in Ln: 
            if n < n_min:
                Lr.append(r_low)
            else:
                Lr.append(r_OK)
    elif (n_min is None) and (n_max is not None):
        for n in Ln: 
            if n > n_max:
                Lr.append(r_high)
            else:
                Lr.append(r_OK)
    else:
        for n in Ln: 
            if n < n_min:
                Lr.append(r_low)
            elif n < n_max:
                Lr.append(r_OK) 
            else:
                Lr.append(r_high)
    return "".join(Lr)
    

def get_permutation_from_L_given_index(
        L, 
        apermindex,
    ):
    """
    generate a permutation for each value of apermindex/
    """
    nL = len(L)
    for i in range(nL-1):
        apermindex, j = divmod(apermindex, nL-i)
        L[i], L[i+j] = L[i+j], L[i]
    return L



def get_Lc_from_L(
        L
    ):
    """
    if L = [1,2,3]
    returns [(1,2), (2,3)]
    """
    nL = len(L)
    return [(L[i], L[i+1]) for i in range(nL-1)]

def get_Ln_from_Lr(
        Lr,
    ):
    """
    Convert a list of character into a list of integer.
    """
    return [ord(r) for r in Lr]


def get_LV_from_Ls(
        Ls,
        Lr_zero=["_"],
    ):
    LV = []
    for s in Ls:
        LV.append(np.array(get_Ln_from_s(s,Lr_zero)))
    return LV


def get_Ls_from_Mpn(
        Mpn, # matrix of point with integers
        sep="_",
    ):
    LLn = Mpn.tolist()
    return get_Ls_from_LLn(LLn)

def get_Ls_from_LLn(
        LLn,
        sep="_",
    ):
    return [sep.join([str(n) for n in Ln]) for Ln in LLn]


def get_LLn_from_Ls(
        Ls,
        Lr_zero=None,
        sep=None,
    ):
    return [get_Ln_from_s(
                            s,
                            Lr_zero=None,
                            sep=None,
                        ) for s in Ls]


def get_Ln_from_s(
        s,
        Lr_zero=None,
        sep=None,
    ):
    """
    Convert a string into a list of integer.
    """
    if sep is None:
        sep ="_"
    if Lr_zero is not None:
        return [ord(r) if r not in Lr_zero else 0 for r in s]
    else:
        if sep is None:
            sep ="_"
        return [int(s2) for s2 in s.split(sep)]


def get_Lr(
        n=10
    ):
    """
    Generate a list of characters.
    """
    n0=97
    return [chr(i) for i in range(n0,n0+n)]


def get_line_n_in_file(
        file,
        n,
        ):
    return bashr(f"sed -n '{n}p'< {file}")


def get_Ls_completed(
        Ls,
        index=1,
        sep="_",
        ):
    """
    For a list of strings whith an incremental integer, None are added
    where a value is missing.
    
    """
    Ls.sort()
    try:
        i = int(Ls[0].split(sep)[index])
    except:
        logger.warning("\t the index or the separator is not correct")
    Li = [int(s.split(sep)[index]) for s in Ls]
    Li2 = sorted(list(set(list(range(Li[-1]+1))) - set(Li)))
    Ls2 = Ls.copy()
    for i,i2 in enumerate(Li2):
        Ls2.insert(i2+i, None)
    return Ls2 
    
        


def split_file_in_Lfile(
            file,
            nmax_mol=1,                # number max of molecules per file
            nmax_file=1,
            path=None,
            ref="part",
            end=None,
            deb=None,
            ):
    """
    Create several files from a big file
    """
    logger.info("\t split_file_in_Lfile")
    #---------------------------------------------------------------------------
    n_file = 0
    n_mol= 0
    do = True
    Lfout = []
    if end:
        logger.info(f"\t use of indication of end :{end}")
        with open(file, "r") as fr:
            line = next(fr)
            while do:
                fout = file.replace(".", f"_{ref}_{n_file}.")
                Lfout.append(fout)
                with open(fout, "w") as fw:
                    while True:
                        try:
                            fw.write(line)
                            line = next(fr)
                            if line[:4] == end:
                                n_mol+= 1
                                if n_mol>= nmax_mol:
                                    line = next(fr)
                                    n_file+= 1
                                    if nmax_file:
                                        if n_file== nmax_file:
                                            logger.debug("\t end")
                                            return_fileLfout
                                    n_mol= 0
                                    break
                        except Exception:
                            do = False
                            logger.debug("\t end of file")
                            break
    elif deb:
        logger.info(f"\t use of indication of beginning :{deb}")
        with open(file, "r") as fr:
            while True:
                line = next(fr)
                if line[:4] == deb:
                    logger.info(f"\t first {deb} found")
                    break
            while do:
                fout = file.replace(".", f"_{ref}_{n_file}.")
                Lfout.append(fout)
                with open(fout, "w") as fw:
                    while True:
                        try:
                            if line[:4] == deb:
                                while True:
                                    fw.write(line)
                                    line = next(fr)
                                    if line[:4] == deb:
                                        n_mol+= 1
                                        if n_mol>= nmax_mol:
                                            n_file += 1
                                            if nmax_file:
                                                if n_file== nmax_file:
                                                    logger.debug("\t end")
                                                    return Lfout
                                            n_mol= 0
                                            break
                        except Exception:
                            do = False
                            logger.debug("\t end of file")
                            break
                        break
    return Lfout

def fuse_LM_pickle(
        Lfile,
        fout=None,
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
        

def note(
        fname="note.txt",
        s="",
        ):
    """
    Create a short message in a file
    """
    if ".txt" not in fname:
        fname += ".txt"
    bash(f"echo \"{s}\">{fname}")


def open_path():
    """
    open the current directory in nautilus.
    """
    os.system(f"nohup nautilus {get_path()} & ")
    

def concat_Lfile(
        Lfile,
        fname="all.txt",
        with_big_files=False,
        ):
    """
    Concatenate Lfile in fname.
    """
    logger.info("concat_Lfile")
    if type(Lfile) is str:
        Lfile = get_Lfile(Lfile)
    if with_big_files:
        logger.info("\t with big files")
        with open(fname, 'w') as fw:
            for file in Lfile:
                if ya(file):
                    with open(file) as fr:
                        for line in fr:
                            fw.write(line)  
                else:
                    logger.warning(f"\t No file {file}")
    else:
        logger.info("\t with normal files")
        with open(fname, 'w') as fw:
            for file in Lfile:
                if ya(file):
                    with open(file) as fr:
                        fw.write(fr.read())
                else:
                    logger.warning(f"\t No file {file}")
    




def get_int_from_string(s):
    """
   Gets a int unique from a s string.
    """
    return int.from_bytes(s.encode(), 'little')


def get_string_from_int(n):
    """
    get the unique string associated to the n integer
    """
    return n.to_bytes(math.ceil(n.bit_length() / 8), 'little').decode()


def alea():
    """
    Generates random boolean
    :return boolean
    """
    v = random.random()
    if v > 0.5:
        return True
    return False


def bash(
        s, 
        info=False,
        niv_log=0,
    ):
    """
    Executes a bash command and doesn't return the results.
    """
    inde = "\t" * niv_log
    if info:
        logger.info(f"{inde}{s}")
    res = subprocess.getoutput(s)


def bashr(
        s, 
        info=True,
        niv_log=0,
    ):
    """
    Executes a bash command and returns the results.
    """
    inde = "\t" * niv_log
    if info:
        logger.info(f"{inde}{s}")
    res =  subprocess.getoutput(s)
    # if info:
    #     logger.info(res)
    return res


def cat(
        fname, 
        nmax=None
    ):
    """
    
    """
    if nmax is None:
        with open(fname, "r") as fr:
            while True:
                try:
                    print(next(fr)[:-1])
                except StopIteration:
                    break
    else:
        with open(fname, "r") as fr:
            i = 0
            while True:
                try:
                    print(next(fr)[:-1])
                except StopIteration:
                    break
                i += 1
                if i == nmax:
                    break


def compare_files(fname1, fname2, fout=None):
    """
    Shows the differences between the 2 files.
    """
    logger.info("compare_files")
    # --------------------------------------------------------------------------
    if fout is None:
        with    open(fname1) as f1, \
                open(fname2) as f2:
            differ = Differ()
            for line in differ.compare(f1.readlines(), f2.readlines()):
                print(line, end="")
    else:
        with    open(fname1) as f1, \
                open(fname2) as f2, \
                open(fout, "w") as g3 :
            differ = Differ()
            for line in differ.compare(f1.readlines(), f2.readlines()):
                print(line, end="", file=f3)


def complete_L_with_last_value(L, N):
    """
    Complete a list up to N value with the last value.
    :param L: list
    :param N: integer
    """
    logger.debug("complete_list_with_last_value")
    logger.debug(f"\t L={L}")
    if len(L) >= N:
        L2 = L[:N]
    else:
        n = len(L)
        L2 = L + L[-1:] * (N - n)
    logger.debug(f"\t L2={L2}")
    return L2




def copy_dir_in_dir(
        dir1, # Name of the directory to copy
        dir2, # Name of the new directory
        ):
    """
    Copy of directory in a directory.
    """
    # if os.path.isdir(dir1):
    #     make_dir(dir2)
    #     Lx = glob(dir1 + '/*')
    #     for x in Lx:
    #         copy_dir(x, dir2 + '/' + x.split('/')[-1])
    # else:
    #     shutil.copy(dir1, dir2)
    #copy_tree(dir1, dir2)
    call(['cp', '-a', dir1, dir2]) # Linux
#

def copy_file_in_dir(
        file,
        dir
        ):
    return copy_Lfile_in_dir([file], dir)


def copy_into(
        fname=None,
        path=None, # Directory where to copy
        ):
    """
    General function to copy file and directory
    """
    logger.info("copy_into")
    # --------------------------------------------------------------------------
    if path is None:
        logger.warning("A path must be given!")
        return
    # --------------------------------------------------------------------------
    if os.path.isdir(fname):
        if "/" in fname:
            fname2 = fname.split("/")[-1]
            if fname2 == "":
                fname2 = fname.split("/")[-2]
        else:
            fname2 = fname
        shutil.copytree(fname, f"{path}/{name2}")
    elif os.path.isfile(fname):
        shutil.copy(fname, path)


def copy_Ldir_with_elements(
        path,               # path to the directory to copy
        dir_new,            # Name of the new directory in the current path
        Lfile_to_copy=[],   # List of file to copy
        Ldir_to_copy=[],    # List of dir to copy
        with_print=True,
        ):
    """
    copy a repertory by selecting a list of files and a list of directories
    to keep in each subdirectories.
    """
    logger.debug("copy_dir_with_list_of_files")
    # --------------------------------------------------------------------------
    if ya(dir_new):
        remove_dir(dir_new)
    os.mkdir(dir_new)
    # --------------------------------------------------------------------------
    Ldir = get_Ldir(path=path)
    Ldir  = [dir.split("/")[-1] for dir in Ldir]
    for i, dir in enumerate(Ldir):
        if with_print:
            print(f"copy_Ldir_with_elements {i:>6}", end="\r")
        os.mkdir(f"./{dir_new}/{dir}")
        for f in Lfile_to_copy:
            try:
                copyfile(f"./{path}/{dir}/{f}", f"./{dir_new}/{dir}/{f}")
            except FileNotFoundError:
                logger.info(f"{f} not in {dir}")
        for dir2 in Ldir_to_copy:
            try:
                copy_dir(f"./{path}/{dir}/{dir2}", f"./{dir_new}/{dir}/{dir2}")
            except FileNotFoundError:
                logger.info(f"{dir2} not in {dir}")


def copy_Lfile_from_Ldir_on_cluster(
        Ldir=None,
        Lfile=None,
        user="krezel",
        path_cluster=None,
        IP_cluster="192.168.11.17",
        with_print=True,
        ):
    nLdir = len(Ldir)
    for i,dir in enumerate(Ldir):
        if with_print:
            print(f"copy_Lfile_from_Ldir_on_cluster {i:>6}/{nLdir:<6}", end="\r")
        bash(f"mkdir  {dir} {user}@{IP_cluster}:{path_cluster}")
        for file in Lfile:
            bash(f"scp  {dir}/{file} {user}@{IP_cluster}:{path_cluster}/{dir}")


def copy_Lfile_in_dir(
        Lfile, # tuple if you want to change the name
        dir,
        ):
    if yapa(dir):
        os.makedirs(dir)
    for file in Lfile:
        if type(file) is tuple:
            file, file_new = file
        else:
            file_new = file
        copyfile(file, f"{dir}/{file_new}")


def copy_Lfile_from_dir_to_dir2(
        Lfile,
        dir,
        dir2=None,
        imax=None,
    ):
    path0 = get_path()
    if dir2 is None:
        dir2 = dir + "_new"
    if yapa(dir2):
        os.makedirs(dir2)
    if type(Lfile) is str:
        os.chdir(dir)
        Lfile = get_Lfile(Lfile)
        os.chdir("..")
    for i,file in enumerate(Lfile):
        if i == imax:
            break
        try:
            copyfile(f"{dir}/{file}", f"{dir2}/{file}")
        except FileNotFoundError:
            logger.warning(f"{file} not found!")

    os.chdir(path0)



def count_files_in_sdir(fname="toto", dir=None):
    """
    Get the number of files named fname.
    """
    if dir is None:
        return int(bash(f"find . -type f -name '{fname}'| wc -l"))
    else:
        return int(bash(f"find . -type f -name '{fname}'| grep '{dir}' | wc -l"))


def create_csv_from_df(
        df, 
        fout=None, 
        add_index=False,
        index_label=False,
        sep=",",
        with_header=True,
        ):
    """
    Gets a csv file from a dataframe.
    """
    if fout is None:
        fout = get_s_from_object(df) +".csv"
    elif fout[-4:] != ".csv":
        fout += ".csv"
    df.to_csv(
                fout, 
                index=add_index,
                index_label=index_label,
                sep=sep,
                header=with_header,
            )


def create_csv_from_LLs(
        LLs, 
        fout=None, 
        Ls_column=None, 
        sep=";", 
        add_header=True, 
        add_index=False,
        ):
    """
    Creates a cvs file from a list.
    :param L: list or dataframe,
    :param Lcolumn: list of headers
    """
    



def create_csv_from_L_with_pandas(
        L, 
        fout=None, 
        Lcolumn=None, 
        sep=";", 
        add_header=True, 
        add_index=False,
        ):
    """
    Creates a cvs file from a list.
    :param L: list or dataframe,
    :param Lcolumn: list of headers
    """
    logger.info("create_csv_from_L")
    # --------------------------------------------------------------------------
    if fout is None:
        fout = get_s_from_object(L) + ".csv"
    elif ".csv" not in fout:
        fout += ".csv"
    # --------------------------------------------------------------------------
    if Lcolumn is not None:
        df = pd.DataFrame(L, columns=Lcolumn)
    else:
        df = pd.DataFrame(L)
    # --------------------------------------------------------------------------
    df.to_csv(fout, sep=sep, index=add_index, header=add_header)


def create_json_file_from_D(D, fname=None):
    """
    Create a json file from a dictionnary..
    """
    if fname is None:
        fname = get_s_from_object(D) +".json"
    with open(fname, "w", encoding="utf-8") as fw:
        json.dump(D, fw, ensure_ascii=False, indent=4)


def create_Ldir_from_Lpdb(Lpdb=None):
    """
    Creates a list of directories from a list of id_PDB.
    """
    logger.info("create_Ldir_from_Lpdb")
    if Lpdb is None:
        Lpdb = get_Lfile("*.pdb")
        nLpdb = len(Lpdb)
        logger.info(f"\t {nLpdb} new pdb")
    Ldir2 = []
    for pdb in Lpdb:
        try:
            if ya(pdb[:-4]):
                remove_dir(pdb[:-4])
            os.makedirs(pdb[:-4])
            move_file(pdb, pdb[:-4])
            Ldir2.append(pdb[:-4])
            remove_file(pdb)
        except Exception:
            logger.info(traceback.format_exc())
    nLdir2 = len(Ldir2)
    logger.info(f"\t {nLdir2} new directories created.")
    return Ldir2


def create_new_file_from_Li(fname, Li, fout=None):
    """
    creates a new file by extracting the Li lines.
    """
    with open(fname, "r") as fp:
        Ls = fp.readlines()
    Ls = [s for i,s in enumerate(Ls) if i in Li]
    if fout is None:
        fout = fname.replace(".", "_extract.")
    create_txt_from_L(Ls, fout=txt)


def create_txt_from_D(
        D,
        fname="D_to_txt.txt"
        ):
    with open(fname, 'w') as file:
         file.write(str(D)) # use `json.loads` to do the reverse


def dump_L_in_txt(
        L,
        txt,
        info=False,
        line_sep=None, # to add line od separation
        mod=4
    ):
    """
    Add the list L in the txt file
    """
    create_txt_from_L(
                        L,
                        fout=txt,
                        info=info,
                        line_sep=line_sep, # to add line od separation
                        mod=mod,
                        state="a",
                        )


def create_txt_from_L(
        L,
        fout="xxxx.txt",
        info=False,
        line_sep=None, # to add line od separation
        mod=4,
        state="w",
        ):
    """
    Creates a text file from a list of string
    :param L:list of string,
    :returns: None.
    """
    if info:
        logger.info("create_txt_from_L")
    else:
        logger.debug("create_txt_from_L")
    # --------------------------------------------------------------------------
    txt = fout
    # --------------------------------------------------------------------------
    if len(L) == 0:
        touch(txt)
        return
    # --------------------------------------------------------------------------
    if "." not in txt:
        txt += ".txt"
   # --------------------------------------------------------------------------
    s_end = "\n"
    if type(L[0]) is str:
        if L[0][-1:] == "\n":
            logger.debug(f"\t terminaison OK")
            s_end = ""
    # --------------------------------------------------------------------------
    if line_sep is not None:
        with open(txt, state) as fw:
            for i,x in enumerate(L):
                if type(x) is tuple:
                    for x2 in x:
                        if type(x2) is tuple:
                            for x3 in x2:
                                fw.write(f"\t\t {x3}{s_end}")
                        else:
                            fw.write(f"\t {x2}{s_end}")
                    fw.write(f"{s_end}")
                else:
                    fw.write(f"{x}{s_end}")
                if (line_sep is not None) and (i%mod == mod-1):
                    fw.write(f"{line_sep}{s_end}")
    else:
        with open(txt, state) as fw:
            for x in L:
                fw.write(f"{x}{s_end}")




def create_txt_from_LL(
        LL,
        fout="xxxx.txt",
        info=False,
        line_sep=None,
        mod=4,
        ):
    """
    Creates a text file from a list of string
    :param L:list of string,
    :returns: None.
    """
    if info:
        logger.info("create_txt_from_LL")
    else:
        logger.debug("create_txt_from_LL")
    # --------------------------------------------------------------------------
    if len(LL) == 0:
        touch(fout)
        return
    # --------------------------------------------------------------------------
    if "." not in fout:
        fout += ".txt"
    if LL[0][-1:] == "\n":
        logger.debug(f"\t terminaison OK")
        s_end = ""
    else:
        s_end = "\n"
    # --------------------------------------------------------------------------
    LL = [[str(x) for x in L] for L in LL]
    # --------------------------------------------------------------------------
    with open(fout, "w") as fw:
        for L in LL:
            fw.write(f"{'='*40}{s_end}")
            for i,x in enumerate(L):
                fw.write(f"{x}{s_end}")
                if (line_sep is not None) and (i%mod == mod-1):
                    fw.write(f"{line_sep}{s_end}")
                

def create_txt_from_s(s, fname="test.txt"):
    """
    Creates a txt file from a string
    """
    if "." not in fname:
        fname += ".txt"
    with open(fname, "w") as fw:
        fw.write(s)


def create_txt_from_S(
        S,
        fout="xxxx.txt",
        info=False
        ):
    L = list(S)
    create_txt_from_L(
            L,
            fout=fout,
            info=info)

def create_xyz_from_Mp(Mp, fout="test"):
    """
    Create a xyz file that can be read by pymol for example.
    """
    Ls = [str(len(Mp)), "test"]
    for p in Mp:
        Ls.append(f" He{p[0]:>10.5f}{p[1]:>10.5f}{p[2]:>10.5f}")
    create_txt_from_L(Ls, f"{fout}.xyz")


def discriminate_Lc_with_Ln(
        Ln, # List of intergers
        Lc, # List of couples of integers
        ):
    """
    create a tuple of 3 tuples from a list of couples and a list of integers (Ln),
    with the couples inside, on the border and outside Ln
    :returns: tuple of 3 tuples: Tc_in, Tc_on, Tc_out
    """
    logger.debug("\n discriminate_Lc_with_Ln:")
    logger.debug(f"\t Ln={Ln}")
    logger.debug(f"\t Lc={Lc}")
    Lc_in = []
    Lc_on = []
    Lc_out = []
    Sn = set(Ln)
    for c in Lc:
        n = len(set(c) & Sn)
        if n == 2:
            Lc_in.append(c)
        elif n == 1:
            Lc_on.append(c)
        elif n == 0:
            Lc_out.append(c)
    logger.debug("\t ---------------")
    logger.debug(f"\t Lc_in={Lc_in}")
    logger.debug(f"\t Lc_on={Lc_on}")
    logger.debug(f"\t Lc_out={Lc_out}")
    return (tuple(Lc_in), tuple(Lc_on), tuepl(Lc_out))


def download_file(url, fname):
    """
    Download of a file from an url.
    """
    urllib.request.urlretrieve(url, fname)

    


def dump_Lvar(
        Lvar,
        fname="var",
        di=1,
        size=6, # size to write the number in the fname
        ):
    logger.info("dump_Lvar")
    Lfile = []
    for i,var in enumerate(Lvar):
        if type(var) is tuple:
            dump_var(var[0], var[1])
            Lfile.append( var[1])
        else:
            file = f"{fname}_{i*di:0>{size}}.p"
            dump_var(var, file)
            Lfile.append(file)
    return Lfile
            

def dump_var(
        var=None, 
        fname=None,
        concat=False,
        ):
    """
    record of var into a pickle
    :param var: string, variable to dump.
    :returns: creates a pickle file for the variable.
    """
    logger.debug("dump_var")
    if fname is None:
        fname = get_s_from_object(var)
    # -------------------------------------------------------------------------
    if ".p.gz" in fname:
        with gzip.open(fname, "wb") as fw:
            pickle.dump(var, fw)
        return fname
    # -------------------------------------------------------------------------
    path0 = get_path()
    if ".p" not in fname:
        fname += ".p"
    # -------------------------------------------------------------------------
    if ".." in fname:
        L = fname.split("/")
        path = "/".join(L[:-1])
        fname = L[-1]
        os.chdir(path)
    # -------------------------------------------------------------------------
    if concat and ya(fname):
        var1 = deepcopy(var)
        var2 = load_var(fname)
        xtype = type(var2)
        if xtype is list:
            var1 += var2
        elif xtype is np.ndarray:
            var1 = fuse_LM([var1, var2], axis=0)
        with open(fname, "wb") as fw:
            pickle.dump(var1, fw)
    else:         
        with open(fname, "wb") as fw:
            pickle.dump(var, fw)
    # -------------------------------------------------------------------------
    os.chdir(path0)
    return fname


def extract_file_from_file(
        fname, 
        s_deb=None, 
        s_end=None, 
        fout=None,
        ):
    """
    Extract a file from a file, with s_deb and s_end as limits.
    """
    if fout is None:
        if "." in fname:
            L = fname.split(".")
            fout = L[0] + "_extract."+L[1]
        else:
            fout = fname + "_extract"
    # --------------------------------------------------------------------------
    with    open(fname, "r") as fr, \
            open(fout, "w") as fw:
        if s_deb is not None:
            n = len(s_deb)
            for line in fr:
                if line[:n] == s_deb:
                    break
        if s_end is not None:
            n = len(s_end)
            for line in fr:
                if line[:n] == s_end:
                    break
                fw.write(line)
        else:
            for line in fr:
                fw.write(line)
    return fout


def filter_D(D, func):
    """
    https://thispointer.com/python-filter-a-dictionary-by-conditions-on-keys-or-values/
    Iterate over all the key value pairs in dictionary and call the given
    callback function() on each pair. Items for which callback() returns True,
    add them to the new dictionary. In the end return the new dictionary.

    Filter a dictionary to keep elements only whose values are string of length 6
    :Ex newDict = filterTheDict(dictOfNames, lambda elem: len(elem[1]) == 6)
    """
    D2 = dict()
    # Iterate over all the items in dictionary
    for (key, value) in D.items():
        # Check if item satisfies the given condition then add to new dict
        if func((key, value)):
            D2[key] = value
    return D2


def find_first_line_with_s_from_file(
        fname, 
        s_deb=None, 
        s_end=None, 
        i0=0, 
        sep=" ",
        ):
    """
    Finds the first line into a file with s at position i0 after splitting.
    """
    ya = False
    with open(fname, "r") as fr:
        if s_end is None:
            for i, line in enumerate(fr):
                s1 = ssplit(line, sep)[i0]
                if s1 == s_deb:
                    ya = True
                    break
        else:
            for i, line in enumerate(fr):
                if line:
                    s1 = ssplit(line, sep)[i0]
                    if s1 == s_deb:
                        ya = True
                        break
                    elif s1 == s_end:
                        break
    if ya:
        return (i, line)
    else:
        return None


def fuse_2D(D1, D2):
    """Given two dictionaries, merge them into a new dict as a shallow copy."""
    D = D1.copy()
    D.update(D2)
    return D


def fuse_LD(
        LD, 
        with_mergedeep=False,
        info=False,
        ):
    """
    Fuses a list of dictionnaries into a new dictionnary.
    """
    logger.info("fuse_LD")
    from mergedeep import merge
    # ------------------------------------
    Dall = deepcopy(LD[0])
    for D in LD[1:]:
        merge(Dall,D)
    return Dall    
    


def fuse_LD_with_D(
        LD,
    ):
    """
    Fuses a list of dictionnaries into a new dictionnary.
    """
    logger.info("fuse_LD_with_D")
    Dall = deepcopy(LD[0])
    for D in LD[1:]:
        for k,D2 in D.items():
            if k in Dall.keys():
                Dall[k] = {**Dall[k], **D2}
            else:
                Dall[k] = D2
    return Dall

def fuse_LD_with_D_and_best_score(
        LD,
        score="score",
    ):
    """
    Fuses a list of dictionnaries into a new dictionnary.
    """
    logger.info("fuse_LD_with_D_and_best_score")
    Dall = deepcopy(LD[0])
    for D in LD[1:]:
        for k,D2 in D.items():
            if k in Dall.keys():
                score1 = Dall[k][score]
                score2 = D2[score]
                if score2 > score1:
                    Dall[k] = D2
            else:
                Dall[k] = D2
    return Dall


def fuse_LD_with_L(
            LD,           # list of directories to fuse
            info=False,   # to show what is done
    ):
    """
    Fuses a list of dictionnaries into a new dictionnary.
    """
    logger.debug("fuse_LD_with_L")
    Dall = deepcopy(LD[0])
    nLD = len(LD[:1])
    for i,D in enumerate(LD[1:]):
        if info:
            print(f"{i:>8}/{nLD:<8}", end="\r")
        for k,L in D.items():
            if k in Dall.keys():
                xtype = type(L)
                if xtype is list:
                    Dall[k] += L
                elif xtype is np.ndarray:
                    Dall[k] = fuse_LM([Dall[k], L], axis=0)
            else:
                Dall[k] = L
    return Dall

def fuse_LD_with_S(
            LD,           # list of directories to fuse
            info=False,   # to show what is done
    ):
    """
    Fuses a list of dictionnaries into a new dictionnary.
    """
    logger.info("fuse_LD_with_S")
    Dall = deepcopy(LD[0])
    nLD = len(LD[:1])
    for i,D in enumerate(LD[1:]):
        if info:
            print(f"{i:>8}/{nLD:<8}", end="\r")
        for k,S in D.items():
            if k in Dall.keys():
                Dall[k] |= S
            else:
                Dall[k] = S
    return Dall

def add_D2_to_D_with_S(
        D,
        D2,
    ):
    """
    If the values of D are set, D2 is added to D
    """
    for k,S in D2.items():
        if k in D.keys():
            D[k] |= S
        else:
            D[k] = S



def fuse_Ldf(
        Ldf
        ):
    """
    Fusion of a list of df in a new df
    """
    return pd.concat(Ldf)




def fuse_Lfile(
        Lfile, 
        fout="fuse_out.txt",
        with_big_files=False,
        ):
    """
    Fuses a list a files into a new file.
    """
    concat_Lfile(
                    Lfile, 
                    fname=fout,
                    with_big_files=with_big_files,
                )


def fuse_LL(LL):
    """
    Fuses a list of lists into a list.
    """
    L2 = []
    for L in LL:
        if L is not None:
            L2 += list(L)
    return L2


def fuse_LM(
        LM, 
        axis=0, 
    ):
    """
    Fuses a list of matrices into a matrix
    """
    if axis == 0:
        if LM[0] is None:
            return
        nl0, nc0  = LM[0].shape
        n = 0
        for M in LM:
            nl, nc  = M.shape
            if nc != nc0:
                logger.warning("All the M must have the same number of colum for fusion on axis 0")
                return
            n += nl
        M2 = np.empty((n,nc0), dtype=LM[0].dtype)
        i = 0
        for M in LM:
            nl, nc  = M.shape
            M2[list(range(i,i+nl))]=M
            i +=nl
    elif axis ==1:
        nl0, nc0  = LM[0].shape
        n = 0
        for M in LM:
            nl, nc  = M.shape
            if nl != nl0:
                logger.warning("All the M must have the same number of lines for fusion on axis 1")
                return
            n += nc
        M2 = np.empty((nl0,n), dtype=LM[0].dtype)
        i = 0
        for M in LM:
            nl, nc  = M.shape
            M2[:,list(range(i,i+nc))]=M
            i +=nc
    return M2


def fuse_LS(LS):
    """
    Fuses a list of sets into a set.
    """
    # method 1:
    # -------------------------
    # S2 = set()
    # for S in LS:
    #     S2 = S2 | S
    # return S2
    
    # method 2:
    # -------------------------
    return set(chain.from_iterable(LS))
               
    # method 3:
    # -------------------------            
    #eturn set().union(*LS)

def fuse_LT(
        LT
        ):
    """
    Fuses a list of tuple into a list.
    """
    L = []
    for T in LT:
        if T:
            L += list(T)
    return L

def fuse_LV(
        LV,
        axis=None
        ):
    """
    Fuses a list of vectors into a matrix or a vector
    """
    if axis is None:
        return np.append(LV[:1],LV[1:])
    else:
        if axis == 0:
            return np.vstack(LV)
        elif axis == 1:
            return np.hstack(LV)


def fuse_Lvar(
        Lvar,
        fname="toto.p",
        ):
    """
    Fuse a list of pickle variable list  in a new pickle list .
    """
    logger.info("fuse_Lvar")
    if type(Lvar) is str:
        Lvar = get_Lfile(Lvar)
    logger.info(f"\t number of variables:{len(Lvar)}")
    L = []
    for var in Lvar:
        L += load_var(var)
    dump_var(L, fname)


def fuse_M(M):
    """
    Fuse a Matrix in a vector
    """
    return M.flatten()


def fuse_SS(SS):
    """
    Fuses a set of sets into a set.
    """
    return fuse_LS(SS)


def get_Tbox_from_Mp(
        Mp,
        ):
    """
    Gets the center and the size of the box that contains Mp
    """
    p_max = np.max(Mp, axis=0)
    p_min = np.min(Mp, axis=0)
    pc = (p_max + p_min)/2
    dx = p_max[0]-p_min[0]
    dy = p_max[1]-p_min[1]
    dz = p_max[2]-p_min[2]
    return pc, dx, dy, dz
    
    

def get_c_rand_from_LL(LL):
    """
    Get a random couple of values from a list of lists.
    """
    nLL=len(LL)
    if nLL <= 1:
        return None
    Li = random.sample(range(nLL), 2)
    return tuple(sorted([random.choice(LL[Li[0]]), random.choice(LL[Li[1]])]))


def get_cdir():
    """
    Gets the name of current directory
    """
    path = get_path()
    return path.split("/")[-1]

def get_Lsdir(
        sdir="x",
        maxdepth=1,
    ):
    res = bashr(f"find -type d -maxdepth {maxdepth} -name {sdir}", info=True)
    return res.split("\n")

# def get_Lsdir(
#         path=None,
#         with_recursive=False,
#         with_fname=None,
#         without_fname=None,
#         with_Sfname=None,
#         without_Sfname=None):
#     """
#     Gets the list of all the sub-directories in the current directory.
#     """
#     # --------------------------------------------------------------------------
#     logger.debug("get_Lsdir")
#     # --------------------------------------------------------------------------
#     if with_recursive:
#         Lsdir =  get_Lsdir_rec(path=path)
#     else:
#         # with remove of ./
#         Lsdir =  [f.path[2:] for f in os.scandir(path) if f.is_dir()]
#     # --------------------------------------------------------------------------
#     # if s_deb:
#     #     n = len(s_deb)
#     #     Lsdir = [sdir for sdir in Lsdir if sdir[:n]==s_deb]
#     # if s_end:
#     #     n = len(s_end)
#     #     Lsdir = [sdir for sdir in Lsdir if sdir[-n:]==s_end]
#     # --------------------------------------------------------------------------
#     if with_fname:
#         Lsdir = [sdir for sdir in Lsdir if ya(f"{sdir}/{with_fname}")]
#     if without_fname:
#         Lsdir = [sdir for sdir in Lsdir if yapa(f"{sdir}/{without_fname}")]
#     # --------------------------------------------------------------------------
#     if with_Sfname is not None:
#         Lsdir2 = []
#         for sdir in Lsdir:
#             Sfile = set(get_Lfile(path=sdir))
#             if not(with_Sfname.issubset(Sfile)):
#                 Lsdir2.append(sdir)
#         Lsdir = Lsdir2
#     # --------------------------------------------------------------------------
#     if without_Sfname is not None:
#         Lsdir2 = []
#         for sdir in Lsdir:
#             Sfile = set(get_Lfile(path=sdir))
#             if not(without_Sfname.issubset(Sfile)):
#                 Lsdir2.append(sdir)
#         Lsdir = Lsdir2
#     # --------------------------------------------------------------------------
#     return Lsdir


def get_ch_substitute_from_smile(ch):
    """
    Rule of substitution for smile in order to be used as filename
    """
    return str(Lnot_alnum.index(ch))


def get_D_from_2L(
        L1, 
        L2,
        niv_log=0,
        ordered=False,
    ):
    """
    Creation of a dictionnary from two list of the same size.
    """
    inde = "\t" * niv_log
    logger.info(f"{inde}get_D_from_2L")
    if ordered:
        return OrderedDict(zip(L1, L2))
    else:
        return dict(zip(L1, L2))


def get_D_from_D_ordered(D_ordered):
    """
    Gets a dictionnary from an ordered dictionnary.
    """
    return json.loads(json.dumps(D_ordered))



def load_D_from_json(json_file):
    """
    Create a dictionnary from a json file.
    """
    with open(json_file,"r") as fr:
        try:
            return json.loads(fr.read())
        except json.JSONDecodeError:
            logger.warning(f"Error parsing JSON file {json_file}")
            return


def get_D_from_json_file(json_file):
    """
    Create a dictionnary from a json file.
    """
    with open(json_file) as fr:
        try:
            return json.loads(fr.read())
        except json.JSONDecodeError:
            logger.warning(f"Error parsing JSON file {json_file}")
            return

def get_Dn_from_L(
        L,
        sorted=False,
        reverse=False,
        ):
    """
    Transfroms a list into a dictionnary with the number of occurence.
    """
    # List of values
    Lv = list(set(L))
    # Count of each value
    Dvn = {v:L.count(v) for v in Lv}
    if sorted is True:
        Dvn = OrderedDict(
                    sorted(
                        Dvn.items(),
                        reverse=reverse,
                        key=lambda kv: kv[1],
                        )
                    )
    return Dvn

def get_Di_from_L(
        L,
        sorted=False,
        reverse=False,
        ):
    """
    Transfroms a list into a dictionnary with the index in the list
    """
    # List of values
    if type(L) is set:
        L = list(L)
    # Count of each value
    return  {k:i for i,k in enumerate(L)}


def get_D_from_Lc(
        Lc,
        reverse_1=False,
        sorted_2=False,
        reverse_2=False,
        do_fuse=False, # to fuse the elments
        with_count=False,
        niv_log=0,
        ):
    """
    Gets of a dictionnary from a matrix of couples
    """
    logger.debug("\t"*niv_log + "get_D_from_Lc")
    # -----------------------------------------------------------------          
    if reverse_1:
        i0=1
        i1=0
    else:
        i0=0
        i1=1
    # -----------------------------------------------------------------
    D = dict()
    if do_fuse is True:
        for c in Lc:
            if c[i0] in D.keys():
                D[c[i0]] += list(c[i1])
            else:
                D[c[i0]] = list(c[i1])
    else:
        for c in Lc:
            if c[i0] in D.keys():
                D[c[i0]].append(c[i1])
            else:
                D[c[i0]] = [c[i1]]
    if with_count:
        for k,v in D.items():
            D[k] = get_Dn_from_L(
                            v,
                            sorted=sorted_2,
                            reverse=reverse_2,
                            )
    else:
        for k,v in D.items():
            D[k] = list(set(v))
    return D


def get_D_from_Lcv(
        Lcv,
        do_sorted=False,
        reverse_index=False,
        reverse_value=False,
        return_LT=False,
        K=1,                    # coefficient to apply to the values
        ):
    """
    Gets of a dictionnary from a matrix of couples
    """
    if reverse_index:
        i0=1
        i1=0
    else:
        i0=0
        i1=1
    # --------------------------------------------------------------------------
    D = dict()
    for cv in Lcv:
        ci0 = int(cv[i0])
        ci1 = int(cv[i1])
        if ci0 in D.keys():
            D[ci0].append((ci1, cv[2]*K))
        else:
            D[ci0] = [(ci1, cv[2]*K)]
    if do_sorted is False:
        return D
    D2 = dict()
    for ci0, LT in D.items():
        LT = sorted(LT, key=lambda t: t[1], reverse=reverse_value)
        if return_LT:
            D2[ci0] = LT
        else:
            D2[ci0] =[x[0] for x in LT]
    return D2



def get_D_from_LT(LT):
    """
    Converts a list of tuple into a dictionnary
    """
    if len(LT[0]) > 2:
        Li = list(range(len(LT)))
        LT = zip(LT, Li)
    return dict(LT)


def get_D_from_s(s):
    """
    Gets a dict. from a string as in the case of raw formula C6H6O3 => { "C":6, "H":6, "O":3}
    """
    return json.loads(s)
    # D = dict()
    # n = len(s)
    # i = 0
    # i0= 0
    # alpha=True
    # v = None
    # k = None
    # while True:
    #     ch = s[i]
    #     if ch.isalpha() and (alpha is False):
    #         v=s[i0:i]
    #         i0=i
    #         alpha = True
    #         D[k]=int(v)
    #     if not(ch.isalpha()) and (alpha is True):
    #         alpha = False
    #         k=s[i0:i]
    #         i0=i
    #     i += 1
    #     if i == n:
    #         if alpha:
    #             k=s[i0:i]
    #             D[k] = 1
    #         else:
    #             v=s[i0:i]
    #             D[k] = int(v)
    #         break
    # return D


def get_D_from_xml(
        xml
    ):
    import xmltodict
    """
    Convertion of a xml into a dictionnary.
    """
    # Ls = load_Ls_from_txt(xml)
    # return xmltodict.parse("".join(Ls))
    if ".xml" in xml:
        with open(xml) as fd:
            D = xmltodict.parse(fd.read())
    else:
        D = xmltodict.parse(xml)
            
    return D



def get_D_partial_from_xml(
        xml,
        id_ref,
        info,
    ):
    """
    return a dictionnary with the information for each id
    """
    D = dict()
    with open(xml, "r") as fr:
        id_DB2 = ""
        S = set()
        while True:
            try:
                line = next(fr)
                if (id_ref in line):
                    res = c
                    id_DB1 = res.group(1)
                    D[id_DB2]=",".join(list(S))
                    S = set()
                    id_DB2 = id_DB1
                    continue
                if (info in line):
                    res = re.search(r">(.*)<", line)
                    if res is not None:
                        S.add(res.group(1))
            except:
                logger.info(traceback.format_exc())
                break
    return D


def get_D_inversed(
        D,
        key=None, # if the values are dict, use key to make the reversing
        ):
    """
    Gets the inversed dictionnary.
    """
    Dinv = {}
    if key is None:
        for k, v in D.items():
            if type(v) in [list, tuple, set]:
                for x in v:
                    Dinv[x] = Dinv.get(x, []) + [k]
            else:
                Dinv[v] = Dinv.get(v, []) + [k]
        return Dinv
    else:
        for k, D2 in D.items():
            if key in D2.keys():
                v = D2[key]
                if type(v) in [list, tuple, set]:
                    for x in v:
                        Dinv[x] = Dinv.get(x, []) + [k]
                else:
                    Dinv[v] = Dinv.get(v, []) + [k]
        return Dinv

get_D_inv = get_D_inversed

def get_D_sorted_by_key(D):
    """
    sort a dictionnary by key.
    """
    return dict(sorted(D.items()))


def get_df_diff_bw_df(
        df1, 
        df2,
        ):
    """
    Creates a dataframe with the differences between df1 and df2
    """
    return pd.concat(df1, df2).drop_duplicates(keep=False)


def get_df_from_LL(
        LL, 
        Lcolumn=None,
        ):
    """
    Adaptation to my way of defining the functions
    """
    return pd.DataFrame(LL, columns=Lcolumn)


def load_df_from_csv(
        fname,
        sep=",",
        niv_log=0,
        decimal=".",
        header=None,
        nan=0,
        Lcolumn=None,
    ):
    """
    Gets of a dataframe from a csv file.
    Need to be learned.
    """
    inde = "\t" * niv_log
    logger.info(f"{inde}load_df_from_csv")
    niv_log += 1
    inde = "\t" * niv_log
    # -----------------------------------------------------------
    df = pd.read_csv(
                            fname,
                            sep=sep, 
                            # delimiter=None, 
                            header=header, 
                            names=Lcolumn, 
                            #index_col=[0,1,2], 
                            # usecols=None, 
                            # dtype=None, 
                            # engine=None, 
                            # converters=None, 
                            # true_values=None,
                            # false_values=None, 
                            # skipinitialspace=False, 
                            # skiprows=None, 
                            # skipfooter=0, 
                            # nrows=4, 
                            # na_values=None, 
                            # keep_default_na=True, 
                            # na_filter=True, 
                            # verbose=False, 
                            # skip_blank_lines=True, 
                            # parse_dates=None, 
                            # infer_datetime_format=None, 
                            # keep_date_col=False, 
                            # date_parser=None, 
                            # date_format=None, 
                            # dayfirst=False, 
                            # cache_dates=True, 
                            # iterator=False, 
                            # chunksize=None, 
                            # compression=None, 
                            # thousands=None, 
                             decimal=decimal, 
                            # lineterminator=None, 
                            # quotechar='"', 
                            # quoting=0, 
                            # doublequote=True, 
                            # escapechar=None, 
                            # comment=None, 
                            # encoding=None, 
                            # encoding_errors='strict', 
                            # dialect=None, 
                            # on_bad_lines='error', 
                            on_bad_lines='warn',
                            # delim_whitespace=False, 
                            # low_memory=True, 
                            # memory_map=False, 
                            # float_precision=None, 
                            # storage_options=None, 
                            # dtype_backend='pyarrow'
                            )
    #return df.dropna()
    return df.replace(np.nan, nan)
    return df


def get_df_from_csv(
            fname,
            sep="\t",
            sep_head=None,
            sep_string="\r",
            Lcolumn=None,
            i_header=0,
            i0=None,
            imax=None,
            add_header=True,
    ):
    """
    Gets of a dataframe from a csv file, when pd.read_csv fails
    """
    logger.info("get_df_from_csv")
    # -------------------------------------------------------
    if i0 is None:
        i0 = i_header +1
    if i0 < i_header:
        raise Exception(" i0 can't be < to i_header")
    # -------------------------------------------------------
    if sep_head is None:
        sep_head = sep
    # -------------------------------------------------------
    Ls_all = load_Ls_from_txt(fname)
    LL2 = []
    logger.debug(f"\t Header line (n°{i_header}): {Ls_all[i_header]}")
    logger.debug("")
    Ls0 =  Ls_all[i_header].replace("\ufeff", "").split(sep_head)
    Ls0 = [s0.strip() for s0 in Ls0]
    logger.debug(f"{Ls0}")
    if add_header:
        if Lcolumn is None:
            Lcolumn = Ls0
    else:
        LL2.append(Ls0)
    # ---------------------------------------------------------
    logger.debug("\t We suppose that the first line of data is correct")
    Ls = split_string(Ls_all[i0], sep=sep, sep_string=sep_string)
    nLs = len(Ls)
    nLs0 = len(Ls0)
    if nLs > nLs0:
        Lcolumn = Lcolumn + ["x"] * (nLs - nLs0)
    elif nLs < nLs0:
        Lcolumn = Lcolumn[:nLs]
    nLs0 = nLs
    logger.debug(f"\t nLs0={nLs0}")
    # ---------------------------------------------------------
    n_error=0
    Li_error=[]
    if imax is None:
        imax=len(Ls_all)
    for i, s in enumerate(Ls_all[i0+1:imax]):
        if s.strip() == "":
            continue
        Ls = split_string(s, sep=sep, sep_string=sep_string)
        nLs = len(Ls)
        if nLs == nLs0:
            LL2.append(Ls)
            n_error
        else:
            Li_error.append(i+i0)
    if len(LL2):
        if len(Li_error):
            logger.info(f"\t Error in {Li_error[:10]}")
        return get_df_from_LL(LL2, Lcolumn=Lcolumn)
    else:
        logger.info("No correct lines!")



def get_df_from_D(
        D,  # dictionnary with name:L
        orient="columns", # or columns or index
        ):
    """
    df = get_df_from_D({"A":[1,2,3], "B":[4,5,6]})
    df = get_df_from_D({"A":{"x":1,"y":2,"z":3}, "B":{"x":4,"y":5,"z":6}}, orient="index")
    """
    return pd.DataFrame.from_dict(
                                    D,
                                    orient=orient,
                                )


def get_df_from_pdf(
        pdf,
        Lcolumn=None, # List of the title of the column
        n_page_deb=1,
        n_page_end=-1,
        ):
    # -------------------------------------------------------------------------
    logger.info("get_df_from_pdf")
    if n_page_deb < 1:
        n_page_deb = 1
    # ----------------
    if Lcolumn is None:
        raise Exception("A list of the title of the columns must given")
    D = dict()
    for column in Lcolumn:
        D[column] = []
    Lpage = list(extract_pages("/home/krezel@ICOA-03.LOCAL/Téléchargements/jm0c00422_si_001.pdf"))
    for page in Lpage[n_page_deb-1:n_page_end]:
        for element in page:
            if isinstance(element, LTTextBoxHorizontal):
                Lx = list(element)
                for column in Lcolumn:
                    n = len(column)
                    if Lx[0].get_text()[:n] == column:
                        D[column] += [x.get_text().replace("\xa0\n","") for x in element][1:]
    Ln = [len(L) for L in D.values()]
    if len(set(Ln)) != 1:
        nmin= min(Ln)
        D = {k:L[:nmin] for k,L in D.items()}
    return pd.DataFrame.from_dict(D)






def get_df_from_xls(fname):
    """
    Gets a dataframe from a xls file.
    """
    logger.debug("get_df_from_xls")
    # --------------------------------------------------------------------------
    logger.info("if there are many sheet in the xls, return a list of df")
    return pd.ExcelFile(fname)


def get_df_rand_from_df(
        df0,    # initial dataframe
        k,      # number of rows
        ):
    """
    Gets a random df from  df0 with k rows

    Args:
        df0: dataframe
        k: integer

    Returns:
        dataframe
    """
    Li = list(range(len(df0)))
    Li = sorted(random.sample(Li, k=k))
    return df0.iloc[Li]



def get_DT_n_from_LT(LT):
    """
    create a dictionnary with the n number items for each tuple
    """
    D = {}
    for T in LT:
        if T in D.keys():
            D[T] += 1
        else:
            D[T] = 1
    return D



def get_Dvar_n_from_L(L):
    """
    Makes a dictionnary with the number of item for each element.
    """
    D = {}
    for x in L:
        if x in D.keys():
            D[x] += 1
        else:
            D[x] = 1
    return D


def get_Dvn_from_L(
        L,
        ):
    """
    Transfroms a list into a dictionnary with the number of occurence.
    """
    return Counter(L)



def get_i_from_s_by_ref(
        s, 
        s_ref=None
        ):
    """
    Gets the integer associated to the string s_ref in the string s.
    :Ex if s ="AA_BB_CC_r10_c20_XX_TT"
    :Ex get_i_from_s_by_ref(s,"_c") returns 20
    """
    logger.debug("get_i_from_s_by_ref")
    # --------------------------------------------------------------------------
    if ref is None:
        return
    if ref[0] != "_":
        ref = "_" + s_ref
    i = s.index(s_ref)
    s2 = s[i+len(s_ref):]
    if "_" in s:
        i = s2.index("_")
        return int(s2[:i])
    else:
        return int(s2)


def get_increase_value_from_L(L):
    d=0
    v1=L[0]
    for v2 in L[1:]:
        if v2 > v1:
            d += v2-v1
        v1=v2
    return d

Lnot_alnum =['#', '', '%', '(', ')', '+', '-', '.', '=', '[', ']', '_']

def get_kwargs():
    """
    Gets the dictionnary of the parameters
    """
    logger.debug("get_kwargs")
    frame = inspect.currentframe().f_back
    keys, _, _, values = inspect.getargvalues(frame)
    kwargs = {}
    for key in keys:
        if key != 'self':
            kwargs[key] = values[key]
    return kwargs


def get_L_from_L0_without_duplicate(L):
    """
    Remove duplicate entries from a list while preserving order
    This function uses Python's standard equivalence testing methods in
    order to determine if two elements of a list are identical. So if in the list [a,b,c]
    the condition a == b is True, then regardless of whether a and b are strings, ints,
    or other, then b will be removed from the list: [a, c]
    Parameters
    ----------
    list_with_dupes : list
        A list containing duplicate elements
    Returns
    -------
        The list with the duplicate entries removed by the order preserved
    Examples
    --------
    >>> a = [1,3,2,4,2]
    >>> print(remove_dupes(a))
    [1,3,2,4]
    """
    S = set()
    #func = S.add
    return [v for v in L if not (v in S or S.add(v))]




def get_L_from_sorted_LT(
            LT, # List of tuple (A,x) with A an object and x a float or integer
            reverse=False, # to sort by decreasing value
            vmin=None,
            vmax=None,
            ):
    """
    get a list from a list of tuples composed of an object and a value.
    """
    if vmin is not None:
        LT = [T for T in LT if T[1]> vmin]
    if vmax is not None:
        LT = [T for T in LT if T[1] < vmax]
    LT = sorted(
            LT,
            key=lambda t: t[1],
            reverse=reverse,
            )
    return [T[0] for T in LT]



def get_L_lower(L):
    return [x.lower() for x in L]


def get_L_upper(L):
    return [x.upper() for x in L]


def get_La_rand(n=10):
    """
    Gets a random list of n alphanumeric.
    """
    return random.choices(Lalpha, k=n)


def get_latex_from_df(df):
    """
    Gets latex code for table from df.
    """
    return df.to_latex()


def get_Lc_bw_L1_L2(L1, L2):
    """
    Gets a list of combinations between two lists
    """
    return list(itertools.product(L1, L2))


def get_Lc_combinations_from_LLi(LLi):
    """
    """
    if len(LLi) == 0:
        return []
    elif len(LLi) == 1:
        return [(i,) for i in LLi[0]]
    Lc = list(itertools.product(LLi[0], LLi[1]))
    if len(LLi) > 1:
        for i in range(2, len(LLi)):
            Lc = list(itertools.product(Lc, LLi[i]))
            Lc = [simplify_tuple(t) for t in Lc]
    return Lc



def get_Lc_inv(Lc):
    """
    Get the  list of couples inversed.
    """
    return [(c[1], c[0]) for c in Lc]



def get_Lcolor_1(nL=10):
    """
    Generates a list of n different colors
    """
    n = 2
    L2 = []
    while True:
        #logger.debug(f"n={n}")
        L1 = list(map(list, product(list(range(n)), repeat=3)))
        #logger.debug(f"\t{L1}")
        L1 =  list(set([tuple(x) for x in L1]))
        #logger.debug(f"\t{L1}")
        L1 = (np.array(L1)/(n-1)).tolist()
        #logger.debug(f"\t{L1}")
        L1 =  list(set([tuple(x) for x in L1]) - {(0,0,0), (1,1,1)})
        #logger.debug(f"\t{L1}")
        L3 = sorted(list(set(L1)-set(L2)))
        LT3 = [(x,sum(x)) for x in L3]
        LT3 = sorted(LT3, key=lambda x:x[1])
        L3 = [x[0] for x in LT3]
        nL3 = len(L3)
        nL2 = len(L2)
        nL1 = len(L1)
        if nL2 + nL3 > nL:
            L2 += L3[:nL-nL2]
            break
        L2 += L3
        n += 1
    return L2


def get_Ldf_from_pdf(
        pdf,
        Lpage="all"
        ):
    from tabula import read_pdf
    return read_pdf(pdf, pages=Lpage)


def get_Ldir(
        dir_name=None,
        path=None,
        maxdepth="",
        nmax_days_old=None,
        ):
    """
    Gives the list of subdirectoies contained in a directory.
    """
    logger.debug("get_Ldir")
    # --------------------------------------------------------------------------
    Ldir = [dir for dir in os.listdir(os.getcwd()) if os.path.isdir(dir)]
    if dir_name is not None:
        Ldir  = [dir for dir in Ldir if re.search(dir_name, dir) is not None]
    return Ldir
                




def get_Ldir_empty():
    """
    Gets the list of empty sub-directories.
    """
    Lsdir = get_Lsdir()
    Lsdir2 = []
    for sdir in Lsdir:
        if len(get_Lfile(path=sdir)) == 0:
            Lsdir2.append(sdir)
    return Lsdir2


def get_Ldir_with_fname(
        fname,
        path=None,
        Ldir0=None, # List of diretory to scan
        nmax_days_old=None,
        with_print=False,
        ):
    """
    Get the list of directories in path that have the fname file.
    """
    logger.info("get_Ldir_with_fname")
    # --------------------------------------------------------------------------
    if nmax_days_old:
        before = datetime.now() - timedelta(days = nmax_days_old)
    Ldir = []
    if Ldir0 is None:
        logger.info("\t scan in all sub-directories")
        if path is None:
            path = get_path()
        with os.scandir(path) as Le1:
            for i,e1 in enumerate(Le1):
                if with_print:
                    print(f"get_Ldir_with_fname {i:>6}", end="\r")
                if e1.is_dir():
                    if ya(f"{e1.name}/{fname}", nmax_days_old=nmax_days_old):
                        Ldir.append(e1.name)
    else:
        logger.info("\t Ldir0 is given")
        for dir in Ldir0:
            logger.debug(f"\t dir={dir}")
            if ya(dir):
                if ya(f"{dir}/{fname}", nmax_days_old=nmax_days_old):
                    Ldir.append(dir)
    return Ldir


def get_Ldir_without_fname(
        fname,
        path=None,
        Ldir0=None, # List of diretory to scan
        nmax_days_old=None,
        ):
    """
    Get the list of directories in path that don't have the fname file.
    """
    logger.info("get_Ldir_without_fname")
    # --------------------------------------------------------------------------
    if path is None:
        path = get_path()
    Ldir = []
    if Ldir0 is None:
        logger.info("\t scan in all sub-directories")
        if path is None:
            path = get_path()
        with os.scandir(path) as Le1:
            for i,e1 in enumerate(Le1):
                #print(f"get_Ldir_without_fname {i:>6}", end="\r")
                if e1.is_dir():
                    if yapa(f"{e1.name}/{fname}", nmax_days_old=nmax_days_old):
                        Ldir.append(e1.name)
    else:
        logger.info("\t Use of Ldir0 given")
        for dir in Ldir0:
            logger.debug(f"\t dir={dir}")
            if ya(dir):
                if yapa(f"{dir}/{fname}", nmax_days_old=nmax_days_old):
                    Ldir.append(dir)
    return Ldir



def get_Lfile(
        s="",
        nmax_days_old=None,
        maxdepth=1,        # Levels of subdirectories for search, by defauult the files in the current directory
        path=None,
        max_size=None,
        min_size=None,
        inverse=False,
        Ls_in=None, # Set of string to have in  the name
        ):
    """
    Gives the list of files
    """
    logger.debug("get_Lfile")
    # --------------------------------------------------------------------------
    path0 = get_path()
    if path:
        os.chdir(path)
    # --------------------------------------------------------------------------
    ref= None
    if "/" in s:
        Ls = s.split("/")
        s = Ls[-1]
        ref = "/".join(Ls[:-1])
        os.chdir(ref)
    # --------------------------------------------------------------------------
    if type(maxdepth) is int:
        maxdepth = f"-maxdepth {maxdepth}"
    # --------------------------------------------------------------------------
    if s != "":
        if s[0] == "!":
            s = f"! -iname '{s[1:]}'" 
        else:
            s = f"-iname '{s}'"
    # --------------------------------------------------------------------------
    size=""
    if max_size is not None:
        if max_size == 0:
            size += "-empty"
        else:
            size += f" -size -{max_size}"
    if min_size is not None:
        size += f" -size +{min_size}"
    # --------------------------------------------------------------------------
    
    cmd = f"find . {maxdepth} {s}  -type f  {size}"
    s_res = bashr(cmd, info=False)
    #logger.debug(f"\t {s_res}")
    if s_res != "":
        Lfile = s_res.split("\n")
    else:
        Lfile = []
    # --------------------------------------------------------------------------
    if nmax_days_old is not None:
        logger.debug("\t filter on nmax_days_old")
        Lfile = [file for file in Lfile if is_newer(file, nmax_days_old)]
    # --------------------------------------------------------------------------
    if Ls_in is not None:
        logger.info("Ls_in is not None")
        s2 = '|'.join(Ls_in)
        regex = f"({s2})"
        Lfile = [file for file in Lfile if re.search(regex, file)]
    if ref is not None:
        Lfile = [f"{ref}/{file}" for file in Lfile]
    os.chdir(path0)
    # --------------------------------------------------------------------------
    return sorted(Lfile)


def get_Li_correspondance(
        Li1, 
        Li2,
        ):
    """
    Gives the correspondance between two list of the same size.
    """
    logger.debug("get_Li_correspondance")
    if len(Li1) != len(Li2):
        raise Exception("Li1 and Li2 nust of the same size")
    D = dict()
    for i1, i2 in zip(Li1, Li2):
        D[i1] = i2
    return [D[i] for i in sorted(D.keys())]


def get_Li_from_cr_brackout_in_s(cr0, s):
    """
    Search the character c0 that are out of brackets ( or { or [
    """
    Li = []
    for cr in s:
        if cr == "(":
            Li.append(1)
        elif cr == ")":
            Li.append(-1)
        else:
            Li.append(0)
    Vi1 = np.array(Li)
    # --------------------------------------------------------------------------
    Li = []
    for cr in s:
        if cr == "[":
            Li.append(1)
        elif cr== "]":
            Li.append(-1)
        else:
            Li.append(0)
    Vi2 = np.array(Li)
    # --------------------------------------------------------------------------
    Li = []
    for cr in s:
        if cr == "{":
            Li.append(1)
        elif cr == "}":
            Li.append(-1)
        else:
            Li.append(0)
    Vi3 = np.array(Li)
    # --------------------------------------------------------------------------
    Li = [i for i,cr in enumerate(s) if cr==cr0]
    Li2 = []
    for i in Li:
        if (    (np.sum(Vi1[:i]) == 0) and
                (np.sum(Vi2[:i]) == 0) and
                (np.sum(Vi3[:i]) == 0)  ):
            Li2.append(i)
    return Li2
def get_Li_from_L(L, v=None, vmax=None, vmin=None):
    """
    Gets the list of index with
    """
    V = np.array(L)
    if v:
        return np.where(V == v)[0].tolist()
    elif (vmin is not None) and (vmax is not None):
        return np.argwhere((V > vmin) & (V < vmax))[:,0].tolist()
    elif (vmin is None) and (vmax is not None):
        return np.argwhere(V < vmax)[:,0].tolist()
    elif (vmin is not None) and (vmax is  None):
        return np.argwhere(V > vmin)[:,0].tolist()


def get_Li_unic_from_L(L):
    logger.debug("get_Li_unic_in_L")
    return [L.index(v) for v in set(L)]



Lalpha = list("abcdefghijklmnopqrstuvwxyz")


def get_Linfo_from_txt(
        file,
        repere="$$$$",
        on_line=False, # to take the info on the line and not on the next line

    ):
    """
    Generic function to find info in a text file.
    It works if the repere and the information are not on the same line but on consecutive lines.
    """
    with open(file, "r") as fr:
        L = []
        while True:
            try:
                line = next(fr)
                if repere in line:
                    if on_line:
                        L.append(line[:-1])
                    else:
                        line = next(fr)
                        L.append(line[:-1])
            except:
                break
        return L
    
    


    
    
def get_Lkey_with_max_value_from_D(D):
    """
    Get the list of key with max value.
    """
    Vv=np.array(list(D.values()))
    Vk=np.array(list(D.keys()))
    return Vk[np.where(Vv == max(Vv))].tolist()


def get_LLrand_rand(
        nLL=10, 
        nL_max=10,
        nmin=0,
        nmax=100,
    ):
    """
    create a random list of list (used for tests)
    """
    return [get_Lrand(
                        nL,
                        nmin=nmin,
                        nmax=nmax) for nL in random.choices(range(nL_max), k=nLL)]



def get_longest_sub_s(s1, s2):
    """
    Find the longest common sub-string.
    """
    s = ""
    n1, n2 = len(s1), len(s2)
    for i in range(n1):
        for j in range(n2):
            k=0
            match=""
            while (i+k < n1) and \
                  (j+k < n2) and \
                  (s1[i+k] == s2[j+k]):
                match += s2[j+k]
                k+=1
            if (len(match) > len(s)):
                s = match
    return s


def get_Lpath(
        path=None,
        maxdepth=1,
        nmax_days_old=None,
        ):
    """
    Gives the list of subdirectoies contained in a direcroty.
    """
    logger.debug("get_Lpath")
    logger.debug(f"\t maxdepth={maxdepth}")
    # --------------------------------------------------------------------------
    # method 1 (very slow) with bash
    # --------------------------------------------------------------------------
    # if path:
    #     path0 = get_path()
    #     os.chdir(path)
    # if type(maxdepth) is int:
    #     maxdepth = f"-maxdepth {maxdepth}"
    # if s != "":
    #     s = f"-iname '{s}'"
    # cmd = f"find . {maxdepth} {s} -type d"
    # logger.debug(f"\t {cmd}")
    # s_res = bashr(cmd)
    # if s_res != "":
    #     Ldir = s_res.split("\n")
    # else:
    #     Ldir = []
    # return sorted([dir[2:] for dir in Ldir if len(dir)>2])
    # --------------------------------------------------------------------------
    # method 2
    # --------------------------------------------------------------------------
    # if path is None:
    #     path = get_path()
    # if maxdepth == 1:
    #     Ldir = [f.path for f in os.scandir(path) if f.is_dir() ]
    # else:
    #     Ldir = glob(f"{path}/" + "*/" * maxdepth)
    # return [dir.split("/")[-1] for dir in Ldir]
    # --------------------------------------------------------------------------
    # method 3
    # --------------------------------------------------------------------------
    if path is None:
        path = get_path()
    else:
        if yapa(path):
            return []
    # --------------------------------------------------------------------------
    Lpath = [f.path for f in os.scandir(path) if f.is_dir()]
    if len(Lpath):
        logger.debug(f" Lpath={Lpath}")
    else:
        return []
    if maxdepth == 1:
        return Lpath
    # --------------------------------------------------------------------------
    for path in Lpath:
        Lpath.extend(get_Lpath(path=path, maxdepth=maxdepth-1))
    return Lpath


def get_Mperm_rand(
        nperm=3, # number of permutations
        n=10, # llength of list to apply the permutation
        unique=True,
    ):
    Mperm = np.array([np.random.permutation(range(n)) for _ in range(nperm)])
    if unique is False:
        return Mperm
    Mperm_u = np.unique(Mperm, axis=0)
    if len(Mperm_u) == nperm:
        return Mperm
    else:
        logger.warnin("failed to give unique permutation, try again!")
    

# ===========================================================================================================

def get_Lperm_from_L(
        L,
        with_repet=False,
        nmax=None,
        with_recursive=True,# to apply to recurcive method
        ):
    """
    Gives the list of permutations from L, with or without repetition
    """
    logger.debug("get_Lperm_from_L")
    if nmax is not None:
        recursion = True
    if with_recursive:
        logger.debug("get_Lperm_from_L_with_recursive")
        return get_Lperm_from_L_with_recursive(L, with_repet=with_repet)
    else:
        if nmax < math.factorial(len(L))/100:
            return get_Lperm_from_L_random(L, with_repet=with_repet)
        else:
            return get_Lperm_from_L_linear(L, with_repet=with_repet, nmax=nmax)


def get_Lperm_from_L_linear(
        L,
        with_repet=False,
        nmax=None):
    """
    Gets a list of permutations from a list by a linear method.
    """
    logger.debug("get_Lperm_from_L_linear")
    Lperm = [L]
    n = len(L)
    for k in range(0,n-1):
        for i in range(0, len(Lperm)):
            Lz = Lperm[i][:]
            for c in range(0,n-k-1):
                Lz.append(Lz.pop(k))
                if with_repet==True or (Lz not in Lperm):
                    Lperm.append(Lz)
                    if len(Lperm) >= nmax:
                        return Lperm
    return Lperm


def get_Lperm_from_L_random(
        L,
        with_repet=False,
        nmax=None,
        nperm=None, # Number of permutations
        ):
    """
    Gets a list of permutations from a list by a random method.
    """
    logger.debug("get_Lperm_from_L_random")
    Lperm = []
    n = len(L)
    i = 0
    for _ in range(nmax):
        if nperm:
            Li1 = get_Lrand_from_L(range(n), nperm)
            while True:
                Li2 = list(np.random.permutation(Li1))
                if Li1 != Li2:
                    break
            Lz = L[:]
            for (i1, i2) in zip(Li1, Li2):
                Lz[i2] = L[i1]
        else:
            Vperm = np.random.permutation(range(n))
            Lz = [L[k] for k in Vperm]
        if Lz not in Lperm:
            Lperm.append(Lz)
            i += 1
            if i >= nmax:
                return Lperm
    return Lperm


def get_Lperm_from_L_with_recursive(
        L,
        with_repet=False,
        k=0):
    """
    Gets a list of permutations from a list by recursivity.
    """
    Lperm = [L]
    n = len(L)
    if k == n-1:
        return []
    Lz = L[:]
    for c in range(0, n-k):
        if with_repet==True or (Lz not in Lperm):
            Lperm.append(Lz[:])
            Lperm.extend(get_Lperm_from_L_with_recursive(Lz, with_repet, k+1)[1:])
        Lz.append(Lz.pop(k))
    return Lperm
# ===========================================================================================================

def get_Lrand(
        nL=10,
        nmin=0, 
        nmax=100,
        ):
    return random.choices(range(nmin, nmax+1), k=nL)


def get_Lrand_from_L(
        L, 
        n=None,
        dn=None,
    ):
    """
    Gets a random list of n elements from a list (without repetition).
    """
    logger.debug("get_Lrand_from_L")
    # ------------------------------------------------------------------------
    nL = len(L)
    if dn is None:
        if n is None:
            n=nL
        else:
            n = min(nL, n)
        return random.sample(L, n)
    else:
        n = random.choice(list(range(nL-dn)))
        return L[n:n+dn]
        


def get_Ls_from_ATOM_line_in_pdb(
        line, #
        Lid=[
             6,     #"command",          #0        #6<
             12,    #"atom_id",          #1        #5>
             16,    #"atom_name",        #3        #2<
             17,    #"alternate_locate", #4        #1.
             21,    #"residue_name",     #5        #4>
             22,    #"chain_id",         #6        #1.
             26,    #"residue_seq_num",  #7        #4>
             27,    #"residue_insert",   #8        #1.
             38,    #"x",                #9        #8.3>
             46,    #"y",                #10       #8.3>
             54,    #"z",                #11       #8.3>
             61,    #"occupancy",        #12       #6.2>
             67,    #"temp_factor",      #13       #6.2>
             76,    #"segment_id",       #14
             80,    #"element_symbol"    #15
            ],
        ):
    """
    Gets a list of string Ls from a line beginning by ATOM in a pdb file.
    """
    LL = split_L_into_LL_from_Li(line, Lid)[:-1]
    LL = [L.strip() for L in LL]
    return LL


def get_Ls_from_CONECT_line_in_pdb(
        line, #
        Lid=[6, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56, 61, 66, 71, 76, 81], # List of positions on line
        ):
    """
    Gets a list of string Ls from a line beginning by ATOM in a pdb file.
    """
    LL = split_L_into_LL_from_Li(line, Lid)[:-1]
    LL = [L.strip() for L in LL]
    return LL


def get_Lsdir_with_fname_updated(
        Lsdir=None,
        fname=None,
        date_ref=date(220, 10, 20) #
        ):
    """
    Search in Ldir, the sub directories containing fname updated after date_ref
    """
    if Lsdir is None:
        Lsdir = get_Lsdir()
    Lsdir2 = []
    for sdir in Lsdir:
        if date.fromtimestamp(os.path.getmtime(f"{sdir}/{fname}")) > date_ref:
            Lsdir2.append(sdir)
    return Lsdir2


def get_LT_bw_n_for_parallel(
        n1,n2,             # number total in each group
        dn=400,         # number to be treated at the same time
        ratio=3,        # ratio between dn / m that indicates how to use the rest.
        ):
    """
    This function creates a list of tuples that describe the jobs to do for a parallelization.
    """
    logger.info("get_LT_bw_n_for_parallel")
    # --------------------------------------------------------------------------
    LT = []
    k1, m1 = divmod(n1, dn)
    k2, m2 = divmod(n2, dn)
    logger.debug(f"\t k1={k1} m1={m1}")
    logger.debug(f"\t k2={k2} m2={m2}")
    if (m1 > dn / ratio) and (m2 > dn / ratio):
        logger.debug(f"\t m1 > dn / {ratio}")
        logger.debug(f"\t m2 > dn / {ratio}")
        for i in range(k1+1):
            for j in range(k2+1):
                LT.append((i*dn, min((i+1)*dn, n1), j*dn, min((j+1)*dn, n2)))
    else:
        logger.debug(f"\t m < dn / {ratio}")
        k1 = max(0, k1-1)
        k2 = max(0, k2-1)
        for i in range(k1):
            for j in range(k2):
                LT.append((i*dn, (i+1)*dn, j*dn, (j+1)*dn))
            LT.append((i*dn, (i+1)*dn, (k2-1)*dn, n2))
        LT.append((k1*dn, n1, k2*dn, n2))
    return LT


def get_LT_for_parallel(
        n,              # number total
        dn=400,         # number to be treated at the same time
        ratio=3,        # ratio between dn / m that indicates how to use the rest.
        niv_log=0, 
        ):
    """
    This function creates a list of tuples that describes the jobs to do for a parallelization.
    """
    logger.info("\t"*niv_log + "get_LT_for_parallel")
    # --------------------------------------------------------------------------
    LT = []
    k, m = divmod(n, dn)
    logger.debug(f"\t k={k} m={m}")
    if m > dn / ratio:
        logger.debug(f"\t m > dn / {ratio}")
        for i in range(k+1):
            for j in range(i, k+1):
                LT.append((i*dn, min((i+1)*dn, n), j*dn, min((j+1)*dn, n)))
    else:
        logger.debug(f"\t m < dn / {ratio}")
        k = max(0, k-1)
        for i in range(k):
            for j in range(i, k):
                LT.append((i*dn, (i+1)*dn, j*dn, (j+1)*dn))
            LT.append((i*dn, (i+1)*dn, (k-1)*dn, n))
        LT.append((k*dn, n, k*dn, n))
    return LT



def get_LT_for_parallel_with_split(
        n,              # number total
        dn=400,         # number to be treated at the same time
        n_to_split=None, # the dataframe is considered as 2 dataframes with split at n_to_split, only the interactions bw the 2 groups are searched
        niv_log=0, 
        ):
    """
    This function creates a list of tuples that describes the jobs to do for a parallelization.
    """
    logger.info("\t"*niv_log + "get_LT_for_parallel_with_split")
    logger.debug(f"\t n_={n}") 
    logger.debug(f"\t n_to_split={n_to_split}") 
    # --------------------------------------------------------------------------
    LT = []
    n1 = n_to_split - 1
    n2 = n - n_to_split
    k1, m1 = divmod(n1, dn)
    k2, m2 = divmod(n2, dn)
    logger.debug(f"\t k1={k1} m1={m1}")
    logger.debug(f"\t k2={k2} m2={m2}")
    for i in range(k1+1):
        for j in range(k2+1):
            LT.append((i*dn              ,             min((i+1)*dn, n1), 
                       n_to_split +  j*dn, n_to_split + min((j+1)*dn, n2)))
    return LT


def get_LT_for_parallel_with_remove_T_between_n2_last(
        n1,              # number total
        dn_removed,
        dn=400,         # number to be treated at the same time
        ratio=3,        # ratio between dn / m that indicates how to use the rest.
        niv_log=0, 
        ):
    """
    This function creates a list of tuples that describe the jobs to do for a parallelization.
    """
    logger.info("\t"*niv_log + "get_LT_for_parallel_with_remove_T_between_n2_last")
    # --------------------------------------------------------------------------
    LT = []
    n2 = n1 - dn_removed
    k1, m1 = divmod(n1, dn)
    k2, m2 = divmod(n2, dn)
    logger.debug(f"\t k1={k1} m1={m1}")
    logger.debug(f"\t m1 > dn / {ratio}")
    for i in range(k2):
        for j in range(i, k1+1):
            LT.append((i*dn, min((i+1)*dn, n2), j*dn, min((j+1)*dn, n1)))
        if k1*dn != n1:
            LT.append((i*dn, min((i+1)*dn, n2), j*dn, min((j+1)*dn, n1)))
    if k2*dn != n2:
        for j in range(k2, k1+1):
            LT.append((k2*dn, min((k2+1)*dn, n2), j*dn, min((j+1)*dn, n1)))
    return LT



def get_LT_from_Mp(Mp, nmax=1000):
    """
    Gets an iterator of tuple (Mp, i) from a matrix of points.
    """
    logger.debug("get_LT_from_Mp")
    nMp = len(Mp)
    if nMp <= nmax:
        yield (Mp,0)
    else:
        r = nMp % nmax
        N = nMp // nmax
        logger.debug(f"\t r={r} N={N}")
        rmax = nmax/3
        if 0 < r <= rmax:
            if N >= 2:
                N = N-1
        for i in range(N-1):
            yield (Mp[i*nmax:(i+1)*nmax], i*nmax)

        yield (Mp[(N-1)*nmax:nMp], (N-1)*nmax)


def get_LT_from_L(L, nmax=1000):
    """
    Gets an iterator of tuple (L, i) from a list.
    """
    logger.debug("get_LT_from_L")
    nL = len(L)
    if nL <= nmax:
        yield (L,0)
    else:
        r = nL % nmax
        N = nL // nmax
        logger.debug(f"\t r={r} N={N}")
        rmax = nmax/3
        if 0 < r <= rmax:
            if N >= 2:
                N = N-1
        for i in range(N-1):
            yield ([L[j]for j in range(i*nmax,(i+1)*nmax)], i*nmax)

        yield ([L[j] for j in range((N-1)*nmax,nL)], (N-1)*nmax)

       
        
        
def get_LT_with_sorted_T(L):
    """
    Sorts the values in the tuples from the smallest to the biggest.
    :param L: list of tuples,
    :returns: list of tuples
    """
    logger.debug("get_LT_with_sorted_T")
    # --------------------------------------------------------------------------
    logger.debug(f"\t L={L}")
    return [tuple(sorted(list(e))) for e in L]


def get_LTcomb(n,p):
    """
    Gets the list of tuples of combinations of p elements among n.
    """
    return list(combinations(range(n), p))


def get_M_from_LV(LV, axis=0):
    """
    Gets a matrix M from a list of vectors LV
    """
    return np.stack(LV, axis=axis)


def get_n_rand(n1, n2):
    return random.randint(n1, n2)


def get_nLfile(s=None):
    """
    Gives the number of files in the current directory
    """
    logger.debug("get_nLfile")
    if s is None:
        nLfile =  int(bashr("ls | wc -l", info=False))
    else:
        nLfile  = int(bashr(f"ls | grep '^{s}' |wc -l", info=False))
    logger.debug(f"\t {nLfile} files")
    return nLfile


def get_n_in_Lvar(
        s,
        nLvar=None,
        nmax_days_old=None,
        info=False,
        nothing=[]
        ):
    """
    Gets the number of elements in Lvar
    """
    path = get_path()
    logger.debug(f"get_n_in_Lvar in {path}")
    if type(s) is list:
        Lfile = s
    else:
        if ".p" not in s:
            s += ".p"
        Lfile = get_Lfile(s, nmax_days_old=nmax_days_old)
    logger.debug(f"\t{len(Lfile)} files")
    Lres = []
    # ------------------------------------------------
    if nLvar is not None:
        Lfile = Lfile[:nLvar]
        logger.info(f"\t only {nLvar} files will be read")
    # ------------------------------------------------
    nLfile = len(Lfile)    
    n = 0
    for i,file in enumerate(Lfile):
        if info:
            print(f"{i:>8}/{nLfile:<8}", end="\r")
        n += len(load_var(file, nothing=nothing))
    return n

def get_path(s=None):
    """
    Returns:
        the current directory
    """
    if s:
        Ls = s.split("/")
        return "/".join(Ls[:-1])
    else:
        return os.getcwd()


def get_path_py(file):
    """
    Gets the path to current python file obtained with __file__.

    """
    return os.path.dirname(os.path.realpath(file))



def get_s_ATOM_line_from_Ls(Ls):
    """
    Gets an ATOM line of pdb file from a list of string Ls.
    """
    return f"{Ls[0]:<6}{Ls[1]:>5} {Ls[2]:>4}{Ls[3]:1}{Ls[4]:>3} {Ls[5]:1}{Ls[6]:>4}{Ls[7]:>1}{Ls[8]:>11}{Ls[9]:>8}{Ls[10]:>8} {Ls[11]:>5} {Ls[12]:>5} {Ls[13]:>8}{Ls[14]:>3} "




def get_s_for_filename(s):
    """
    return a string without bad caracters for filename
    """
    return "".join(ch if (ch.isalnum() == True) else get_ch_substitute_from_smile(ch) for ch in s )


def get_s_from_Lc(
        Lc, # List of tuple int, string
        ):
    """
    Creates a string from a list of couples.
    Ex: [("A",6), ("B",2)] => "AAAAAABB".
    """
    s = ""
    for c in L:
        s += c[1] * c[0]
    return s


def get_s_from_object(var):
    """
    Gets the string name of var. 
    :param var: variable to get name from.
    :return: string
    """
    logger.debug("get_s_from_object")
    # --------------------------------------------------------------------------
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    for k, v in callers_local_vars:
        print(k)
        if (v == var) and (k is not None):
            return k

get_s_from_var = get_s_from_object


def get_s_from_p_in_pdb(p):
    """
    Gets the string s from a point p for a pdb line.
    """
    return f"{p[0]:>8.3f}{p[1]:>8.3f}{p[2]:>8.3f}"


def get_s_from_s0_after_removing_Ls(s0, Ls):
    """
    Gets a string by removing a list of strings from a initial string.
    """
    s1= deepcopy(s0)
    for s in Ls:
        s1 = s1.replace(s, "")
    return s1


def get_s_from_s0_bw_Ls(s0, Ls=None):
    """
    Gets a string from a string between 2 smaller strings in Ls.
    """
    s2 = copy(s0)
    if s2:
        if Ls[0] in s2:
            s2 = s2.split(Ls[0])[1]
        else:
            logger.warning(f" no {Ls[0]} in string")
            return
        # ---------------------------------------
        if Ls[1] in Ls2:
            s2 = s2.split(Ls[1])[0]
        else:
            logger.warning(f" no {Ls[1]} in string")
            return
        return s2
    else:
        return


def get_S_inter_from_LS(LS):
    """
    Gets the intersection between a list of sets.
    """
    if len(LS) > 2:
        return set(LS[0]) & get_S_inter_from_LS(LS[1:])
    else:
        return set(LS[0]) & set(LS[1])


def get_s_json_from_D(D):
    """
    get a json string from a dictionnary.
    """
    return json.dumps(D)


def get_s_padded_with_a(s, a="E", n=100):
    """
    Gets a string completed by a up to a size of n.
    """
    ns = len(s)
    return s + a*(n-ns)


def get_s_with_a_only(s):
    """
    Gets a string with only alphanumeric.
    """
    return  "".join([ch for ch in s if ch.isalpha()])


def get_s_with_maj_first(s):
    """
    Gets a string with minuscules letters except the first letter.
    """
    logger.debug("get_s_with_maj_first")
    logger.debug(f"\t s={s}")
    # --------------------------------------------------------------------------
    s = s.lower()
    return s[0].upper() + s[1:]



def get_size(obj):
    """
    Gets theb size occupied in memory by the object obj
    """
    logger.debug("get_size")
    # --------------------------------------------------------------------------
    s_oname = get_s_from_object(obj)
    if s_oname is not None:
        fname = s_oname + ".p"
        logger.debug(f"\t fname='{fname}'")
        if fname in get_Lfile():
            logger.debug(f"\t {fname} found in current directory.")
            return os.stat(fname).st_size
        else:
            return len(pickle.dumps(obj))


def get_Sneighbor_in_L_from_Ln(L, Ln):
    """
    Search the neighbors to the elements Ln in L.
    """
    Sn2 = set()
    nL = len(L)
    for n in Ln:
        if n in L:
            i = L.index(n)
            if i == 0:
                Sn2.add(L[1])
            elif i == (nL-1):
                Sn2.add(L[nL-2])
            else:
                Sn2.add(L[i+1])
                Sn2.add(L[i-1])
    return Sn2


def get_ST_from_M(M):
    return set(map(tuple, M))


def get_sub_D(D1, Lkey):
    """
    Gets a sub-dictionnary from a list of keys.
    """
    D2 = dict()
    for key in Lkey:
        if key in D1.keys():
            D2[key] = D1[key]
        else:
            D2[key] = None
    return D2



def get_TT_from_M(M):
    return tuple(map(tuple, M))


def get_value_in_Ls(Ls, ref=None):
    """
    if Ls ="AA_BB_CC_r10_c20_XX_TT"
    get_value_in_string(Ls,"_c") done 20
    """
    logger.debug("get_value_in_string")
    # --------------------------------------------------------------------------
    if ref is None:
        return
    if ref[0] != "_":
        ref = "_" + ref
    i = Ls.index(ref)
    Ls2 = Ls[i+len(ref):]
    if "_" in Ls2:
        i = Ls2.index("_")
        return int(Ls2[:i])
    else:
        return int(Ls2)


def is_L1_in_L2(L1, L2):
    """
    Checks if the list L1 is a subpart of L2.
    """
    nL1 = len(L1)
    nL2 = len(L2)
    if nL1 > nL2:
        return None
    for i in range(nL2-nL1+1):
        if L1 == L2[i:i+nL1]:
            return True
    return False


def is_L1_in_LL(L1, LL):
    """
    Checks if the list L1 is in LL.
    """
    nL1 = len(L1)
    for L2 in LL:
        nL2 = len(L2)
        for i in range(nL2-nL1+1):
            if L1 == L2[i:i+nL1]:
                return True
    return False


def is_L1_L2_different(L1, L2):
    """
    Return True if lists L1 and L2 are diffrent.
    """
    # ==========================================================================
    # version 1
    # ==========================================================================
    # for (v1, v2) in zip(L1, L2):
    #     if v1 != v2:
    #         return True
    # return False
    # ==========================================================================
    # version 2
    # ==========================================================================
    return not(tuple(L1) == tuple(L2))


def is_L1_L2_equal(L1, L2):
    """
    Return True if lists L1 and L2 are equal.
    """
    return tuple(L1) == tuple(L2)


def is_newer(fname, nmax_days_old):
    """
    Return True if the file is newer than nmax_days_old.
    """
    logger.debug("is_newer")
    return not(is_older(fname, nmax_days_old))


def is_older(
        fname,
        nmax_days_old,
        time_before=None,
        ):
    """
    Return True if the file is older than nmax_days_old.
    """
    logger.debug("is_older")
    if time_before is None:
        time_before = datetime.now() - timedelta(days = nmax_days_old)
    logger.debug(f"\t time_before:{time_before}")
    time_file = datetime.fromtimestamp(os.path.getmtime(fname))
    logger.debug(f"\t date of the file:{time_file}")
    if time_file <= time_before:
        return True
    return False

def is_vowel(s):
    """
    Return True if is a vowel.
    """
    if set(s.upper()) & {"A","E", "I", "O", "U", "Y"}:
        return True
    return False


def keep_biggest_Li(LLi):
    """
    Gets the biggest list from LLi
    """
    LLi = sorted(LLi, key=len, reverse=True)
    n = len(LLi[0])
    LLi2 = []
    for Li in LLi:
        if len(Li) == n:
            LLi2.append(Li)
        else:
            break
    return LLi2


def keep_first_in_L(L):
    """
    Remove the duplicates and keep the order in L
    """
    Di = {i: 0 for i in iter(set(L))}
    L2 = []
    for x in L:
        if Di[x] == 0:
            L2.append(x)
        Di[x] += 1
    return L2


def keep_first_in_L_with_n_caracters(L, n):
    """
    Create a new list with n first elements different.
    """
    L = sorted(L)
    L2 = []
    last = ""
    for x in L:
        if x[:n] != last:
            L2.append(x)
            last = x[:n]
    return L2


def liberate_memory():
    """
    Liberates the memory.
    """
    logger.info("liberate_memory")
    gc.collect()


def load_s_from_txt(
        fname,
    ):
    logger.debug("load_Ls_from_txt")
    with open(fname, "r") as fr:
        Ls = fr.readlines()
    return Ls[0]



def load_Ls_from_txt(
        fname,
        begin=None,
        add_begin=True,
        s_end=None,
        add_end=True,
        i_min=0,
        i_max=-1,
        nmax=None,
        sep=",",
        LT=None, # list of  tuple (position, character to obtain.)
        s_searched=None,
        di=None, # to make jump of di lines
        ):
    """
    Gets a list of string Ls from a txt file.
    """
    logger.info("load_Ls_from_txt")
    # --------------------------------------------------------------------------
    # To extract all the lines
    # --------------------------------------------------------------------------
    def get_Ls_from_txt_with_nmax():
        Ls = []
        with open(fname, "r") as fr:
            n = 0
            for i,line in enumerate(fr):
                if di:
                    if i % di != 0:
                        continue
                Ls.append(line[:-1])
                n += 1
                if n == nmax:
                    return Ls
    # --------------------------------------------------------------------------
    def get_Ls_from_txt_with_nmax_and_s_searched():
        logger.info("get_Ls_from_txt_with_nmax_and_s_searched")
        Ls = []
        with open(fname, "r") as fr:
            n = 0
            for i,line in enumerate(fr):
                if di:
                    if i % di != 0:
                        continue
                if s_searched in line:
                    Ls.append(line[:-1])
                    n += 1
                    if n == nmax:
                        return Ls
    # --------------------------------------------------------------------------
    def get_Ls_from_txt_with_nmax_LT():
        logger.info("get_Ls_from_txt_with_nmax_LT")
        Ls = []
        with open(fname, "r") as fr:
            n = 0
            for i,line in enumerate(fr):
                if di:
                    if i % di != 0:
                        continue
                Ls2 = line[:-1].split(sep)
                ya = True
                for T in LT:
                    if Ls2[T[0]] != T[1]:
                        ya = False
                        break
                if ya:
                    Ls.append(line)
                    n += 1
                    if n == nmax:
                        return Ls
    # --------------------------------------------------------------------------
    if begin is None and s_end is None:
        if nmax is None:
            logger.info("nmax not defined")
            with open(fname, "r") as fp:
                Ls = fp.readlines()
            if len(Ls) == 0:
                return Ls
            Ls2 = [s[:-1] for s in Ls[:-1]]
            if Ls[-1].strip() == "":
                return Ls2
            else:
                return Ls2 + [Ls[-1][:-1]]
        else:
            logger.info(f"{nmax=}")
            if LT is not None:
                return get_Ls_from_txt_with_nmax_LT()
            elif s_searched is not None:
                return get_Ls_from_txt_with_nmax_and_s_searched()
            else:
                return get_Ls_from_txt_with_nmax()

    else:
        Ls = []
        if begin is not None:
            if s_end is None:
                # --------------------------------------------------------------
                # To extract the lines with a begin and no end
                # --------------------------------------------------------------
                with open(fname, "r") as fr:
                    for line in fr:
                        if line[:len(begin)] == begin:
                            break
                    if add_begin:
                        Ls.append(line[i_min:i_max])
                    for line in fr:
                        Ls.append(line[i_min:i_max])
                    return Ls
            else:
                # --------------------------------------------------------------
                # To extract the lines with a begin and a end
                # --------------------------------------------------------------
                with open(fname, "r") as fr:
                    for line in fr:
                        if line[:len(begin)] == begin:
                            break
                    if add_begin:
                        Ls.append(line[i_min:i_max])
                    for line in fr:
                        if line[:len(s_end)] == s_end:
                            if add_end:
                                Ls.append(line[i_min:i_max])
                            break
                        Ls.append(line[i_min:i_max])
                    return Ls

def load_LLvar(
        s,
        nLvar=None,    # to limit the number of files to read
        nmax_days_old=None,
        nothing=[],
        info=False,
        nmax=None,   # To limit the number of the elements
        ):
    """
    Load a list of pickles into a list.
    """
    path = get_path()
    logger.info(f"load_Lvar in {path}")
    if type(s) is list:
        Lfile = s
    else:
        if ".p" not in s:
            s += ".p"
        Lfile = get_Lfile(s, nmax_days_old=nmax_days_old)
    logger.info(f"\t{len(Lfile)} files")
    # ------------------------------------------------
    if nLvar is not None:
        Lfile = Lfile[:nLvar]
        logger.info(f"\t only {nLvar} files will be read")
    # ------------------------------------------------
    Lres = []
    nLfile = len(Lfile)    
    for i,file in enumerate(Lfile):
        if info:
            print(f"{i:>8}/{nLfile:<8}", end="\r")
        Lres.append(load_var(file, nothing=nothing))
        if nmax is not None:
            if len(Lres) > nmax:
                break
    return Lres               
                
                

def load_Lvar(
        s,
        nLvar=None,    # to limit the number of files to read
        Li=None, #list of indexes
        nmax_days_old=None,
        nothing=[],
        info=False,
        ):
    """
    Load a list of pickles into a list.
    """
    path = get_path()
    logger.info(f"load_Lvar in {path}")
    if type(s) is list:
        Lfile = s
    elif type(s) is str:
        Lfile = get_Lfile(s, nmax_days_old=nmax_days_old)

    logger.info(f"\t{len(Lfile)} files")
    # ------------------------------------------------
    if nLvar is not None:
        Lfile = Lfile[:nLvar]
        logger.info(f"\t only {nLvar} files will be read")
    elif Li is not None:
        Lfile = [Lfile[i] for i in Li]
        logger.info(f"\t only {len(Li)} files will be read")
    # ------------------------------------------------
    nLfile = len(Lfile)    
    Lres = []
    for i,file in enumerate(Lfile):
        if info:
            print(f"{i:>8}/{nLfile:<8}", end="\r")
        Lres.append(load_var(file, nothing=nothing))
    return Lres



def load_var(
        fname,
        nothing=None,
        ):
    """
    the load a variable recorded into a pickle
    :param var: string, variable to dump.
    :returns: a variable
    """
    logger.debug(f"load_var {fname}")
    # --------------------------------------------------------------------------
    if ".p.gz" in fname:
        with gzip.open(fname,"rb") as fgz:
            return pickle.load(fgz)
    # --------------------------------------------------------------------------
    if ".p" not in fname:
        fname += ".p"
    # --------------------------------------------------------------------------
    if yapa(fname):
        logger.warning(f"\t pickle {fname} not found!")
        return 
    try:
        with open(fname, "rb") as fw:
            return pickle.load(fw)
    except pickle.UnpicklingError as e:
        #logger.debug(traceback.format_exc(e))
        return 
    except (AttributeError,  EOFError, ImportError, IndexError) as e:
        # secondary errors
        #logger.debug(traceback.format_exc())
        return






def make_Ldir(
        Lpath
    ):
    for path in Lpath:
        make_dir(path)


def make_dir(path):
    """
    Creates a directory.
    """
    if ya(path):
        remove_dir(path)
        time.sleep(0.1)
    os.mkdir(path)


def modify_section_in_file(
        file,
        s0,          # string to replace with deb and end
        ):
    """
    Modify a section defined by a begin and a end.
    The section waited is as follow:
    s0=\"""
SECTION SOLVENT
    FILE water_rdock.pdb
    TRANS_MODE FIXED
    ROT_MODE FIXED
    OCCUPANCY 0.5
END_SECTION\"""
    """
    logger.info("modify_section_in_file")
    # --------------------------------------------------------------------------
    Ls = s0.split("\n")
    for i, s in enumerate(Ls):
        if s != "":
            break
    Ls = Ls[i:]
    file_new = file.replace(".", "_new.")
    with open(file, "r") as fr, \
         open(file_new, "w") as fw:
         do_write = True
         for line in fr:
            if line[:-1] == Ls[0]:
                for s in Ls:
                    fw.write(s+"\n")
                do_write = False
            elif do_write == False:
                if line[:-1] == Ls[-1]:
                    do_write = True
            elif do_write:
                fw.write(line)
    remove_file(file)
    rename_file(file_new, file)


def modify_section_in_Lfile(
        Lfile,
        s0,
        ):
    """
    Modification of a section for a list of file.
    The section waited is as follow:
    s0=\"""
SECTION SOLVENT
    FILE water_rdock.pdb
    TRANS_MODE FIXED
    ROT_MODE FIXED
    OCCUPANCY 0.5
END_SECTION\"""
    """
    logger.info("modify_section_in_file")
    # --------------------------------------------------------------------------
    for file in Lfile:
        modify_section_in_file(
            file,
            s0,
            )


def move_file(
        fname,
        path,
        ):
    fname2 = f"{path}/{fname}"
    if ya(fname2):
        logger.info("\t remove of previous file")
        remove_file(fname2)
    shutil.move(fname, f"{path}")


def move_Lfile_in_dir(
        Lfile,
        dir,
        ):
    if yapa(dir):
        make_dir(dir)
    if type(Lfile) is str:
        Lfile = get_Lfile(Lfile)
    for file in Lfile:
        shutil.move(file, f"{dir}/{file}")


def move_dir(
        dir=None, 
        path=None, 
        overwrite=False
        ):
    """
    Moves a list of subdirectory into path.
    """
    if path is None:
        path = get_path()
    cdir = dir.split("/")[-1]
    dir2 = f"{path}/{cdir}"
    if ya(dir2):
        if overwrite:
            os.system(f"rm -r {dir2}")
        else:
            logger.info(f"\t {dir2} exist!")
            return
    bash(f"mv {cdir} {path}")


def move_Ldir(
        Ldir=None, 
        path=None, 
        overwrite=False
        ):
    """
    Moves a list of subdirectory into path.
    """
    if path is None:
        path = get_path()
    for dir in Ldir:
        move_dir( 
                    dir,
                    path,
                    overwrite,
                )

def remove_file(
        fname
        ):
    """
    Remove a file named fname.
    """
    if ya(fname):
        os.remove(fname)


def remove_Lfile_in_dir(
        xdir,
        Lfile="",
    ):
    path0 = get_path()
    remove_Lfile(Lfile)
    os.chdir(path0)


def remove_Lfile(
        Lfile=""
    ):
    """
    Removes a list of files
    """
    logger.info("remove_Lfile")
    if type(Lfile) is str:
        Lfile = get_Lfile(Lfile)
    for file in Lfile:
        remove_file(file)


def remove_lines_with_s(
        fname, 
        s=""
        ):
    """
    Removes the lines containing s in fname.
    """
    fout = fname.replace(".", "_new.")
    with    open(fname, "r") as fr, \
            open(fout, "w") as fw:
        for line in fr:
            if s not in line:
                fw.write(line)
    os.rename(fout, fname)


def rename_all_dname_with(s_old, s_new):
    """
    Rename all the files having s_old in the their name with s_new at the place of old.
    """
    Lsdir = get_Lsdir(s_sub=s_old)
    for dir in Lsdir:
        dir2 = dir.replace(s_old, s_new)
        os.rename(dir, dir2)


def rename_all_fname_with(s_old, s_new):
    """
    Rename all the files having old in the their name with new at the place of old.
    """
    Lfile = get_Lfile("*" + s_old + "*")
    for f in Lfile:
        f2 = f.replace(s_old, s_new)
        os.rename(f, f2)


def rename_file(s_old, s_new):
    if ya(s_old):
        os.rename(s_old, s_new)


def rename_file_in_Ldir(
            s1,
            s2,
            dname="*",
            fname="*",
    ):
    """
    Rename the files in all the sub-directories, replacing s1 by s2
    """
    for dir in get_Ldir(dname):
        os.chdir(dir)
        for file in get_Lfile(fname):
            file2 = file.replace(s1, s2)
            rename_file(file, file2)
        os.chdir("..")


def rename_Lfile(
        Lfile=None,
        s_old=None,
        s_new=None,
        s_to_add_after=None,
        s_to_add_before=None,
        to_exchange=False,  #to not delete the existing files if they have the same name.
        ):
    """
    rename a list of file by changing the same string.
    """
    logger.info("rename_Lfile")
    if Lfile is None:
        Lfile = get_Lfile()
    elif type(Lfile) is str:
        Lfile = get_Lfile(Lfile)
    if to_exchange:
        for file in Lfile:
            if s_old in file:
                file2 = file.replace(s_old, s_new)
                if yapa(file2):
                    os.rename(file, file2)
                else:
                    file2 = file2.replace(".", "_to_remove")
                    os.rename(file, file2)
    else:
        if s_old is not None:
            for file in Lfile:
                if s_old in file:
                    file2 = file.replace(s_old, s_new)
                    os.rename(file, file2)
        elif s_to_add_after is not None:
            for file in Lfile:
                os.rename(file, file + s_to_add_after)
        elif s_to_add_before is not None:
            for file in Lfile:
                os.rename(file, s_to_add_before + file)


def rename_Lfile_index(
        Lfile,
        sep="_",
        n=2, #index of the part to modify
        n0=4, #number of zeros to apply
        ):
    """
    rename the index in the Lfile
    """
    if Lfile is None:
        Lfile = get_Lfile()
    elif type(Lfile) is str:
        Lfile = get_Lfile(Lfile)
    # --------------------------------
    for file in Lfile:
        end = ""
        if "." in file:
            file2, end = file.split(".")
        L2 = file2.split(sep)
        i = int(L2[n])
        L2[n] = f"{i:0>{n0}}"
        file2 = "_".join(L2)
        file2 = f"{file2}.{end}"
        if yapa(file2):
            os.rename(file, file2)
        else:
            file2 = file2.replace(".", "_to_remove")
            os.rename(file, file2)


def rename_Ldir(
        s_old=None,
        s_new=None,
        Ldir=None,
        to_exchange=False,  #to not delete the existing files if they have the same name.
        ):
    """
    rename a list of directories by changing the same string.
    """
    if Ldir is None:
        Ldir = get_Ldir()
    elif type(Lfile) is str:
        Ldir = get_Ldir(Ldir)
    for dirx in Ldir:
        if s_old in dirx:
            dirx2 = dirx.replace(s_old, s_new)
            os.rename(dirx, dirx2)


def rename_dir(
        dir,
        dir2,
    ):
    os.rename(dir, dir2)


def repeat_L(L, n):
    """
    Repeats L in order to have n elements in the new list.
    """
    (k, r) = divmod(n, len(L))
    L2 = L*k+L[:r]
    return L2


def replace_cs_in_file(cs, fname, fout=None):
    """
    :param fname: name of file to modify,
    :param Lc_s: list of tuples of string defining the replacement to do.
    :param fout: name of the new file if not None
    """
    logger.debug("replace_cs_in_file")

    if fout is None:
        cmd = "sed -i "
        cmd += f" -r 's/{cs[0]}/{cs[1]}/g' "
        cmd += f" {fname}"
    else:
        cmd = "sed "
        cmd += f" -e 's/{cs[0]}/{cs[1]}/g' "
        cmd += f" {fname}"
        cmd += f" < {fname} > {fout}"
    logger.debug(f"\t {cmd}")
    os.system(cmd)


def replace_in_file(
        fname=None,
        s1="\t",
        s2="    ",
        fname_new=None,
        ):
    """
    Replace s1 by s2 in fname.
    If fname_new is not given, the initial file is modified.
    """
    logger.info("replace_in_file")
    # --------------------------------------------------------------------------
    to_replace = False
    if fname_new is None:
        fname_new = fname.replace(".", "_old.")
        to_replace = True
    with open(fname, "r") as fr, \
        open(fname_new, "w") as fw:
        do_write = True
        for line in fr:
            fw.write(line.replace(s1, s2))
    if to_replace:
        remove_file(fname)
        rename_file(fname_new, fname)


def replace_in_dir(
        xdir,
        s1="\t",
        s2="    ",
        file=""
    ):
    """
    Replae s1 by s2 in all files in xdir.
    """
    path0 = get_path()
    os.chdir(xdir)
    for file in get_Lfile(file):
        replace_in_file(
                            file,
                            s1,
                            s2,
                        )
    os.chdir(path0)


def replace_in_Ldir(
        Ldir,
        s1="\t",
        s2="    ",
        file=""
    ):  
    for xdir in Ldir:
        replace_in_dir(
                            xdir,
                            s1,
                            s2,
                            file,
                    )



def remove_dir(dir):
    """
    Removes a directory.
    """
    if ya(dir):
        os.system(f"rm -rf {dir}")
        #shutil.rmtree(dir, ignore_errors=True)


def remove_Ldir(
        Ldir,
        path=".",
    ):
    """
    Removes a list of directories.
    """
    if type(Ldir) is str:
        bashr(f"find {path}/ -name {Ldir} -exec rm -rf {{}} \;", info=True)
    else:
        for dir in Ldir:
            remove_dir(dir)



def sort_LT_by_extremities(
        LT,
        n=1, # number of indices that make the extremity
        ):
    """
    sort a list of tuples into a list where the last value a tuple is the first
    of the newt one.
    This method must be redone with networkx.
    """
    logger.debug("sort_LT_by_extremities")
    # ==========================================================================
    # Versiion 1
    # ==========================================================================
    DD = dict()
    for i,T in enumerate(LT):
        if T[:n] not in DD.keys():
            DD[T[:n]] = {"left": i}
        else:
            DD[T[:n]]["left"] = i
        if T[-n:] not in DD.keys():
            DD[T[-n:]] = {"right": i}
        else:
            DD[T[-n:]]["right"] = i
    logger.debug(f"\t DD={DD}")
    # --------------------------------------------------------------------------
    n_left = None
    for k,D in DD.items():
        if "right" not in D.keys():
            n_left = k
            logger.debug(f"\t n_left={n_left}")

    if n_left is None:
        logger.info("\t Cycle!")
        return None
    # --------------------------------------------------------------------------
    LT2 = []
    while True:
        try:
            i = DD[n_left]["left"]
            T = LT[i]
            n_left = T[-n:]
            LT2.append(T)
        except:
            break
    # --------------------------------------------------------------------------
    return LT2

    # ==========================================================================
    # Versiion
    # ==========================================================================
    # logger.debug(f"\t LT={LT}")
    # Dd = {}
    # Df = {}
    # for T in LT:
    #     Dd[T[0]] = T
    #     Df[T[-1]] = T
    # logger.debug(f"\t Dd={Dd}")
    # logger.debug(f"\t Df={Df}")
    # # -------------------------------------------------------------------------
    # Li = list(set(Dd.keys()) - set(Df.keys()))
    # logger.debug(f"\t Li={Li}")
    # LT2 = []
    # n = len(LT)
    # k = Li[0]
    # for i in range(n):
    #     T = Dd[k]
    #     k = T[-1]
    #     LT2.append(T)
    # return LT2

    # ==========================================================================
    # Versiion 3
    # ==========================================================================
    # D1 = {}
    # D2 = {}
    # for L in LL:
    #     D1[tuple(L[:n])] = L
    #     D2[tuple(L[-n:])] = L
    # LL2 = deque([LL[0]])
    # while True:
    #     t = tuple(LL2[-1][-n:])
    #     if t in list(D1.keys()):
    #         LL2.append(D1[t])
    #     else:
    #         breaksplit_L_in_LL
    # while True:
    #     t = tuple(LL2[0][:n])
    #     if t in list(D2.keys()):
    #         LL2.appendleft(D2[t])
    #     else:
    #         break
    # return list(LL2)


def split_L_in_LL(
            L0,
            nLL=None,
            nL=1,            # number of element per list,
            nLL_max=None,        # Number max of list.
            nLL_min=None,
            with_nL_impose=True,
            niv_log=0,
            ):
    """
    Splits a list L into a list of list LL.
    :param nLL: number of list,
    :param nL: number of element per list,
    ;param nL_max: interger. Number max of list.
    :param with_N_impose: boolean, to impose nL elements a list.
    :returns : list of lists.
    """
    inde = "\t" * niv_log
    logger.info(f"{inde}split_L_in_LL")
    niv_log += 1
    inde = "\t" * niv_log
    # ---------------------------------------------
    if type(L0) is set:
        L0 = list(L0)
    nL0 = len(L0)
    logger.info(f"{inde}with nL0={nL0}")
    if nL0 == 0:
        return []
    elif nL0 == 1:
        return [L0]
    if nLL_min:
        if nL0 <= nLL_min:
            return [[x] for x in L0]
    if (nLL_min is not None) and (nLL_max is not None):
        if nLL_min > nLL_max:
            raise Exception("nLL_min > nLL_max")
    # --------------------------------------------
    if nLL:
        logger.info(f"{inde}with nLL={nLL}")
    else:
        logger.info(f"{inde}with nL={nL}")
        if nLL_min:
            logger.info(f"{inde}with nLL_min={nLL_min}")
            if int(nL0/nL) < nLL_min:
                nL = int(nL0/nLL_min)
                logger.info(f"{inde}with nL={nL}")
            elif nLL_max:
                logger.info(f"{inde}with nLL_max={nLL_max}")
                if int(nL0/nL) > nLL_max:
                    nL = int(nL0/nLL_max) + 1
                logger.info(f"{inde}with nL={nL}")
        elif nLL_max:
            logger.info(f"{inde}with nLL_max={nLL_max}")
            if int(nL0/nL) > nLL_max:
                nL = int(nL0/nLL_max) + 1
            logger.info(f"{inde}with nL={nL}")

    # --------------------------------------------------------------------------
    def get_LL_with_nLL(nLL):
        logger.debug(f"{inde}get_LL_with_nLL")
        if len(L0) <= nLL:
            logger.debug(f"{inde}len(L0) <= nLL")
            return [[x] for x in L0]
        k, m = divmod(nL0, nLL)
        LL = [L0[i * k + min(i, m):(i + 1) * k + min(i + 1, m)]
              for i in range(nLL)]
        logger.info(f"{inde}{len(LL)=}")
        return LL
    
    def get_LL_with_nL(nL): 
        logger.debug(f"{inde}get_LL_with_nL")
        k, m = divmod(nL0, nL)
        if with_nL_impose:
            logger.debug(f"{inde}with with_nL_impose is True")
            LL = [L0[i * nL:(i + 1) * nL] for i in range(k)]
            if m > 0:
                LL.append(L0[-m:])
        else:
            logger.debug(f"{inde}with with_nL_impose is False")
            if m == 0:
                nLL = k
                k, m = divmod(nL0, nLL)
                LL = [L0[i * k + min(i, m):(i + 1) * k + min(i + 1, m)]
                      for i in range(nLL)]
            else:
                nLL = k + 1
                k, m = divmod(nL0, nLL)
                LL = [L0[i * k + min(i, m):(i + 1) * k + min(i + 1, m)]
                      for i in range(nLL)]
        logger.info(f"{inde}{len(LL)=}")
        return LL
    # --------------------------------------------------------------------------
    nL0 = len(L0)
    if nLL is not None:
        LL = get_LL_with_nLL(nLL)
    elif nL is not None:
        k, m = divmod(nL0, nL)
        logger.debug(f"{inde}k={k}, m={m}")
        if nLL_max:
            if k > nLL_max:
                return get_LL_with_nLL(nLL_max)
        if nLL_min:
            if k < nLL_min:
                return get_LL_with_nLL(nLL_min)
        LL = get_LL_with_nL(nL)
    logger.info(f"{inde}len(LL)={len(LL)}")
    return LL
        




def split_L_into_LL_from_Li(L, Li=[1]):
    """
    Splits a list L into a list of list LL, on index defined in Li
    """
    logger.debug("split_L_into_LL_from_Li")
    # --------------------------------------------------------------------------
    LL = [L[:Li[0]]]
    # --------------------------------------------------------------------------
    if len(Li) > 2:
        for i, id in enumerate(Li[1:]):
            LL.append(L[Li[i]:Li[i+1]])
    # --------------------------------------------------------------------------
    LL.append(L[Li[-1]:])
    return LL


def split_L_into_LL_with_overlap(L, n=2, dn=1):
    """
    Splits a list L into a list of list by using dn elements as overlap.

    Args:
        L: list
        n: interger, number of element in the new lists
        dn: number of elements that make the ovelap between two consecutive
        elements in the new list.

    Returns:
        list of list.
    """
    k = n - dn
    nL = len(L) - dn
    LL = []
    i1 = 0
    while True:
        LL.append(L[i1:i1 + n])
        i1 += k
        if i1 >= nL:
            break
    return LL


def split_s_on_cr_brackout(s, cr):
    """
    split the string on the characters cr if cr is out of brackets (, {, [.
    """
    Li = get_Li_from_cr_brackout_in_s(cr, s)
    return split_s_on_Li(s, Li)


def split_s_on_Li(s, Li):
    """
    split the string s at i in Li.
    """
    if len(Li) == 0:
        return [s]
    # --------------------------------------------------------------------------
    n = len(s)
    if Li[0] == 0:
        Li = Li[1:]
        s = s[1:]
        Li = list(np.array(Li) -1)
    # --------------------------------------------------------------------------
    if len(Li) == 0:
        return [s]
    # --------------------------------------------------------------------------
    n = len(s)
    if Li[-1] == n -1:
        Li = Li[:-1]
        s=s[:-1]
    n = len(Li)
    # --------------------------------------------------------------------------
    if n == 0:
        return [s]
    # --------------------------------------------------------------------------
    Ls = [s[:Li[0]]]
    if n > 1:
        k=0
        while True:
            s2 = s[Li[k]+1:Li[k+1]]
            Ls.append(s2)
            k += 1
            if k >= n -1 :
                break
    Ls.append(s[Li[-1]+1:])
    return Ls


def split_string(
        s, # string)
        sep_string="\"",
        sep="\t",
        ):
    """
    Splits a string with strings inside to preserve.
    """
    logger.debug(s)
    Ls1 = s.split(sep_string)
    logger.debug(Ls1)
    Ls2 = []
    for i,s1 in enumerate(Ls1):
        if i%2:
            Ls2.append(s1)
        else:
            if s1 == sep:
                continue
            elif s1.strip() == "":
                continue
            else:
                if s1[0] == sep:
                    s1 = s1[1:]
                if s1[-1]== sep:
                    s1 = s1[:-1]
                Ls2 += s1.split(sep)
    return Ls2


def ssplit(s0, sep=";"):
    """
    splits a string into a list of string with remove of spaces
    """
    return [s.strip() for s in s0.strip().split(sep) if s]



import signal
from contextlib import contextmanager

class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        


def touch(fname):
    """
    Short-cut for touch bash command
    """
    if ya(fname):
        remove_file(fname)
    bash(f"touch {fname}")


def value_is_in_range(value, range_min, range_max):
    """
    Counts the number of values between range
    """
    if range_min <= value <= range_max:
        return True
    else:
        return False


def ya(fname, nmax_days_old=None):
    """
    Tests if the fname file exists.
    """
    #logger.debug("ya")
    # --------------------------------------------------------------------------
    if "*" in fname:
        Lfile =  get_Lfile(fname, nmax_days_old=nmax_days_old)
        nLfile = len(Lfile)
        if nLfile == 0:
            return False
        return True
    else:
        if os.path.exists(fname):
            if nmax_days_old:
                if is_newer(fname, nmax_days_old):
                    return True
                return False
            return True
        return False



def ya_fname(dir, fname):
    return ya(f"{dir}/{fname}")


def yapa(fname, nmax_days_old=None):
    """
    Tests if the fname file don't exist.
    """
    #logger.debug("yapa")
    # --------------------------------------------------------------------------
    return not(ya(fname, nmax_days_old))


def zip_Ldir(Ldir, zname=None):
    if zname is None:
        zname = get_s_from_object(Ldir) + ".zip"
    create_txt_from_L(Ldir, "Ldir.txt")
    bash(f"zip {zname} -r@ < Ldir.txt")


def zip_Lfile_from_Ldir(
        Ldir=None,
        Lfile=None,
        zname=None,
        ):
    """
    Create a zip file with only Lfile from Ldir
    """
    if Ldir is None:
        return "No Ldir"
    if Lfile is None:
        return "No Lfile"
    # --------------------------------------------------------------------------
    nLdir = len(Ldir)
    for i,dir in enumerate(Ldir):
        print(f"zip_Lfile_from_Ldir {i:>6}/{nLdir:<6}", end="\r")
        for file in Lfile:
            bash(f"cp --parent {dir}/{file} ./to_zip")




