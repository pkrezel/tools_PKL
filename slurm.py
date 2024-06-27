#! /usr/bin/env python
# -*- coding: UTF-8 -*-
import os, sys
import logging
logger = logging.getLogger(__name__)
#=======================================================================
import os, sys
#=======================================================================
from tools_PKL.basic import (create_txt_from_L)
#=======================================================================


def create_slurm_job(
        file_to_run,
        with_GNU_parallel=True,
        ref="job_by_PKL",
        fname="job.sh",
        queue="cpu",
        n_core=40,
        Lnode_to_exclude=None,
        user_cluster="pkrezel",
        conda_env="SBC_3.9",
        nLs=100, # number of independant tasks
        add_short=False,
        cmd_to_insert_in_0="",
        cmd_to_insert_in_1="",
        cmd_to_insert_in_2="",
        s_parallel="parallel",
    ):
    """
    """
    info1 = ""
    if Lnode_to_exclude is not None:
        if len(Lnode_to_exclude):
            info1 = f"#SBATCH --exclude={','.join(Lnode_to_exclude)}"
    if queue == "short":
        add_short = True
    # --------------------------------------------------------------------------
    info2 = ""
    info3 = "srun='srun --exclusive -n1 -N1'"
    time = "#SBATCH --time=96:00:00"
    if add_short:
        info2="#SBATCH --qos=short"
        time = "#SBATCH --time=1:00:00"
        queue = "short"
        info3 = "srun='srun --exclusive -n1'"
    # -------------------------------------------------------------------------
    if with_GNU_parallel:
        Ls = f"""
#!/bin/bash
{time}
#SBATCH -J '{ref}'
#SBATCH --partition={queue}
#SBATCH --ntasks={n_core}
{info2}
{info1}

source /home/{user_cluster}/.bashrc
conda activate {conda_env}
{cmd_to_insert_in_0}
{cmd_to_insert_in_1}
nLs=$(head -1  nLs.txt)
chmod +x {file_to_run}
srun='srun -n1 -N1 --exclusive '
parallel="parallel --delay .2 -j $SLURM_NTASKS "
$parallel "$srun python {file_to_run} {{1}} $nLs > {s_parallel}_{{1}}.log" ::: $(seq 1 $nLs)
{cmd_to_insert_in_2}
""".split("\n")
    else:
        Ls = f"""
#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH -J '{ref}'
#SBATCH --partition={queue}
#SBATCH --n_core={n_core}
{info1}
{info2}

#SBATCH --array=1-$nLs%{n_core}

source /home/{user_cluster}/.bashrc
conda activate {conda_env}
{cmd_to_insert_in_0}
{cmd_to_insert_in_1}
LINE=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" Ls.txt)
srun python {file_to_run} $LINE
{cmd_to_insert_in_2}
""".split("\n")
    #for file in proj*; do scp  $file  {user_local}@{IP_local}:{path_local_ref} && rm $file; done
    create_txt_from_L(Ls[1:], fname)