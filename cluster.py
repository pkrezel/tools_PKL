#! /usr/bin/env python
# -*- coding: UTF-8 -*-
import logging
logger = logging.getLogger(__name__)
# ==============================================================================
import os, sys
import time
import inspect
# ==============================================================================
from tools_PKL.basic import (create_txt_from_L, get_path, ya, yapa, remove_dir,
    dump_var, load_var, bash, bashr, get_Lrand, split_L_in_LL, dump_Lvar,
    create_txt_from_s, fuse_LL, make_dir, load_Lvar, concat_Lfile, copy_file_in_dir,
    get_s_from_object)
from tools_PKL.ssh import bash_on_remote
from tools_PKL.slurm import create_slurm_job
# ==============================================================================


#test2



def do_on_cluster(
    f0=None,
    s1_py=None,
    user_local="krezel@icoa-03.local",
    IP_local="192.168.11.160",
    # ----------------------------
    Dcluster=None,
    user_cluster=None,
    IP_cluster=None,
    conda_env=None,
    # ----
    queue=None,
    cmd_smina=None,
    # ----------------------------
    with_GNU_parallel=True,
    ref=None,
    level="INFO", # the logging level on the cluster
    nLmin_by_job=200, # Number min of cases to treat by job
    ncore_max=80, # number max of core to use
    njob_max=1000,
    dir_Lres="dir_Lres",
    with_return_Lres=True,
    with_fuse_LL=True, # to fuse the differents results of the parallelization
    Lmodule_to_update=None,
    ):
    # -----------------------------------------------
    logger.info("do_on_cluster")
    if Dcluster is not None:
        user_cluster = Dcluster["user_cluster"]
        IP_cluster = Dcluster["IP_cluster"]
        conda_env = Dcluster["conda_env"]
        cmd_smina = Dcluster["cmd_smina"]
        queue = Dcluster["queue"]
    if ref is None:
        ref = f0.__name__
    def decorator(func):
        def wrapped(*args,s1_py = s1_py, **kwargs):
            # --------------------------------------------------------------------
            update_module_on_cluster(
                Lmodule_to_update,
                IP_cluster,
                user_cluster,
            )
            func(*args, **kwargs)
            # --------------------------------------------------------------------------
            Lkw = list(kwargs.keys())
            # --------------------------------------------------------------------------
            path_cluster = f"/home/{user_cluster}"
            path_local = get_path()
            logger.info(f"* path_local={path_local}")
            # to be sure that the same job is not still running
            # --------------------------------------------------------------------------
            cmd = f"scancel --user {user_cluster}--name {ref}"
            bash_on_remote(
                                cmd,
                                user_remote=user_cluster,
                                IP_remote=IP_cluster,
                            )
            # --------------------------------------------------------------------------
            logger.info(f"Remove of directory {ref} on cluster ...")
            Lcmd = [f"rm -rf {path_cluster}/{ref}"]
            bash_on_remote(
                                Lcmd, 
                                user_cluster, 
                                IP_cluster,
                            )
            # --------------------------------------------------------------------------
            logger.info(f"Remove of directory of results in local")
            remove_dir(dir_Lres)
            # ========================================================================
            # Creation of a new directory to gather all the informations
            # usefull to launch the script on the cluster.
            # --------------------------------------------------------------------------
            logger.info(f"Creation of the directory to send")
            make_dir(ref)
            logger.info("copy of variables")
            if "LTvar_to_copy"  in Lkw:
                LTvar_to_copy = kwargs["LTvar_to_copy"]
                for T in LTvar_to_copy:
                    if type(T) is str:
                        logger.info(f"\t copy of {T}")
                        copy_file_in_dir(T, ref)
                    else:
                        logger.info(f"copy of {T[0]}")
                        dump_var(T[1], f"{ref}/{T[0]}")
            os.chdir(ref)
            make_dir(dir_Lres)
            path_local_ref = get_path()
            # --------------------------------------------------------------------------
            LL = split_L_in_LL(
                                args[0],
                                nL=nLmin_by_job,
                                nLL_max=njob_max,
                                )
        
            if len(LL) < ncore_max:
                n_core = len(LL)
            else:
                n_core = ncore_max
            Ls = dump_Lvar(LL, "L")
            create_txt_from_L(Ls, "Ls.txt")
            nLs = len(Ls)
            create_txt_from_s(str(nLs), "nLs.txt")
            # --------------------------------------------------------------------------
            # Dump of informations that can't be pass directly
            # --------------------------------------------------------------------------
                
            # ========================================================================
            # Creation of job_all.sh to launch of the different jobs
            # can be sbatch or bash
            # --------------------------------------------------------------------------
            Ls = f"""
#!/bin/bash
sbatch --wait job1.sh
wait
    """.split("\n")
            create_txt_from_L(Ls[1:], "job_all.sh")
            # ========================================================================
            # Creation of job1.sh slurm file
            # --------------------------------------------------------------------------
            create_slurm_job(
                                "do1.py",
                                with_GNU_parallel=with_GNU_parallel,
                                ref=ref,
                                queue=queue,
                                fname="job1.sh",
                                n_core=n_core,
                                user_cluster=user_cluster,
                                conda_env=conda_env,
                                cmd_to_insert_in_0=f"cd /home/{user_cluster}/{ref}",
                                s_parallel="parallel1",
                                )
            # --------------------------------------------------------------------------
            # do1.py file
            # --------------------------------------------------------------------------
            # --------------------------------------------------------------------------
            if f0:
                module = f0.__module__
                sf0 = f0.__name__
                s1_py = ""
                s_py = f"""
import sys
import os
import traceback
sys.path.insert(0, "/home/{user_cluster}/modules")
# =============================================================================
from tools_SBC.logging_SBC import  (set_log, logger)
from tools_PKL.basic import (load_var, dump_var, get_line_n_in_file, move_file, touch,
                remove_file, remove_Lfile)
# =============================================================================
from {module} import {sf0}
# ------------------------------------------------------------------------------
i_line = int(sys.argv[1])
file = get_line_n_in_file("Ls.txt", i_line)
Lx = load_var(file)
exec("Lx2 = [{sf0}(x) for x in Lx]")
dump_var(Lx2, f"{dir_Lres}/{{file}}")
remove_file(file)
"""
            elif s1_py :
                s_py = f"""
import sys
import os
import traceback
sys.path.insert(0, "/home/{user_cluster}/modules")
# ==============================================================================
from tools_PKL.basic import (load_var)
"""
            Ls2 = []
            if "LTvar_to_copy"  in Lkw:
                for T in LTvar_to_copy:
                    if "." not in T[0]:
                        Ls2.append(f"{T[0]} = load_var('{T[0]}')")
            # -----------------------------------------------------------------------
            Ls = s_py.split("\n")[1:] + Ls2 + s1_py.split("\n")[1:]
            create_txt_from_L(Ls, "do1.py")
            # ========================================================================
            # Export of the directory on the cluster and launch of the job
            # -------------------------------------------------------------------------
            os.chdir("..")
            logger.info(f"Transfert  of {ref} on cluster ...")
            bash(
                            f"scp -C -r {ref} {user_cluster}@{IP_cluster}:{path_cluster}", 
                            info=True,
                )
            # --------------------------------------------------------------------------
            # Launch of the jobs
            # -------------------------------------------------------------------------
            logger.info(f"Launch of jobs ...")
            Lcmd = [f"cd {path_cluster}/{ref}",
                    "bash job_all.sh"]
            bash_on_remote(
                            Lcmd, 
                            user_cluster, 
                            IP_cluster,
                        )
            logger.info(f"on going ...")
            logger.debug(f"wait for results")
            start_time = time.time()
            # --------------------------------------------------------------------------
            logger.info(f"done in {round(time.time() - start_time)} s")
            # --------------------------------------------------------------------------
            # Loading of results
            # -------------------------------------------------------------------------
            logger.info(f"loading of the results")
            os.chdir(path_local_ref)
            remove_dir(dir_Lres)
            bash(
                            f"scp -C -r {user_cluster}@{IP_cluster}:{path_cluster}/{ref}/{dir_Lres} {path_local_ref}", 
                            info=True,
                        )
            # ------------------------------------------------------------------------

            Lcmd = [f"rm -rf {path_cluster}/{ref}"]
            #bash_on_remote(Lcmd, user_cluster, IP_cluster)
            if with_return_Lres:
                os.chdir(dir_Lres)
                LLres = load_Lvar("L*")
                os.chdir(path_local)
                if with_fuse_LL:
                    return fuse_LL(LLres)
                else:
                    return LLres
            os.chdir(path_local)
        return wrapped
    return decorator







def update_module_on_cluster(
    project="tools_SBC",
    IP_cluster=None,
    user_cluster=None,
    user_local="krezel@ICOA-03.LOCAL",
    ):
    if type(project) is not list:
        Lproject = [project]
    else:
        Lproject = project
    for project in Lproject:
        cmd = (f"rsync -e ssh -az"
               #f" --exclude-from=/home/{user_local}/Documents/Programmes/Gitlab/tools_SBC/rsync_exclude.txt"
               f" --delete-after {project} {user_cluster}@{IP_cluster}:/home/{user_cluster}/modules")
        bashr(cmd, info=True)




# =======================================================================================================================================================



def treat_on_cluster(
        Lmol,
        s1_py = "",
        # ----------------------------
        user_local="krezel@icoa-03.local",
        IP_local="192.168.11.160",
        # ----------------------------
        Dcluster=None,
        user_cluster="krezel",
        IP_cluster="192.168.11.17",
        cmd_smina="smina.static",
        conda_env="SBC",
        queue="cpu",
        # ----------------------------
        with_GNU_parallel=True,
        ref="exemple",
        level="INFO", # the logging level on the cluster
        nLmin_by_job=200, # Number min of cases to treat by job
        nmax_core=80, # number max of core to use
        nmax_job=1000,
        dir_Lres="dir_Lres",
        ):

    logger.info("minimize_Lmol_with_smina_on_cluster")
    # --------------------------------------------------------------------------
    if Dcluster is not None:
        user_cluster = Dcluster["user_cluster"]
        IP_cluster = Dcluster["IP_cluster"]
        conda_env = Dcluster["conda_env"]
        cmd_smina = Dcluster["cmd_smina"]
        queue = Dcluster["queue"]
    # --------------------------------------------------------------------------  
    path_cluster = f"/home/{user_cluster}"
    path_local = get_path()
    logger.info(f"* path_local={path_local}")
    # --------------------------------------------------------------------------
    # to be sure that the same job is not still running
    # --------------------------------------------------------------------------
    cmd = f"scancel --user {user_cluster}--name {ref}"
    bash_on_remote(
                        cmd,
                        user_remote=user_cluster,
                        IP_remote=IP_cluster,
                    )
    # --------------------------------------------------------------------------
    logger.info(f"Remove of directory {ref} on cluster ...")
    Lcmd = [f"rm -rf {path_cluster}/{ref}"]
    bash_on_remote(
                        Lcmd, 
                        user_cluster, 
                        IP_cluster,
                    )
    # ========================================================================
    logger.info("\t Creation of the repertories in local ...")
    make_dir(ref)
    os.chdir(ref)
    path_local_ref = get_path()
    make_dir(dir_Lres)
    if type(Lmol) is str:
        Lmol = load_Lmol_from_sdf(Lmol)
    nLmol = len(Lmol)
    logger.info(f"\t {nLmol} molecules to minimize")
    if ".pdb" in protein:
        protein = protein[:-4]
    os.system(f"cp ../{protein}.pdb {protein}.pdb")
    # -------------------------------------------------------------------------
    LL0 = split_L_in_LL(
                        L0,
                        nL=nLmin_by_job,
                        nLL_max=nmax_job,
                        )
    if len(LLmol) < nmax_core:
        n_core = len(LLmol)
    else:
        n_core = nmax_core
    Ls = dump_Lvar(LL0, "L")
    create_txt_from_L(Ls, "Ls.txt")
    nLs = len(Ls)
    create_txt_from_s(str(nLs), "nLs.txt")
    # ========================================================================
    # Creation of job_all.sh to launch of the different jobs
    # can be sbatch or bash
    # --------------------------------------------------------------------------
    Ls = f"""
#!/bin/bash
sbatch --wait job1.sh
wait
""".split("\n")
    create_txt_from_L(Ls[1:], "job_all.sh")
    # ========================================================================
    # Creation of job1.sh slurm file
    # --------------------------------------------------------------------------
    create_slurm_job(
                        "do1.py",
                        with_GNU_parallel=with_GNU_parallel,
                        ref=ref,
                        queue=queue,
                        fname="job1.sh",
                        n_core=n_core,
                        #Lnode_to_exclude=Lnode_to_exclude,
                        user_cluster=user_cluster,
                        conda_env=conda_env,
                        cmd_to_insert_in_0=f"cd /home/{user_cluster}/{ref}",
                        s_parallel="parallel1",
                        )
    # --------------------------------------------------------------------------
    # minimize_Lmol_with_smina_on_cluster.py file
    # --------------------------------------------------------------------------
    Ls = s1_py.split("\n")
    create_txt_from_L(Ls[1:], "do1.py")
    # ========================================================================
    # Export of the directory on the cluster and launch of the job
    # -------------------------------------------------------------------------
    os.chdir("..")
    logger.info(f"Transfert  of {ref} on cluster ...")
    bash(
                    f"scp -C -r {ref} {user_cluster}@{IP_cluster}:{path_cluster}", 
                    info=True,
        )
    # --------------------------------------------------------------------------
    # Launch of the jobs
    # -------------------------------------------------------------------------
    logger.info(f"Launch of jobs ...")
    Lcmd = [f"cd {path_cluster}/{ref}",
            "bash job_all.sh"]
    bash_on_remote(
                    Lcmd, 
                    user_cluster, 
                    IP_cluster,
                )
    logger.info(f"on going ...")
    logger.debug(f"wait for results in {path_local_ref}")
    start_time = time.time()
    # --------------------------------------------------------------------------
    logger.info(f"done in {round(time.time() - start_time)} s")
    # --------------------------------------------------------------------------
    # Loading of results
    # -------------------------------------------------------------------------
    logger.info(f"loading of the results")
    bash(
                    f"scp -C -r {user_cluster}@{IP_cluster}:{path_cluster}/{ref}/{dir_Lres} {path_local}", 
                    info=True,
                )
    # ------------------------------------------------------------------------
    os.chdir(path_local)
    Lcmd = [f"rm -rf {path_cluster}/{ref}"]
    bash_on_remote(Lcmd, user_cluster, IP_cluster)
    if return_Lmol:
        os.chdir(dir_Lres)
        return load_Lmol("Lmol*")





def get_n_job_on_cluster_core(
        ssh_client=None,
        user=None,
        name=None
        ):
    name2 = ""
    if name is not None:
        name2 = f"-n {name}"
    cmd = f"squeue -u {user} {name2} -h -r | wc -l"
    stdin, stdout, stderr = ssh_client.exec_command(cmd)
    Ls = stdout.read().splitlines()
    try:
        n = int(Ls[0])
    except:
        n=-1
    return n



def get_n_job_on_cluster(
        ssh_client=None,
        user=None,
        name=None
        ):
    """
    Gets the number of jobs still running or pending on the cluster
    """
    if type(name) is list:
        n = 0
        Lname = name
        for name in Lname:
            n += get_n_job_on_cluster_core(
                                            ssh_client=ssh_client,
                                            user=user,
                                            name=name
                                            )
    else:
        n = get_n_job_on_cluster_core(
                                        ssh_client=ssh_client,
                                        user=user,
                                        name=name
                                        )
    return n
            
            
        

def test_cluster_with_array_hello(
        name="test2",
        path_local=None,
        path_cluster=None,
        queue="cpu",
        user_local="krezel@ICOA-03",
        user="krezel",
        IP_local="192.168.11.160",
        IP_cluster="192.168.11.17",
        ref="cluster_test",
        conda_env="SBC",   # Name of the environment.
        ):
    """
    Simple test to verify that the cluster works.
    A simple test.py is launched on the cluster that prints its path and prints "OK".

    Info on ARRAY: https://crc.ku.edu/hpc/how-to/arrays
    https://rcpedia.stanford.edu/topicGuides/jobArrayPythonExample.html
    """
    logger.info("cluster_test1")
    if path_cluster is None:
        path_cluster = f"/home/{user}"
    path_local = get_path()
    # --------------------------------------------------------------------------
    logger.info(f"\t Remove of directory {ref} on cluster ...")
    cmd = f"ssh {user}@{IP_cluster} 'rm -r {path_cluster}/{ref}'"
    logger.info(f"\t\t {cmd}")
    os.system(cmd)
    # --------------------------------------------------------------------------
    # Creation of a new directory to gather all the informations
    # usefull to launch the script on the cluster.
    # --------------------------------------------------------------------------
    remove_dir(ref)
    os.makedirs(ref)
    os.chdir(ref)
    path_local_ref = get_path()
    # --------------------------------------------------------------------------
    # Creation of slurm file
    # --------------------------------------------------------------------------
    Ls = f"""
#!/bin/bash

#SBATCH -p {queue}
#SBATCH -J '{ref}'
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --array=1-10
#SBATCH -o hello-%j-%a.out
#SBATCH --error=test_%A_%a.err
source ~/.bashrc
conda activate SBC
cd /home/{user}/{ref}
srun python test.py $SLURM_ARRAY_TASK_ID
""".split("\n")
    create_txt_from_L(Ls[1:], "job.sh")
    # --------------------------------------------------------------------------
    Ls = f"""
import sys
n = sys.argv[1]
print(f"Hello! I am a task number {{n}}")
""".split("\n")
    create_txt_from_L(Ls[1:], "test.py")
    # --------------------------------------------------------------------------
    os.chdir(path_local)
    # --------------------------------------------------------------------------
    # Export of the directory on the cluster and launch of the job
    # -------------------------------------------------------------------------
    logger.info(f"\t Transfert  of {ref} on cluster ...")
    cmd = f"scp -r {ref} {user}@{IP_cluster}:{path_cluster}"
    logger.info(f"\t\t {cmd}")
    os.system(cmd)

    logger.info("\t Launch on the cluster")
    cmd = f"ssh {user}@{IP_cluster} 'cd {path_cluster}/{ref} ; sbatch job.sh'"
    logger.info(f"\t\t {cmd}")
    os.system(cmd)




def test_cluster_with_array_pickle(
        name="test2",
        path_local=None,
        path_cluster=None,
        queue="cpu",
        user_local="krezel@ICOA-03",
        user="krezel",
        IP_local="192.168.11.160",
        IP_cluster="192.168.11.17",
        ref="cluster_test",
        conda_env="SBC",   # Name of the environment.
        ):
    """
    Simple test to verify that the cluster works.
    A simple test.py is launched on the cluster that prints its path and prints "OK".

    Info on ARRAY: https://crc.ku.edu/hpc/how-to/arrays
    """
    logger.info("cluster_test_with_array")
    if path_cluster is None:
        path_cluster = f"/home/{user}"
    path_local = get_path()
    logger.info(f"\t path_local={path_local}")
    # --------------------------------------------------------------------------
    logger.info(f"\t Remove of directory {ref} on cluster ...")
    cmd = f"ssh {user}@{IP_cluster} 'rm -r {path_cluster}/{ref}'"
    logger.info(f"\t\t {cmd}")
    os.system(cmd)
    # --------------------------------------------------------------------------
    # Creation of a new directory to gather all the informations
    # usefull to launch the script on the cluster.
    # --------------------------------------------------------------------------
    remove_dir(ref)
    os.makedirs(ref)
    os.chdir(ref)
    path_local_ref = get_path()
    # --------------------------------------------------------------------------
    L = get_Lrand(nL=1000)
    LL = split_L_in_LL(
                L,
                nL=21
                )
    Lfp = []
    for i,L in enumerate(LL):
        fp = f"L_{i}"
        dump_var(L, fp)
        Lfp.append(fp)
    create_txt_from_L(Lfp, "Lfp.txt")
    nLfp = len(Lfp)
    # --------------------------------------------------------------------------
    # Creation of slurm file
    # --------------------------------------------------------------------------
    Ls = f"""
#!/bin/bash

#SBATCH -p {queue}
#SBATCH -J '{ref}'
#SBATCH --no-requeue
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --array=1-{nLfp}%100
#SBATCH --output array-%a.out
#SBATCH --error array-%a.err
source ~/.bashrc
conda activate SBC
cd /home/{user}/{ref}
LINE=$(sed -n "$SLURM_ARRAY_TASK_ID"p Lfp.txt)
srun python test.py $LINE
scp  L2*  {user_local}@{IP_local}:{path_local_ref}
""".split("\n")
    create_txt_from_L(Ls[1:], "job.sh")
    # --------------------------------------------------------------------------job_sbatch
    Ls = f"""
import os, sys
import numpy as np
import time
sys.path.insert(0, "/home/krezel/modules")
# =============================================================================
from tools_SBC.basic import load_var, dump_var, get_path
print("OK")
fp = str(sys.argv[1])
print(fp)
L = load_var(fp)
print(L)
L2 = [x*10 for x in L]
print(L2)
dump_var(L2, fp.replace("L", "L2"))
""".split("\n")
    create_txt_from_L(Ls[1:], "test.py")
    # --------------------------------------------------------------------------
    os.chdir(path_local)
    # --------------------------------------------------------------------------
    # Export of the directory on the cluster and launch of the job
    # -------------------------------------------------------------------------
    logger.info(f"\t Transfert  of {ref} on cluster ...")
    cmd = f"scp -r {ref} {user}@{IP_cluster}:{path_cluster}"
    logger.info(f"\t\t {cmd}")
    os.system(cmd)

    logger.info("\t Launch on the cluster")
    cmd = f"ssh {user}@{IP_cluster} 'cd {path_cluster}/{ref} ; sbatch job.sh'"
    logger.info(f"\t\t {cmd}")
    os.system(cmd)


def test_cluster_with_array_txt(
        name="test2",
        path_local=None,
        path_cluster=None,
        queue="cpu",
        user_local="krezel@ICOA-03",
        user="krezel",
        IP_local="192.168.11.160",
        IP_cluster="192.168.11.17",
        ref="cluster_test",
        conda_env="SBC",   # Name of the environment.
        ):
    """
    Simple test to verify that the cluster works.
    A simple test.py is launched on the cluster that prints its path and prints "OK".

    Info on ARRAY: https://crc.ku.edu/hpc/how-to/arrays
    """
    logger.info("cluster_test_with_array")
    if path_cluster is None:
        path_cluster = f"/home/{user}"
    path_local = get_path()
    logger.info(f"\t path_local={path_local}")
    # --------------------------------------------------------------------------
    logger.info(f"\t Remove of directory {ref} on cluster ...")
    cmd = f"ssh {user}@{IP_cluster} 'rm -r {path_cluster}/{ref}'"
    logger.info(f"\t\t {cmd}")
    os.system(cmd)
    # --------------------------------------------------------------------------
    # Creation of a new directory to gather all the informations
    # usefull to launch the script on the cluster.
    # --------------------------------------------------------------------------
    remove_dir(ref)
    os.makedirs(ref)
    os.chdir(ref)
    path_local_ref = get_path()
    # --------------------------------------------------------------------------
    Lfile = []
    for i in range(10):
        file = f"H1_{i}.txt"
        create_txt_from_s("Hello", file )
        Lfile.append(file)
    create_txt_from_L(Lfile, "Lfile.txt")
    nLfile = len(Lfile)
    # --------------------------------------------------------------------------
    # Creation of slurm file
    # --------------------------------------------------------------------------
    Ls = f"""
#!/bin/bash

#SBATCH -p {queue}
#SBATCH -J '{ref}'
#SBATCH --no-requeue
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --array=1-{nLfile}%100
#SBATCH --output array-%a.out
#SBATCH --error array-%a.err
source ~/.bashrc
conda activate SBC
cd /home/{user}/{ref}
LINE=$(sed -n "$SLURM_ARRAY_TASK_ID"p Lfile.txt)
srun python test.py $LINE
scp  H2*  {user_local}@{IP_local}:{path_local_ref}
""".split("\n")
    create_txt_from_L(Ls[1:], "job.sh")
    # --------------------------------------------------------------------------
    Ls = f"""
import os, sys
import numpy as np
import time
sys.path.insert(0, "/home/krezel/modules")
# =============================================================================
from tools_SBC.basic import (load_var, dump_var, get_path, create_txt_from_L,
    load_Ls_from_txt)
file = str(sys.argv[1])
print(file)
Ls = load_Ls_from_txt(file)
print(Ls)
Ls.append("OK")
create_txt_from_L(Ls, file.replace("H1", "H2"))
""".split("\n")
    create_txt_from_L(Ls[1:], "test.py")
    # --------------------------------------------------------------------------
    os.chdir(path_local)
    # --------------------------------------------------------------------------
    # Export of the directory on the cluster and launch of the job
    # -------------------------------------------------------------------------
    logger.info(f"\t Transfert  of {ref} on cluster ...")
    cmd = f"scp -r {ref} {user}@{IP_cluster}:{path_cluster}"
    logger.info(f"\t\t {cmd}")
    os.system(cmd)

    logger.info("\t Launch on the cluster")
    cmd = f"ssh {user}@{IP_cluster} 'cd {path_cluster}/{ref} ; sbatch job.sh'"
    logger.info(f"\t\t {cmd}")
    os.system(cmd)





def test_cluster_GPU_with_cupy(
        name="test_GPU",
        path_local=None,
        path_cluster=None,
        queue="rtx2080",
        user_local="krezel@ICOA-03",
        user="krezel",
        IP_local="192.168.11.160",
        IP_cluster="192.168.11.18",
        ref="test_on_cluster_GPU",
        conda_env="SBC",   # Name of the environment.
        n_core=4,
        ):
    """
    Test if cupy works on the cluster of GPU
    """
    logger.info("cluster_test_with_array")
    if path_cluster is None:
        path_cluster = f"/home/{user}"
    path_local = get_path()
    logger.info(f"\t path_local={path_local}")
    # --------------------------------------------------------------------------
    logger.info(f"\t Remove of directory {ref} on cluster ...")
    cmd = f"ssh {user}@{IP_cluster} 'rm -r {path_cluster}/{ref}'"
    logger.info(f"\t\t {cmd}")
    os.system(cmd)
    # --------------------------------------------------------------------------
    # Creation of a new directory to gather all the informations
    # usefull to launch the script on the cluster.
    # --------------------------------------------------------------------------
    remove_dir(ref)
    os.makedirs(ref)
    os.chdir(ref)
    path_local_ref = get_path()
    # --------------------------------------------------------------------------
    create_txt_from_L(list(range(100)), "Ls.txt")
    nLs = 100
    # --------------------------------------------------------------------------
    # Creation of slurm file
    # --------------------------------------------------------------------------
    Ls = f"""
#!/bin/bash

#SBATCH -C {queue}
#SBATCH -J '{ref}'
#SBATCH --no-requeue
#SBATCH --gres=gpu:GeForce:1
#SBATCH --time=00:10:00
#SBATCH --array=1-{nLs}%{n_core}
source ~/.bashrc
conda activate SBC
cd /home/{user}/{ref}
LINE=$(sed -n "$SLURM_ARRAY_TASK_ID"p Ls.txt)
srun python cluster_test_cupy_on_GPU.py $LINE
""".split("\n")
    create_txt_from_L(Ls[1:], "job.sh")
    # --------------------------------------------------------------------------
    Ls = f"""
import os, sys
import cupy as cp
# -----------------------------
M = cp.random.rand(3000,3000)
cp.linalg.svd(M)
cp.cuda.Stream.null.synchronize()
print("done")
""".split("\n")
    create_txt_from_L(Ls[1:], "cluster_test_cupy_on_GPU.py")
    # --------------------------------------------------------------------------
    os.chdir(path_local)
    # --------------------------------------------------------------------------
    # Export of the directory on the cluster and launch of the job
    # -------------------------------------------------------------------------
    logger.info(f"\t Transfert  of {ref} on cluster ...")
    cmd = f"scp -r {ref} {user}@{IP_cluster}:{path_cluster}"
    logger.info(f"\t\t {cmd}")
    os.system(cmd)

    logger.info("\t Launch on the cluster")
    cmd = f"ssh {user}@{IP_cluster} 'cd {path_cluster}/{ref} ; sbatch job.sh'"
    logger.info(f"\t\t {cmd}")
    os.system(cmd)
    
    

def test_cluster_GPU_with_numba(
    name="test_GPU",
    path_local=None,
    path_cluster=None,
    queue="rtx2080",
    user_local="krezel@ICOA-03",
    user="krezel",
    IP_local="192.168.11.160",
    IP_cluster="192.168.11.18",
    ref="test_on_cluster_GPU",
    conda_env="SBC",   # Name of the environment.
    n_core=4,
    ):
    """
    Test if cupy works on the cluster of GPU
    """
    logger.info("cluster_test_with_array")
    if path_cluster is None:
        path_cluster = f"/home/{user}"
    path_local = get_path()
    logger.info(f"\t path_local={path_local}")
    # --------------------------------------------------------------------------
    logger.info(f"\t Remove of directory {ref} on cluster ...")
    cmd = f"ssh {user}@{IP_cluster} 'rm -r {path_cluster}/{ref}'"
    logger.info(f"\t\t {cmd}")
    os.system(cmd)
    # --------------------------------------------------------------------------
    # Creation of a new directory to gather all the informations
    # usefull to launch the script on the cluster.
    # --------------------------------------------------------------------------
    remove_dir(ref)
    os.makedirs(ref)
    os.chdir(ref)
    path_local_ref = get_path()
    # --------------------------------------------------------------------------
    create_txt_from_L(list(range(20)), "Ls.txt")
    nLs = 20
    # --------------------------------------------------------------------------
    # Creation of slurm file
    # --------------------------------------------------------------------------
    Ls = f"""
#!/bin/bash

#SBATCH -C {queue}
#SBATCH -J '{ref}'
#SBATCH --no-requeue
#SBATCH --gres=gpu:GeForce:1
#SBATCH --time=00:10:00
#SBATCH --array=1-{nLs}%{n_core}
source ~/.bashrc
conda activate SBC
cd /home/{user}/{ref}
LINE=$(sed -n "$SLURM_ARRAY_TASK_ID"p Ls.txt)
srun python cluster_test_numba_on_GPU.py $LINE
""".split("\n")
    create_txt_from_L(Ls[1:], "job.sh")
    # --------------------------------------------------------------------------
# import numpy as np
# import numba
# from numba import cuda
# from time import perf_counter_ns
# # -----------------------------
# print(np.__version__)
# print(numba.__version__)
# cuda.detect()

# @cuda.jit
# def add_array(a,b,c):
#     i = cuda.grid(1)
#     if i < a.size:
#         c[i] = a[i] + b[i]
        
# N = 1_000_000
# a = np.arange(N, dtype=np.)
# b = np.arange(N, dtype=np.float32)
# dev_a = cuda.to_device(a)
# dev_b = cuda.to_device(b)
# dev_c = cuda.device_array_like(dev_a)

# threads_per_block = 256
# blocks_per_grid = (N + (threads_per_block - 1)) // threads_per_block

# add_array[blocks_per_grid, threads_per_block](dev_a, dev_b, dev_c)

# c = dev_c.copy_to_host()
# print(np.allclose(a + b, c))

# # ----------------------------------------------------
# cuda.synchronize()

# timing = np.empty(101)
# for i in range(timing.size):
#     tic = perf_counter_ns()
#     add_array[blocks_per_grid, threads_per_block](dev_a, dev_b, dev_c)
#     cuda.synchronize()
#     toc = perf_counter_ns()
#     timing[i] = toc - tic
# timing *= 1e-3  # convert to μs

# print(f"Elapsed time: {{timing.mean():.0f}} ± {{timing.std():.0f}} μs")

    Ls = f"""
# ----------------------------------------------------
import math # Note that for the CUDA target, we need to use the scalar functions from the math module, not NumPy
import numpy as np
from numba import vectorize
SQRT_2PI = np.float32((2*math.pi)**0.5)  # Precompute this constant as a float32.  Numba will inline it at compile time.

@vectorize(['float32(float32, float32, float32)'], target='cuda')
def gaussian_pdf(x, mean, sigma):
    '''Compute the value of a Gaussian probability density function at x with given mean and sigma.'''
    return math.exp(-0.5 * ((x - mean) / sigma)**2) / (sigma * SQRT_2PI)
    
# Evaluate the Gaussian a million times!
x = np.random.uniform(-3, 3, size=1000000).astype(np.float32)
mean = np.float32(0.0)
sigma = np.float32(1.0)

# Quick test on a single element just to make sure it works
gaussian_pdf(x[0], mean, sigma)
print("done")
""".split("\n")
    create_txt_from_L(Ls[1:], "cluster_test_numba_on_GPU.py")
    # --------------------------------------------------------------------------
    os.chdir(path_local)
    # --------------------------------------------------------------------------
    # Export of the directory on the cluster and launch of the job
    # -------------------------------------------------------------------------
    logger.info(f"\t Transfert  of {ref} on cluster ...")
    cmd = f"scp -r {ref} {user}@{IP_cluster}:{path_cluster}"
    logger.info(f"\t\t {cmd}")
    os.system(cmd)

    logger.info("\t Launch on the cluster")
    cmd = f"ssh {user}@{IP_cluster} 'cd {path_cluster}/{ref} ; sbatch job.sh'"
    logger.info(f"\t\t {cmd}")
    os.system(cmd)