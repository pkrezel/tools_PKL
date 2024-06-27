#! /usr/bin/env python
# -*- coding: UTF-8 -*-
import os, sys
import logging
logger = logging.getLogger(__name__)
#=======================================================================
import os, sys
import re
# print(__name__)
# logger.debug("\t test logger.debug")
# logger.info("\t test logger.info")
# logger.warning("\t test logger.warning")
#=======================================================================
import time
from shutil import copy, move
#=======================================================================
from tools_PKL.basic import (bash, bashr, get_path, load_Ls_from_txt, remove_file,
    create_txt_from_L, load_var, dump_var, remove_dir, make_dir, get_path_py, remove_dir)
from tools_PKL.scp import SCPClient
#=======================================================================



def make_dir_on_remote(
        dir,
        path_remote="/",
        user_remote="krezel",
        IP_remote="192.168.11.17",
        ):
    Lcmd = [
                f"cd {path_remote}",
                f"if [ ! -d '{dir}' ]; then mkdir {dir}; fi",
    ]
    bash_on_remote(
        Lcmd,
        user_remote,
        IP_remote,
        info=True,
    )


def bash_on_remote(
        cmd,
        user_remote,
        IP_remote,
        info=True,
        niv_log=0,
        in_background=False
    ):
    if type(cmd) is list:
        cmd = ";".join(cmd)
    sback = ""
    if in_background:
        sback=" &"
    bash(
            f"ssh {user_remote}@{IP_remote} '{cmd}'{sback}", 
            info=info,
            niv_log=niv_log,
        )


def do_ssh(
            ssh_client,
            cmd, # command as string
            with_stdin=False,
            with_stout=True,
            with_stderr=False,
            
        ):
    logger.info(cmd)
    stdin, stdout, stderr = client.exec_command(cms)
    if with_stdin:
        L = stdin.read().splitlines()
        logger.info(L)
    if with_stdout:
        L = stdout.read().splitlines()
        logger.info(L)
    if with_stderr:
        L = stderr.read().splitlines()
        logger.info(L)

def launch_py_file_on_remote(
        py_file, # py file to be launched
        path,
        user_remote="krezel",
        IP_remote="192.168.11.17",
        env="SBC",
        Lvar=None,
        Dvar=None,
        path_py="."
        ):
    """
    Launch a py file in the current directory on a computer on the network.
    """
    logger.info("launch_py_file_on_remote")
    if path[-1]=="/":
        path = path[:-1]
    if path_py[-1]=="/":
        path_py = path_py[:-1]
    if ".py" not in py_file:
        py_file += ".py"
    logger.debug("\t copy of py file")
    bash(f"scp  {path_py}/{py_file} {user_remote}@{IP_remote}:{path}", info=True)
    Ls=f"""
#!/bin/bash
source ~/.bashrc
conda activate {env}
cd {path}
python {py_file}
""".split("\n")[1:]
    create_txt_from_L(Ls, "code.sh")
    logger.debug("\t copy of code.sh")
    bash(f"scp  code.sh {user_remote}@{IP_remote}:{path}", info=True)
    if Lvar is not None:
        dump_var_on_remote(
                            Lvar,
                            user_remote=user_remote,
                            IP_remote=IP_remote,
                            path=path)
    if Dvar is not None:
        logger.debug("\t dump of Dvar")
        dump_var_on_remote(
                            Dvar,
                            user_remote=user_remote,
                            IP_remote=IP_remote,
                            path=path) 
    logger.debug("\t launch of code.sh")
    bash(f"ssh {user_remote}@{IP_remote} sh {path}/code.sh", info=True) 
    
    
def dump_file_on_remote(
        file,
        path,
        user_remote="krezel",
        IP_remote="192.168.11.17",
        ):
    bash(f"scp  {file} {user_remote}@{IP_remote}:{path}")
    
    
def load_file_from_remote(
        file, # py file to be launched
        user_remote="krezel",
        IP_remote="192.168.11.17",
        path_remote=""
        ):
    path0 = get_path()
    bash(f"scp  {user_remote}@{IP_remote}:{path_remote}/{file} {path0}", info=True)


def copy_dir_on_remote(
        dir,
        path=None,
        user_remote="krezel",
        IP_remote="192.168.11.17",
    ):
    if path is None:
        path = f"/home/{user_remote}"
    bash(f"scp  -C -r {dir} {user_remote}@{IP_remote}:{path}")


def copy_dir_from_remote(
        path_cluster,
        user_remote="krezel",
        IP_remote="192.168.11.17",
    ):
    path_local = get_path()
    bash(f"scp -C -r {user_remote}@{IP_remote}:{path_cluster} {path_local}")
    
    
def load_var_from_remote(
        var,
        user_remote="krezel",
        IP_remote="192.168.11.17",
        path_remote="",
        ):
    if ".p" not in var:
        var += ".p"
    load_file_from_remote(
            var, # py file to be launched
            user_remote=user_remote,
            IP_remote=IP_remote,
            path_remote=path_remote,
            )
    return load_var(var)


def dump_var_on_remote(
        var,
        fname=None,
        user_remote="krezel",
        IP_remote="192.168.11.17",
        path="/",
        ):
    fname = dump_var(var, fname)
    dump_file_on_remote(
                        fname, # py file to be launched
                        path,
                        user_remote,
                        IP_remote,)

    

def get_Lfile_on_remote(
        s="*",
        path_remote="/",
        user_remote="krezel",
        IP_remote="192.168.11.17",
        maxdepth=1,
        with_find=True,
        ):
    """
    This solution is very slow, I don't understand why
    """
    if with_find:
        cmd = f"find {path_remote} -maxdepth {maxdepth} -name '{s}' -not -type d >  /home/{user_remote}/Lfile.txt"
    else:
        cmd = f"cd {path_remote}; ls  >  /home/{user_remote}/Lfile.txt"
    bash_on_remote(
                    cmd,
                    user_remote=user_remote,
                    IP_remote=IP_remote,
                    info=True,
                )
    load_file_from_remote(
                            "Lfile.txt",
                            user_remote=user_remote,
                            IP_remote=IP_remote,
                            path_remote=f"/home/{user_remote}",
                        )
    Lfile = load_Ls_from_txt("Lfile.txt")
    remove_file("Lfile.txt")
    if with_find:
        return [file.replace(f"{path_remote}/","") for file in Lfile[1:]]
    else:
        return [file for file in Lfile if re.search(s,file)]

 


def get_Ldir_on_remote(
        path_remote="/",
        user_remote="krezel",
        IP_remote="192.168.11.17",
        ):
    """
    Get the list of directories on remote computer
    """
    path_local = get_path()
    bashr(f"ssh {user_remote}@{IP_remote} 'cd {path_remote}; find -type d > /home/{user_remote}/Ldir.txt'", info=True)
    bashr(f"scp {user_remote}@{IP_remote}:/home/{user_remote}/Ldir.txt {path_local}", info=True)
    Ldir = load_Ls_from_txt("Ldir.txt")
    Ldir = [dirx[2:] for dirx in Ldir][1:]
    Ldir = sorted(Ldir)
    remove_file("Ldir.txt")
    bashr(f"ssh {user_remote}@{IP_remote} 'rm /home/{user_remote}/Ldir.txt'", info=True)
    return Ldir


def get_Ldir_on_remote_last_modified(
        s="*",
        path_remote="/",
        user_remote="krezel",
        IP_remote="192.168.11.17",
        n_hours=8 # time since last modifcation
        ):
    bash(f"ssh {user_remote}@{IP_remote} 'cd {path_remote}; find . -type d -mmin -$((60*{n_hours})) > {path_remote}/Ldir.txt'")
    path_local = get_path()
    bash(f"scp {user_remote}@{IP_remote}:{path_remote}/Ldir.txt {path_local}")
    Ldir = load_Ls_from_txt("Ldir.txt")
    Ldir = [dirx[2:] for dirx in Ldir][1:]
    Ldir = sorted(Ldir)
    remove_file("Ldir.txt")
    return Ldir


def get_Ldir_without_fname_on_remote(
        fname="done",
        path_remote=None,
        user_local="krezel@ICOA-03",
        IP_local="192.168.11.160",
        user_remote="krezel",
        IP_remote="192.168.11.17",
        ref="run",
        conda_env="SBC",
        ):
    logger.info("get_Ldir_without_fname_on_remote")
    path_local = get_path()
    bash(f"ssh {user_remote}@{IP_remote} 'rm -Rf /home/{user_remote}/{ref}' ", info=False)
    make_dir(ref)
    os.chdir(ref)
    Ls = f"""
import sys
import os
sys.path.insert(0, "/home/{user_remote}/modules")
# =============================================================================
from tools_SBC.basic import (get_Ldir_without_fname, load_var, dump_var, bash, get_path, remove_file)

bash("conda activate {conda_env}")
os.chdir("{path_remote}")
Ldir = get_Ldir_without_fname("{fname}")
dump_var(Ldir)
bash(f"scp Ldir.p {user_local}@{IP_local}:{path_local}")
remove_file("Ldir.p")
""".split("\n")
    create_txt_from_L(Ls[1:], "ssh_get_Ldir_without_fname_on_cluster.py")
    os.chdir("..")
    bashr(f"scp -r {ref} {user_remote}@{IP_remote}:/home/{user_remote}", info=True)
    bashr(f"ssh {user_remote}@{IP_remote} 'cd /home/{user_remote}/{ref}; source ~/.bashrc; conda activate {conda_env}; python ssh_get_Ldir_without_fname_on_cluster.py'", info=True)
    Ldir = load_var("Ldir.p")
    remove_file("Ldir.p")
    remove_dir(ref)
    return Ldir  
        
    
def get_Ldir_with_fname_on_remote(
        fname="done",
        user_local="krezel@ICOA-03",
        IP_local="192.168.11.160",
        user_remote="krezel",
        IP_remote="192.168.11.17",
        path_remote=None,
        ref="run",
        conda_env="SBC",
        ):
    logger.info("get_Ldir_with_fname_on_remote")
    path_local = get_path()
    bash(f"ssh {user_remote}@{IP_remote} 'rm -Rf /home/{user_remote}/{ref}' ", info=True)
    make_dir(ref)
    os.chdir(ref)
    Ls = f"""
import sys
import os
sys.path.insert(0, "/home/{user_remote}/modules")
# =============================================================================
from tools_SBC.basic import (get_Ldir_with_fname, load_var, dump_var, bash, get_path, remove_file)
# =============================================================================
os.chdir("{path_remote}")
Ldir = get_Ldir_with_fname("{fname}")
dump_var(Ldir, "Ldir")
bash(f"scp {path_remote}/Ldir.p {user_local}@{IP_local}:{path_local}", info=True)
""".split("\n")
    create_txt_from_L(Ls[1:], "ssh_get_Ldir_with_fname_on_cluster.py")
    os.chdir("..")
    logger.info("Searching ...")
    bashr(f"scp -r {ref} {user_remote}@{IP_remote}:/home/{user_remote}", info=True)
    bashr(f"ssh {user_remote}@{IP_remote} 'cd /home/{user_remote}/{ref}; source ~/.bashrc; conda activate {conda_env}; python ssh_get_Ldir_with_fname_on_cluster.py'", info=True)
    os.chdir(path_local)
    Ldir = load_var("Ldir.p")
    remove_file("Ldir.p")
    bashr(f"ssh {user_remote}@{IP_remote} 'rm {path_remote}/Ldir.p' ", info=True)
    remove_dir(ref)
    return Ldir  


def get_Lkey_in_hdf5_on_remote(
        fname="done",
        path_remote=None,
        user_remote="krezel",
        IP_remote="192.168.11.17",
        ref="run",
        conda_env="SBC",
        ):
    logger.info("get_Lkey_in_hdf5_on_remote")
    path_local = get_path()
    bash(f"ssh {user_remote}@{IP_remote} 'rm -Rf /home/{user_remote}/{ref}' ", info=True)
    # -------------------
    make_dir(ref)
    os.chdir(ref)
    Ls = f"""
import sys
import os

sys.path.insert(0, "/home/krezel/modules")
# =============================================================================
from tools_SBC.basic import (get_Ldir_with_fname, load_var, dump_var, bash, get_path, get_Lkey_in_hdf5)
# =============================================================================
bash("conda activate {conda_env}")
path = get_path()
os.chdir("{path_remote}")
Lkey = get_Lkey_in_hdf5("{fname}")
dump_var(Lkey, "Lkey")
print("done")
""".split("\n")
    fname = "get_Lkey_in_hdf5_on_remote.py"
    create_txt_from_L(Ls[1:], fname)
    os.chdir("..")
    logger.info("Searching ...")
    bash(f"scp -r {ref} {user_remote}@{IP_remote}:/home/{user_remote}", info=False)
    bash(f"ssh {user_remote}@{IP_remote} 'cd /home/{user_remote}/{ref};conda activate {conda_env}; python {fname}'", info=False)
    time.sleep(5)
    bash(f"scp {user_remote}@{IP_remote}:/home/{user_remote}/{ref}/Lkey.p {path_local}", info=False)
    Lkey = load_var("Lkey")
    remove_file("Lkey.p")
    os.chdir("..")
    remove_dir(ref)
    return Lkey  


def download_Lfile_from_remote(
    fname="done",
    user_local="krezel@ICOA-03",
    IP_local="192.168.11.160",
    path_local=".",
    user_remote="krezel",
    IP_remote="192.168.11.17",
    path_remote=None,
    ):
    """
    download a list of file contained in a directory.
    """
    bashr(f"rsync -av -e ssh --remove-source-files {user_remote}@{IP_remote}:{path_remote}/{fname} {path_local}", info=True)
    #BUG : problem avec rsync qui pourtant avait fonctionné.
    
    
def download_dir_from_remote(
        user_local="krezel@ICOA-03",
        IP_local="192.168.11.160",
        path_local=".",
        user_remote="krezel",
        IP_remote="192.168.11.17",
        path_dir_on_remote=None,
        info=True,
        niv_log=0,
    ):
    """
    download a dir from remote by creating a tar archive
    """
    if path_dir_on_remote is None:
        logger.warning("\t"*niv_log + "path_dir_on_remote must be given")
        return
    # with ssh
    #bash(f"ssh {user_remote}@{IP_remote} 'tar -c -f - -C {path_remote} .' | tar -x -f - -C {path_local}")
    # --------------------------------------------------
    # with scp
    bash(f"scp -C -r {user_remote}@{IP_remote}:{path_dir_on_remote} {path_local}", info=info)

    