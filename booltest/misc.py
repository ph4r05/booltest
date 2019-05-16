#!/usr/bin/env python
# -*- coding: utf-8 -*-

import collections
import subprocess
import json
import math
import re
import random
import os
import logging
import shutil
import errno
import cpuinfo
import socket
import psutil
import glob


logger = logging.getLogger(__name__)


def try_chmod_grx(path):
    """
    Attempts to add +x flag
    :param path:
    :return:
    """
    try:
        os.chmod(path, 0o750)
    except Exception as e:
        logger.warning('Exception chmoddin %s: %s' % (path, e))


def try_chmod_gr(path):
    """
    Update permissions
    :param path:
    :return:
    """
    try:
        os.chmod(path, 0o640)
    except Exception as e:
        logger.warning('Error chmodding %s : %s' % (path, e))


def file_backup(path, chmod=0o644, backup_dir=None, backup_suffix=None):
    """
    Backup the given file by copying it to a new file
    Copy is preferred to move. Move can keep processes working with the opened file after move operation.

    :param path:
    :param chmod:
    :param backup_dir:
    :param backup_suffix: if defined, suffix is appended to the backup file (e.g., .backup)
    :return:
    """
    backup_path = None
    if os.path.exists(path):
        backup_path = path
        if backup_dir is not None:
            opath, otail = os.path.split(path)
            backup_path = os.path.join(backup_dir, otail)
        if backup_suffix is not None:
            backup_path += backup_suffix

        if chmod is None:
            chmod = os.stat(path).st_mode & 0o777

        with open(path, 'r') as src:
            fhnd, fname = unique_file(backup_path, chmod)
            with fhnd:
                shutil.copyfileobj(src, fhnd)
                backup_path = fname
    return backup_path


def _unique_file(path, filename_pat, count, mode):
    while True:
        current_path = os.path.join(path, filename_pat(count))
        try:
            return safe_open(current_path, chmod=mode),\
                os.path.abspath(current_path)
        except OSError as err:
            # "File exists," is okay, try a different name.
            if err.errno != errno.EEXIST:
                raise
        count += 1


def unique_file(path, mode=0o777):
    """Safely finds a unique file.

    :param str path: path/filename.ext
    :param int mode: File mode

    :returns: tuple of file object and file name

    """
    path, tail = os.path.split(path)
    filename, extension = os.path.splitext(tail)
    return _unique_file(
        path, filename_pat=(lambda count: "%s_%04d%s" % (filename, count, extension if not None else '')),
        count=0, mode=mode)


def safe_open(path, mode="w", chmod=None, buffering=None, exclusive=True):
    """Safely open a file.

    :param str path: Path to a file.
    :param str mode: Same os `mode` for `open`.
    :param int chmod: Same as `mode` for `os.open`, uses Python defaults
        if ``None``.
    :param int buffering: Same as `bufsize` for `os.fdopen`, uses Python
        defaults if ``None``.
    :param bool exclusive: if True, the file cannot exist before
    """
    # pylint: disable=star-args
    open_args = () if chmod is None else (chmod,)
    fdopen_args = () if buffering is None else (buffering,)
    flags = os.O_CREAT | os.O_EXCL | os.O_RDWR
    if exclusive:
        flags |= os.O_EXCL

    return os.fdopen(os.open(path, flags, *open_args),mode, *fdopen_args)


def slot_obj_dict(o):
    """
    Builds dict for o with __slots__ defined
    :param o:
    :return:
    """
    d = {}
    for f in o.__slots__:
        d[f] = getattr(o, f, None)
    return d


def eq_obj_slots(l, r):
    """
    Compares objects with __slots__ defined
    :param l:
    :param r:
    :return:
    """
    for f in l.__slots__:
        if getattr(l, f, None) != getattr(r, f, None):
            return False
    return True


def eq_obj_contents(l, r):
    """
    Compares object contents, supports slots
    :param l:
    :param r:
    :return:
    """
    if l.__class__ is not r.__class__:
        return False
    if hasattr(l, '__slots__'):
        return eq_obj_slots(l, r)
    else:
        return l.__dict__ == r.__dict__


def try_get_cpu_info():
    """
    Returns CPU info
    https://github.com/workhorsy/py-cpuinfo
    :return:
    """
    try:
        return cpuinfo.get_cpu_info()

    except Exception as e:
        logger.error('Cpuinfo exception %s' % e)
        return None


def try_get_hostname():
    """
    Returns hostname or none
    :return:
    """
    try:
        return socket.getfqdn()

    except Exception as e:
        logger.error('Hostname exception %s' % e)
        return None


def try_get_cpu_percent():
    """
    CPU usage before run
    :return:
    """
    try:
        return psutil.cpu_percent()
    except Exception as e:
        logger.error('Cpu percent exception %s' % e)
        return None


def try_get_cpu_load():
    """
    CPU unix-like load
    :return:
    """
    try:
        return os.getloadavg()
    except Exception as e:
        logger.error('Cpu load exception %s' % e)
        return None


def file_exists(path):
    try:
        return os.stat(path)
    except OSError:
        return None


def normalize_card_name(name):
    name = name.replace(' - ', '_')
    name = name.replace('+', '')
    name = name.replace('-', '_')
    name = name.replace(' ', '_')
    name = name.lower()
    name = re.sub(r'[^a-zA-Z0-9_]', '', name)
    return name


def unpack_keys(card_dir, out_dir, bitsize):
    files = glob.glob(os.path.join(card_dir, '*.rar'))
    filt = '1024b' if bitsize == 1024 else '512b'
    files = [x for x in files if filt in x]
    logger.debug('Files: %s' % files)

    import tempfile
    from pyunpack import Archive

    for fl in files:
        dir_name = tempfile.mkdtemp(prefix='tmp_card', dir=out_dir)
        bname = os.path.basename(fl)
        bname = os.path.splitext(bname)[0]
        bname = normalize_card_name(bname)
        dest_file = os.path.join(out_dir, '%s.csv' % bname)
        fh = open(dest_file, 'w+')
        logger.debug('Processing %s to %s' % (bname, dest_file))

        try:
            Archive(fl).extractall(dir_name)
            csvs = glob.glob(os.path.join(dir_name, '*.csv'))
            for cl in csvs:
                logger.debug(' .. %s' % cl)

                with open(cl) as cfh:
                    data = cfh.read()
                fh.write(data)
                fh.flush()

        finally:
            shutil.rmtree(dir_name)
            fh.close()


def get_jobs_in_progress():
    uname = os.getenv('LOGNAME', None)
    uname_prefix = uname + '@' if uname else None

    p = subprocess.Popen("qstat -f -F json", stdout=subprocess.PIPE, shell=True)
    output, err = p.communicate()
    p_status = p.wait()
    if p_status != 0:
        raise ValueError('Could not get running jobs')

    if len(output) == 0:
        return {}, {}

    lines = output.split(b'\n')
    output = []
    for l in lines:
        if b'"comment":' not in l:
            output.append(l)
        elif l.count(b'"') > 4:
            output.append(b'"comment":"INVALID",')
        else:
            output.append(l)

    output = b"\n".join(output)
    js = json.loads(output)
    jobs = js['Jobs']
    res = {}

    for jid in jobs:
        jb = jobs[jid]
        if uname_prefix is not None and not jb['Job_Owner'].startswith(uname_prefix):
            continue
        obj = collections.OrderedDict()
        obj['jid'] = jid
        obj['id'] = jb['Job_Name']
        obj['running'] = jb['job_state'] == 'R'
        obj['queued'] = jb['job_state'] == 'Q'
        obj['info'] = jb
        res[jb['Job_Name']] = obj
    scheduled = set(k for k, v in res.items() if v['running'] or v['queued'])
    return res, scheduled
