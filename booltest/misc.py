#!/usr/bin/env python
# -*- coding: utf-8 -*-

import collections
import math
import re
import random
import os
import logging
import shutil
import errno


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



