#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os


job_tpl_hdr = '''#!/bin/bash

export BOOLDIR="/storage/brno3-cerit/home/${LOGNAME}/booltest/assets"
export RESDIR="/storage/brno3-cerit/home/${LOGNAME}/bool-res"
export LOGDIR="/storage/brno3-cerit/home/${LOGNAME}/bool-log"

cd "${BOOLDIR}"

'''

job_tpl = '''
./generator-metacentrum.sh -c=%s | ./booltest-json-metacentrum.sh \\
    %s > "${LOGDIR}/%s.out" 2> "${LOGDIR}/%s.err"
'''


class TestBatchUnit(object):
    """
    For creating job batches
    """
    def __init__(self, **kwargs):
        self.res_file_path = None
        self.gen_file_path = None
        self.cfg_file_path = None
        self.res_file = None
        self.block_size = None
        self.degree = None
        self.comb_deg = None
        self.data_size = None
        self.size_mb = None


class BatchGenerator(object):
    """
    Generating batch jobs
    """
    def __init__(self):
        self.generator_files = set()
        self.job_dir = None
        self.job_files = []
        self.job_batch = []
        self.batch_max_bl = 0
        self.batch_max_deg = 0
        self.batch_max_comb_deg = 0
        self.job_batch_max_size = 50
        self.cur_batch_def = None  # type: TestBatchUnit
        self.memory_threshold = 50
        self.num_skipped = 0
        self.num_skipped_existing = 0
        self.job_file_path = None

    def add_unit(self, unit):
        """
        Adds unit of work to the batch
        :param unit:
        :return:
        """
        args = ' --config-file %s' % unit.cfg_file_path
        job_data = job_tpl % (unit.gen_file_path, args, unit.res_file, unit.res_file)
        self.job_batch.append(job_data)
        self.generator_files.add(unit.gen_file_path)

        self.batch_max_bl = max(self.batch_max_bl, unit.block_size)
        self.batch_max_deg = max(self.batch_max_deg, unit.degree)
        self.batch_max_comb_deg = max(self.batch_max_comb_deg, unit.comb_deg)
        self.job_file_path = os.path.join(self.job_dir, 'job-' + unit.res_file + '.sh')

        flush_batch = False
        if self.cur_batch_def is None:
            self.cur_batch_def = unit
            self.job_batch_max_size = 15

            if unit.size_mb < 11:
                self.job_batch_max_size = 25
                if self.batch_max_deg <= 2 and self.batch_max_comb_deg <= 2:
                    self.job_batch_max_size = 50
                if self.batch_max_deg <= 1 and self.batch_max_comb_deg <= 2:
                    self.job_batch_max_size = 100

            if unit.size_mb < 2:
                self.job_batch_max_size = 100
                if self.batch_max_deg <= 2 and self.batch_max_comb_deg <= 2:
                    self.job_batch_max_size = 200
                if self.batch_max_deg <= 1 and self.batch_max_comb_deg <= 2:
                    self.job_batch_max_size = 300

        elif self.cur_batch_def.data_size != unit.data_size \
                or len(self.job_batch) >= self.job_batch_max_size:
            flush_batch = True

        if flush_batch:
            self.flush()

    def flush(self):
        """
        Flushes batch
        :return:
        """
        if len(self.job_batch) == 0:
            return

        job_data = job_tpl_hdr + '\n'.join(self.job_batch)
        unit = self.cur_batch_def

        with open(self.job_file_path, 'w+') as fh:
            fh.write(job_data)

        ram = '12gb' if unit.size_mb > self.memory_threshold else '6gb'
        job_time = '24:00:00'
        if unit.size_mb < 11:
            job_time = '4:00:00'
        self.job_files.append((self.job_file_path, ram, job_time))

        self.cur_batch_def = None
        self.job_batch = []
        self.batch_max_bl = 0
        self.batch_max_deg = 0
        self.batch_max_comb_deg = 0



