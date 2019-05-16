#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os


job_tpl_hdr = '''#!/bin/bash

export BOOLDIR="/storage/brno3-cerit/home/${LOGNAME}/booltest/assets"
export RESDIR="/storage/brno3-cerit/home/${LOGNAME}/bool-res"
export LOGDIR="/storage/brno3-cerit/home/${LOGNAME}/bool-log"
export SIGDIR="/storage/brno3-cerit/home/${LOGNAME}/bool-sig"

cd "${BOOLDIR}"

set -o pipefail
export RRES=0

'''

tpl_clean_signals = '''
IND_BASE=${SIGDIR}/%s
/bin/rm ${IND_BASE}.started 2>/dev/null
/bin/rm ${IND_BASE}.finished 2>/dev/null
/bin/rm ${IND_BASE}.failed 2>/dev/null
'''

job_tpl_prefix = '''
# -------------------------------------------------------------------
IND_BASE=${SIGDIR}/%s
touch ${IND_BASE}.started
/bin/rm ${IND_BASE}.finished 2>/dev/null
/bin/rm ${IND_BASE}.failed 2>/dev/null

'''

job_tpl = '''
./generator-metacentrum.sh -c=%s | ./booltest-json-metacentrum.sh \\
    %s > "${LOGDIR}/%s.out" 2> "${LOGDIR}/%s.err"
'''

job_tpl_data_file = '''
./booltest-json-metacentrum.sh \\
    %s > "${LOGDIR}/%s.out" 2> "${LOGDIR}/%s.err"
'''

tpl_handle_res_common = '''
RRES=$(($RBOOL == 0 ? $RRES : 10 + (($RRES - 10 + 1) % 100)))

echo $RBOOL > ${IND_BASE}.finished
if [ $RBOOL -ne 0 ]; then
    touch ${IND_BASE}.failed
fi
'''

tpl_handle_res = '''
RBOOL=$?
''' + tpl_handle_res_common

tpl_handle_res_retry = '''
C_ITER=0
RBOOL=2
TIME_START=$SECONDS

while (( $C_ITER < 6 && ($RBOOL == 2 || $RBOOL == 1) )); do
    C_ITER=$((C_ITER+1))
    echo "`hostname` iteration <<JOBNAME>> ${C_ITER}..."
    
    TIME_ELAPSED=$(($SECONDS - $TIME_START))
    if (( $TIME_ELAPSED > 600 )); then
        echo "Elapsed time too big: ${TIME_ELAPSED}, quitting"
        break
    fi
    
    <<JOB>>
    RBOOL=$?
done
''' + tpl_handle_res_common

job_tpl_footer = '''
exit $RRES
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
        self.job_clean_batch = []
        self.batch_max_bl = 0
        self.batch_max_deg = 0
        self.batch_max_comb_deg = 0
        self.job_batch_max_size = 50
        self.cur_batch_def = None  # type: TestBatchUnit
        self.memory_threshold = 50
        self.num_units = 0
        self.num_skipped = 0
        self.num_skipped_existing = 0
        self.job_file_path = None
        self.aggregation_factor = 1.0
        self.retry = True
        self.max_hour_job = 24

    def aggregate(self, jobs, fact, min_jobs=1):
        return max(min_jobs, int(jobs * fact))

    def add_unit(self, unit):
        """
        Adds unit of work to the batch
        :param unit:
        :return:
        """
        args = ' --config-file %s' % unit.cfg_file_path
        job_data = job_tpl_prefix % unit.res_file

        job_exec = ''
        if unit.gen_file_path:
            job_exec = job_tpl % (unit.gen_file_path, args, unit.res_file, unit.res_file)
        else:
            job_exec = job_tpl_data_file % (args, unit.res_file, unit.res_file)

        if self.retry:
            chunk = tpl_handle_res_retry.replace('<<JOB>>', job_exec)
            chunk = chunk.replace('<<JOBNAME>>', unit.res_file)
            job_data += chunk
        else:
            job_data += job_exec
            job_data += tpl_handle_res

        self.num_units += 1
        self.job_batch.append(job_data)
        self.job_clean_batch.append(tpl_clean_signals % unit.res_file)

        if unit.gen_file_path:
            self.generator_files.add(unit.gen_file_path)

        self.batch_max_bl = max(self.batch_max_bl, unit.block_size)
        self.batch_max_deg = max(self.batch_max_deg, unit.degree)
        self.batch_max_comb_deg = max(self.batch_max_comb_deg, unit.comb_deg)
        self.job_file_path = os.path.join(self.job_dir, 'job-' + unit.res_file + '.sh')

        flush_batch = False
        under4 = self.max_hour_job <= 4
        if self.cur_batch_def is None:
            self.cur_batch_def = unit

            # 1100 MB data and more
            self.job_batch_max_size = self.aggregate(2, self.aggregation_factor)
            if self.batch_max_deg <= 2 and self.batch_max_comb_deg <= 2:
                self.job_batch_max_size = self.aggregate(3 if under4 else 5, self.aggregation_factor)
            if self.batch_max_deg <= 1 and self.batch_max_comb_deg <= 2:
                self.job_batch_max_size = self.aggregate(4 if under4 else 10, self.aggregation_factor)

            if unit.size_mb < 1100:
                self.job_batch_max_size = self.aggregate(5, self.aggregation_factor)
                if self.batch_max_deg <= 2 and self.batch_max_comb_deg <= 2:
                    self.job_batch_max_size = self.aggregate(10, self.aggregation_factor)
                if self.batch_max_deg <= 1 and self.batch_max_comb_deg <= 2:
                    self.job_batch_max_size = self.aggregate(15, self.aggregation_factor)

            if unit.size_mb < 11:
                self.job_batch_max_size = self.aggregate(25, self.aggregation_factor)
                if self.batch_max_deg <= 2 and self.batch_max_comb_deg <= 2:
                    self.job_batch_max_size = self.aggregate(50, self.aggregation_factor)
                if self.batch_max_deg <= 1 and self.batch_max_comb_deg <= 2:
                    self.job_batch_max_size = self.aggregate(100, self.aggregation_factor)

            if unit.size_mb < 2:
                self.job_batch_max_size = self.aggregate(100, self.aggregation_factor)
                if self.batch_max_deg <= 2 and self.batch_max_comb_deg <= 2:
                    self.job_batch_max_size = self.aggregate(200, self.aggregation_factor)
                if self.batch_max_deg <= 1 and self.batch_max_comb_deg <= 2:
                    self.job_batch_max_size = self.aggregate(300, self.aggregation_factor)

        if self.cur_batch_def.data_size != unit.data_size \
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

        job_data = job_tpl_hdr + '\n'.join(self.job_clean_batch) + '\n\n' + '\n'.join(self.job_batch)
        unit = self.cur_batch_def

        with open(self.job_file_path, 'w+') as fh:
            fh.write(job_data)
            fh.write(job_tpl_footer)

        ram = '12gb' if unit.size_mb > self.memory_threshold else '6gb'
        if unit.size_mb >= 4000:
            ram = '16gb'
        if unit.size_mb >= 7000:
            ram = '32gb'

        job_time = '%s:00:00' % self.max_hour_job
        if unit.size_mb < 110:
            job_time = '4:00:00'
        self.job_files.append((self.job_file_path, ram, job_time))

        self.cur_batch_def = None
        self.job_batch = []
        self.job_clean_batch = []
        self.batch_max_bl = 0
        self.batch_max_deg = 0
        self.batch_max_comb_deg = 0



