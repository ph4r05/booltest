#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Strip-down version of
# https://github.com/ph4r05/input_objects

"""
Input objects taken from polynomial verifiers
"""


import hashlib
import logging
import os
import requests
import traceback
import threading
import time
import collections
import string
import random
import shutil
import subprocess
import signal
import sys

from booltest.crypto_util import aes_ctr, get_zero_vector, aes_ecb, dump_uint


logger = logging.getLogger(__name__)


def is_empty(x):
    """
    Returns true if string is None or empty
    :param x:
    :return:
    """
    return x is None or len(x) == 0


class InputObject(object):
    """
    Input stream object.
    Can be a file, stream, or something else
    """
    def __init__(self, rec=None, aux=None, *args, **kwargs):
        self.sha256 = hashlib.sha256()
        self.sha1 = hashlib.sha1()
        self.data_read = 0

        self.rec = rec
        self.aux = aux

        # readline iterators
        self._data = ''
        self._offset = 0  # position in the read stream
        self._done = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __repr__(self):
        return 'InputObject(data_read=%r)' % self.data_read

    def check(self):
        """
        Checks if stream is readable
        :return:
        """

    def size(self):
        """
        Returns the size of the data
        :return:
        """
        return -1

    def read(self, size=None):
        """
        Reads size of data
        :param size:
        :return:
        """
        raise NotImplementedError('Not implemented - base class')

    #
    # Helper functions
    #

    def handle(self):
        """
        Returns file like handle
        :return:
        """
        raise NotImplementedError('Not implemented - base class')

    def text(self):
        """
        Returns text output
        :return:
        """
        return self.read()

    def to_state(self):
        """
        Returns state dictionary for serialization
        :return:
        """
        js = collections.OrderedDict()
        js['type'] = 'InputObject'
        js['data_read'] = self.data_read
        return js

    def short_desc(self):
        """
        Short description of the current state, for logging
        :return:
        """
        return self.__repr__()

    def tell(self):
        """
        Current position
        :return:
        """
        return self.data_read

    def close(self):
        """
        Closes the file object
        :return:
        """
        pass

    def flush(self):
        """
        FLushes the input object - mostly does nothing
        :return:
        """
        pass

    #
    # Line reading
    #

    def __fill(self, num_bytes):
        """
        Fill the internal buffer with 'num_bytes' of data.
        @param num_bytes: int, number of bytes to read in (0 = everything)
        """
        if self._done:
            return

        while not num_bytes or len(self._data) < num_bytes:
            data = self.read(32768)  # generic read method
            if not data:
                self._done = True
                break

            self._data = self._data + data

    def __iter__(self):
        """
        Line iterator = itself
        :return:
        """
        return self

    def next(self):
        """
        Iterating line by line
        :return:
        """
        line = self.readline()
        if not line:
            raise StopIteration()
        return line

    def _read(self, size=0):
        """
        Sub read for line iterations - reading to the buffer
        :param size:
        :return:
        """
        self.__fill(size)
        if size:
            data = self._data[:size]
            self._data = self._data[size:]
        else:
            data = self._data
            self._data = ""
        self._offset = self._offset + len(data)
        return data

    def readline(self):
        """
        Read a single line
        :return:
        """
        # make sure we have an entire line
        while not self._done and "\n" not in self._data:
            self.__fill(len(self._data) + 512)

        pos = string.find(self._data, "\n") + 1
        if pos <= 0:
            return self._read()
        return self._read(pos)

    def readlines(self):
        """
        Return all lines as an array
        :return:
        """
        lines = []
        while True:
            line = self.readline()
            if not line:
                break
            lines.append(line)
        return lines


class FileInputObject(InputObject):
    """
    File input object - reading from the file
    """
    def __init__(self, fname, *args, **kwargs):
        super(FileInputObject, self).__init__(*args, **kwargs)
        self.fname = fname
        self.fh = None

    def __enter__(self):
        super(FileInputObject, self).__enter__()
        self.fh = open(self.fname, 'rb')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        super(FileInputObject, self).__exit__(exc_type, exc_val, exc_tb)
        try:
            self.fh.close()
        except:
            logger.error('Error when closing file %s descriptor' % self.fname)

    def __repr__(self):
        return 'FileInputObject(file=%r)' % self.fname

    def __str__(self):
        return self.fname

    def check(self):
        if not os.path.exists(self.fname):
            raise ValueError('File %s was not found' % self.fname)

    def size(self):
        return os.path.getsize(self.fname)

    def read(self, size=None):
        if size is None:
            size = -1
        data = self.fh.read(size)
        self.sha256.update(data)
        self.sha1.update(data)
        self.data_read += len(data)
        return data

    def handle(self):
        return self.fh

    def to_state(self):
        """
        Returns state dictionary for serialization
        :return:
        """
        js = super(FileInputObject, self).to_state()
        js['type'] = 'FileInputObject'
        js['fname'] = self.fname
        return js

    def short_desc(self):
        """
        Short description of the current state, for logging
        :return:
        """
        return 'FileInputObject(data_read=%r, file=%r)' % (self.data_read, self.fname)


class FileLikeInputObject(InputObject):
    """
    Reads data from file like objects - e.g., stdout, sockets, ...
    Does not close the handle on context exit.
    open_call can define how the file-handle is opened.
    """

    def __init__(self, fh=None, desc=None, open_call=None, aux=None, *args, **kwargs):
        super(FileLikeInputObject, self).__init__(*args, **kwargs)
        self.fh = fh
        self.desc = desc
        self.aux = aux
        self.open_call = open_call

    def __enter__(self):
        super(FileLikeInputObject, self).__enter__()
        if self.fh is not None:
            return
        if self.open_call is not None:
            x = self.open_call(self)
            if x is not None:
                self.fh = x
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        super(FileLikeInputObject, self).__exit__(exc_type, exc_val, exc_tb)
        try:
            self.fh.close()
        except:
            logger.error('Error when closing url %s descriptor' % self.desc)

    def __repr__(self):
        return 'FileLikeInputObject()'

    def __str__(self):
        if self.desc is not None:
            return '%s' % self.desc
        return 'file-handle'

    def size(self):
        return -1

    def read(self, size=None):
        data = self.fh.read(size)
        self.sha256.update(data)
        self.sha1.update(data)
        self.data_read += len(data)
        return data

    def handle(self):
        return self.fh


class StdinInputObject(InputObject):
    """
    Reads data from the stdin
    """
    def __init__(self, desc=None, *args, **kwargs):
        super(StdinInputObject, self).__init__(*args, **kwargs)
        self.desc = desc

    def __enter__(self):
        super(StdinInputObject, self).__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        super(StdinInputObject, self).__exit__(exc_type, exc_val, exc_tb)
        sys.stdin.close()

    def __repr__(self):
        return 'StdinInputObject()'

    def __str__(self):
        if self.desc is not None:
            return 'stdin-%s' % self.desc
        return 'stdin'

    def size(self):
        return -1

    def read(self, size=None):
        data = sys.stdin.read(size) if sys.version_info < (3,) else sys.stdin.buffer.read(size)
        self.sha1.update(data)
        self.sha256.update(data)
        self.data_read += len(data)
        return data


class BinaryInputObject(InputObject):
    """
    Provides set binary data
    """
    def __init__(self, data=None, desc=None, *args, **kwargs):
        super(BinaryInputObject, self).__init__(*args, **kwargs)
        self.data = data
        self.desc = desc

    def __repr__(self):
        return 'BinaryInputObject(%r)' % self.desc

    def __str__(self):
        if self.desc is not None:
            return '%s' % self.desc
        return self.__repr__()

    def __enter__(self):
        super(BinaryInputObject, self).__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        super(BinaryInputObject, self).__exit__(exc_type, exc_val, exc_tb)

    def size(self):
        return len(self.data)

    def read(self, size=None):
        bound_end = self.data_read + size if size is not None else None
        data = self.data[self.data_read:bound_end]
        self.sha1.update(data)
        self.sha256.update(data)
        self.data_read += len(data)
        return data


class CommandStdoutInputObject(InputObject):
    """
    Executes command, reads from its stdout - used with generators.
    """
    def __init__(self, cmd=None, seed=None, desc=None, *args, **kwargs):
        super(CommandStdoutInputObject, self).__init__(*args, **kwargs)
        self.cmd = cmd
        self.seed = seed
        self.desc = desc
        self.proc = None
        self.subio = None

    def __repr__(self):
        return 'CommandStdoutInputObject()'

    def __str__(self):
        if self.desc is not None:
            return '%s' % self.desc
        return 'cmd: %s' % self.cmd

    def __enter__(self):
        super(CommandStdoutInputObject, self).__enter__()

        self.proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     bufsize=1024, close_fds=True, shell=True, preexec_fn=os.setsid)
        self.subio = FileLikeInputObject(fh=self.proc.stdout, desc=self.cmd)

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.proc.stdout.close()
            self.proc.terminate()
            self.proc.kill()
            os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
        except Exception as e:
            logger.debug('Exception killing process: %s' % e)

        super(CommandStdoutInputObject, self).__exit__(exc_type, exc_val, exc_tb)

    def size(self):
        return -1

    def read(self, size=None):
        data = self.subio.read(size)
        self.sha1.update(data)
        self.sha256.update(data)
        self.data_read += len(data)
        return data


class AESInputObject(InputObject):
    """
    AES data input generation.
    """
    def __init__(self, seed=None, desc=None, *args, **kwargs):
        super(AESInputObject, self).__init__(*args, **kwargs)
        self.seed = seed
        self.desc = desc

    def __repr__(self):
        return 'AESInputObject(seed=%r)' % self.seed

    def __str__(self):
        if self.desc is not None:
            return '%s' % self.desc
        return 'aes-ctr(sha256(0x%x))' % self.seed

    def size(self):
        return -1

    def read(self, size=None):
        aes = aes_ctr(hashlib.sha256('%x' % self.seed).digest())
        data = aes.encrypt(get_zero_vector(size))
        self.sha1.update(data)
        self.sha256.update(data)
        self.data_read += len(data)
        return data


class LinkInputObject(InputObject):
    """
    Input object using link - remote load
    """
    def __init__(self, url, headers=None, auth=None, timeout=None, allow_head=True, *args, **kwargs):
        super(LinkInputObject, self).__init__(*args, **kwargs)
        self.url = url
        self.headers = headers
        self.auth = auth
        self.r = None
        self.hres_loaded = False
        self.hres = None
        self.timeout = timeout
        self.allow_head = allow_head
        self.kwargs = kwargs

    def head(self, **kwargs):
        return requests.head(self.url, allow_redirects=True,
                     headers=self.headers, auth=self.auth, timeout=self.timeout, **kwargs)

    def __enter__(self):
        super(LinkInputObject, self).__enter__()
        self.r = requests.get(self.url, stream=True, allow_redirects=True, headers=self.headers, auth=self.auth,
                              timeout=self.timeout,
                              **self.kwargs)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        super(LinkInputObject, self).__exit__(exc_type, exc_val, exc_tb)
        try:
            self.r.close()
        except:
            logger.error('Error when closing url %s descriptor' % self.url)

    def __repr__(self):
        return 'LinkInputObject(url=%r)' % self.url

    def __str__(self):
        return self.url

    def get_hres(self):
        if self.hres_loaded:
            return self.hres

        self.hres = self.head(**self.kwargs)
        self.hres_loaded = True
        return self.hres

    def check(self):
        if not self.allow_head:
            return True

        try:
            self.get_hres()
            self.hres.raise_for_status()
            return True
        except Exception as e:
            logger.debug('Exception in getting %s: %s' % (self.url, e), exc_info=e)
            return False

    def size(self):
        if not self.allow_head:
            return -1
        
        try:
            h = self.get_hres()
            return int(h.headers['content-length'])
        except Exception as e:
            logger.debug('Exception in getting size of %s: %s' % (self.url, e), exc_info=e)
            return -1

    def read(self, size=None):
        data = self.r.raw.read(size)
        self.sha256.update(data)
        self.sha1.update(data)
        self.data_read += len(data)
        return data

    def text(self):
        data = self.r.text.encode('utf8')
        self.sha256.update(data)
        self.sha1.update(data)
        self.data_read += len(data)
        return data

    def handle(self):
        return self.r.raw

    def to_state(self):
        """
        Returns state dictionary for serialization
        :return:
        """
        js = super(LinkInputObject, self).to_state()
        js['type'] = 'LinkInputObject'
        js['url'] = self.url
        js['headers'] = self.headers
        js['timeout'] = self.timeout
        js['rec'] = self.rec
        return js


class RequestFailedTooManyTimes(Exception):
    """Request just keeps failing"""


class RequestReturnedEmptyResponse(Exception):
    """Internally used exception to signalize need for reconnect"""


class ReconnectingLinkInputObject(InputObject):
    """
    Input object that is able to reconnect to the source in case of the problem.
    Link should support calling HEAD method and RangeBytes.
    If this is not supported no reconnection will be used.
    """
    def __init__(self, url, rec=None, headers=None, auth=None, timeout=None,
                 max_reconnects=None, start_offset=0, pre_data_reconnect_hook=None, *args, **kwargs):
        super(ReconnectingLinkInputObject, self).__init__(*args, **kwargs)
        self.url = url
        self.headers = headers
        self.auth = auth
        self.rec = rec
        self.timeout = timeout
        self.max_reconnects = max_reconnects
        self.start_offset = start_offset
        self.pre_data_reconnect_hook = pre_data_reconnect_hook

        # Overall state
        self.stop_event = threading.Event()
        self.content_length = None
        self.total_reconnections = 0
        self.reconnections = 0
        self.last_reconnection = 0
        self.head_headers = None
        self.range_bytes_supported = False

        # Current state
        self.r = None
        self.current_content_length = 0

        self.kwargs = kwargs

    def _interruptible_sleep(self, sleep_time):
        """
        Sleeps the current thread for given amount of seconds, stop event terminates the sleep - to exit the thread.
        :param sleep_time:
        :return:
        """
        if sleep_time is None:
            return

        sleep_time = float(sleep_time)

        if sleep_time == 0:
            return

        sleep_start = time.time()
        while not self.stop_event.is_set():
            time.sleep(0.1)
            if time.time() - sleep_start >= sleep_time:
                return

    def _sleep_adaptive(self, current_attempt):
        """
        Sleeps amount of time w.r.t, attempt
        :param current_attempt:
        :return:
        """
        if current_attempt <= 5:
            self._interruptible_sleep(10)
        elif current_attempt <= 15:
            self._interruptible_sleep(60)
        elif current_attempt <= 25:
            self._interruptible_sleep(5 * 60)
        else:
            self._interruptible_sleep(10 * 60)

    def _load_info(self):
        """
        Performs head request on the url to load info & capabilities
        :return:
        """
        r = None
        current_attempt = 0

        # First - determine full length & partial request support
        while not self.stop_event.is_set():
            try:
                r = requests.head(self.url, allow_redirects=True, headers=self.headers, auth=self.auth,
                                  timeout=self.timeout)
                if r.status_code / 100 != 2:
                    logger.error('Link %s does not support head request or link is broken' % self.url)
                    return
                r.raise_for_status()
                break

            except Exception as e:
                logger.warning('Exception in fetching the url: %s' % e)
                logger.debug(traceback.format_exc())
                current_attempt += 1
                if self.max_reconnects is not None and current_attempt >= self.max_reconnects:
                    raise RequestFailedTooManyTimes()
                self._sleep_adaptive(current_attempt)

        self.head_headers = r.headers

        # Load content length, quite essential
        try:
            self.content_length = int(r.headers['Content-Length'])
        except KeyError:
            logger.error('Link %s does not return content length' % self.url)

        # Determine if range partial request is supported by the server
        if 'Accept-Ranges' in r.headers:
            self.range_bytes_supported = 'bytes' in r.headers['Accept-Ranges']

        logger.debug('URL %s head loaded. Content length: %s, accept range: %s, headers: %s'
                     % (self.url, self.content_length, self.range_bytes_supported, self.head_headers))

    def _get_headers(self):
        """
        Builds headers for the request
        :return:
        """
        headers = dict(self.headers) if self.headers is not None else {}

        if (self.start_offset is None or self.start_offset == 0) and self.data_read == 0:
            return headers

        headers['Range'] = 'bytes=%s-' % (self.start_offset + self.data_read)
        return headers

    def _is_all_data_loaded(self):
        """
        Returns true if all requested data is loaded already.
        :return:
        """
        if self.content_length is None:
            logger.warning('Could not determine if finished...')
            return None

        return self.content_length - self.start_offset - self.data_read <= 0

    def _request(self):
        """
        Connects to the server
        :return:
        """
        headers = self._get_headers()

        # Close previous connection
        try:
            if self.r is not None:
                self.r.close()
        except:
            logger.warning('Error when closing old url %s connection' % self.url)

        # Iterate several times until we get the response
        current_attempt = 0
        while not self.stop_event.is_set():
            try:
                logger.info('Reconnecting[%02d, %02d] to the url: %s, timeout: %s, headers: %s'
                            % (current_attempt, self.reconnections, self.url, self.timeout, headers))
                self.r = requests.get(self.url, stream=True, allow_redirects=True, headers=headers, auth=self.auth,
                                      timeout=self.timeout, **self.kwargs)
                self.r.raise_for_status()
                break

            except Exception as e:
                logger.warning('Exception in fetching the url: %s' % e)
                logger.debug(traceback.format_exc())
                current_attempt += 1
                if self.max_reconnects is not None and current_attempt >= self.max_reconnects:
                    raise RequestFailedTooManyTimes()
                self._sleep_adaptive(current_attempt)

        self.reconnections += 1
        self.last_reconnection = time.time()

        # Load content length
        try:
            self.current_content_length = int(self.r.headers['Content-Length'])
        except KeyError:
            logger.error('Link %s does not return content length' % self.url)

    def __enter__(self):
        super(ReconnectingLinkInputObject, self).__enter__()

        # Load basic info
        self._load_info()

        # Initial request
        self._request()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        super(ReconnectingLinkInputObject, self).__exit__(exc_type, exc_val, exc_tb)
        try:
            self.r.close()
        except:
            logger.error('Error when closing url %s descriptor' % self.url)

    def __repr__(self):
        return 'ReconnectingLinkInputObject(url=%r)' % self.url

    def __str__(self):
        return self.url

    def check(self):
        return True

    def size(self):
        return -1

    def read(self, size=None):
        """
        Reading a given size of data from the stream.
        :param size:
        :return:
        """
        while not self.stop_event.is_set():
            try:
                data = self.r.raw.read(size)
                ln = len(data)

                # If we read empty data inspect if it is expected end of stream or not
                if ln == 0:
                    logger.info('Empty data read, total so far: %s, offset: %s, content length: %s'
                                % (self.data_read, self.start_offset, self.content_length))

                    all_data_loaded = self._is_all_data_loaded()

                    # Could not determine if final, end then. End also if read it all
                    if all_data_loaded is None or all_data_loaded is True:
                        return data

                    # Problems -> need to reconnect and try over
                    raise RequestReturnedEmptyResponse()

                # Non-null data, all went right -> pass further
                self.sha256.update(data)
                self.sha1.update(data)
                self.data_read += ln
                return data

            except Exception as e:
                logger.error('Exception when reading data: %s' % e)
                logger.debug(traceback.format_exc())

                # Going to reconnect, ask where we stopped
                if self.pre_data_reconnect_hook is not None:
                    self.pre_data_reconnect_hook(self)
                self._interruptible_sleep(10)
                self._request()
                continue

        # Unreachable
        return None

    def handle(self):
        return self.r.raw

    def to_state(self):
        """
        Returns state dictionary for serialization
        :return:
        """
        js = super(ReconnectingLinkInputObject, self).to_state()
        js['type'] = 'ReconnectingLinkInputObject'
        js['url'] = self.url
        js['start_offset'] = self.start_offset
        js['headers'] = dict(self.headers) if self.headers is not None else None
        js['timeout'] = self.timeout
        js['rec'] = self.rec

        js['max_reconnects'] = self.max_reconnects
        js['content_length'] = self.content_length
        js['total_reconnections'] = self.total_reconnections
        js['reconnections'] = self.reconnections
        js['last_reconnection'] = self.last_reconnection
        js['head_headers'] = dict(self.head_headers) if self.head_headers is not None else None
        js['range_bytes_supported'] = self.range_bytes_supported
        js['current_content_length'] = self.range_bytes_supported
        return js


class TeeInputObject(InputObject):
    """
    Tee input object - reading underlying data stream, with stream copy to a different file like object
    (e.g., a file)
    """
    def __init__(self, parent_fh, copy_fh=None, close_copy_on_exit=False, copy_fname=None, *args, **kwargs):
        super(TeeInputObject, self).__init__(*args, **kwargs)
        self.parent_fh = parent_fh
        self.copy_fh = copy_fh
        self.copy_fname = copy_fname
        self.copy_fname_tmp = None
        self.close_copy_on_exit = close_copy_on_exit

    def __enter__(self):
        super(TeeInputObject, self).__enter__()

        # Open temporary file, write to it, on finish rename
        if self.copy_fh is None and self.copy_fname is not None:
            self.copy_fname_tmp = '%s.%s.%s' % (self.copy_fname, int(time.time()*1000), random.randint(0, 1000))
            self.copy_fh = open(self.copy_fname_tmp, 'wb')
            logger.debug('Tee to temp file %s' % self.copy_fname_tmp)

        try:
            self.parent_fh.__enter__()
            return self
        except Exception as e:
            logger.debug('Exception when entering to the parent fh %s' % e)
            logger.debug(traceback.format_exc())

    def __exit__(self, exc_type, exc_val, exc_tb):
        super(TeeInputObject, self).__exit__(exc_type, exc_val, exc_tb)
        try:
            self.parent_fh.__exit__(exc_type, exc_val, exc_tb)
        except Exception as e:
            logger.debug('Exception when exiting to the parent fh %s' % e)
            logger.debug(traceback.format_exc())

        if self.close_copy_on_exit:
            try:
                self.copy_fh.close()
            except Exception as e:
                logger.debug('Exception when closing copy fh%s' % e)
                logger.debug(traceback.format_exc())

        if self.copy_fname_tmp is not None:
            logger.debug('Moving %s -> %s' % (self.copy_fname_tmp, self.copy_fname))
            shutil.move(self.copy_fname_tmp, self.copy_fname)

    def __repr__(self):
        return 'TeeInputObject(parent_fh=%r, copy_fh=%r)' % (self.parent_fh, self.copy_fh)

    def __str__(self):
        return str(self.parent_fh)

    def check(self):
        return self.parent_fh.check()

    def size(self):
        return self.parent_fh.size()

    def read(self, size=None):
        data = self.parent_fh.read(size)
        self.sha256.update(data)
        self.sha1.update(data)
        self.data_read += len(data)

        cur_ctr = 0
        while True:
            try:
                self.copy_fh.write(data)
                return data

            except Exception as e:
                cur_ctr += 1
                logger.warning('Exception when writing data (%s) to the underlying stream, err %s, %s'
                               % (len(data), cur_ctr, e))

                logger.debug(traceback.format_exc())
                time.sleep(10)

    def handle(self):
        return self.parent_fh.handle()

    def to_state(self):
        js = super(TeeInputObject, self).to_state()
        js['type'] = 'TeeInputObject'
        js['parent'] = self.parent_fh.to_state()
        return js

    def short_desc(self):
        """
        Short description of the current state, for logging
        :return:
        """
        return 'TeeInputObject(parent=%s)' % (self.parent_fh.short_desc())

    def flush(self):
        self.copy_fh.flush()


class MergedInputObject(InputObject):
    """
    Merges multiple file input objects into one.
    (e.g., a file)
    """
    def __init__(self, iobjs, close_after_use=True, *args, **kwargs):
        super(MergedInputObject, self).__init__(*args, **kwargs)
        self.iobjs = iobjs
        self.cur_iobj = 0
        self.finished = False
        self._close_after_use = close_after_use
        self._do_close = [False] * len(self.iobjs)

    def __enter__(self):
        super(MergedInputObject, self).__enter__()
        if len(self.iobjs) == 0:
            return

        try:
            self._enter_sub(self.cur_iobj)
            return self
        except Exception as e:
            logger.debug('Exception when entering to the sub fh %s %s' % (self.cur_iobj, e))
            logger.debug(traceback.format_exc())

    def __exit__(self, exc_type, exc_val, exc_tb):
        super(MergedInputObject, self).__exit__(exc_type, exc_val, exc_tb)
        for idx in range(len(self.iobjs)):
            try:
                self._close_sub(idx)
            except Exception as e:
                logger.debug('Exception when exiting from the sub fh %s %s' % (self.cur_iobj, e))
                logger.debug(traceback.format_exc())

    def _enter_sub(self, idx):
        """
        Enter the sub data source with given index
        :param idx:
        :return:
        """
        self.iobjs[idx].__enter__()
        self._do_close[idx] = True

    def _close_sub(self, idx):
        """
        Close the source with the index if closable
        :param idx:
        :return:
        """
        if self._do_close[idx]:
            self.iobjs[idx].__exit__(None, None, None)
            self._do_close[idx] = False

    def __repr__(self):
        return 'MergedInputObject(iobjs=%r)' % (self.iobjs)

    def __str__(self):
        return self.__repr__()

    def check(self):
        return self.iobjs[self.cur_iobj].check()

    def size(self):
        return -1

    def read(self, size=None):
        while not self.finished:
            data = self.iobjs[self.cur_iobj].read(size)
            if is_empty(data):
                if self.cur_iobj + 1 == len(self.iobjs):
                    self.finished = True
                    return data

                if self._close_after_use:
                    self._close_sub(self.cur_iobj)
                self.cur_iobj += 1
                self._enter_sub(self.cur_iobj)
                continue

            self.sha256.update(data)
            self.sha1.update(data)
            self.data_read += len(data)
            return data
        return None

    def handle(self):
        return self.iobjs[self.cur_iobj].handle()

    def to_state(self):
        js = super(MergedInputObject, self).to_state()
        js['type'] = 'MergedInputObject'
        js['cur_iobj_idx'] = self.cur_iobj
        js['do_close'] = self._do_close
        js['cur_iobj'] = self.iobjs[self.cur_iobj].to_state()
        js['iobjs'] = [x.to_state() for x in self.iobjs]
        return js

    def short_desc(self):
        return 'MergedInputObject(data_read=%r, cur=%s)' % (self.data_read, self.iobjs[self.cur_iobj].short_desc())

    def flush(self):
        self.iobjs[self.cur_iobj].flush()




