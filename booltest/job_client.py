#!/usr/bin/env python
# -*- coding: utf-8 -*-

#import zmq
import argparse
import coloredlogs
import logging
import time
import json
import websockets
import asyncio
import sys
from jsonpath_ng import jsonpath, parse

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.DEBUG)


def jsonpath(path, obj, allow_none=False):
    r = [m.value for m in parse(path).find(obj)]
    return r[0] if not allow_none else (r[0] if r else None)


class JobEntry:
    def __init__(self):
        self.unit = None
        self.uuid = None
        self.time_allocated = None
        self.worker_id = None
        self.idx = None
        self.finished = False
        self.failed = False
        self.retry_ctr = 0


class JobServer:
    def __init__(self):
        self.args = None
        self.job_entries = {}  # type: dict[str,JobEntry]
        self.finished_jobs = set()  # type: set[str]
        self.job_queue = []  # type: list[str]

        # Mapping worker_id -> job_id
        self.worker_map = {}  # type: dict[str,str]
        self.db_lock = asyncio.Lock()

    def job_get(self, worker_id=None):
        if len(self.job_queue) == 0:
            return 0
        uid = self.job_queue.pop()
        jb = self.job_entries[uid]
        jb.time_allocated = time.time()
        jb.worker_id = worker_id
        return jb

    def on_job_fail(self, jb):
        jb.retry_ctr += 1
        if jb.retry_ctr >= 2:
            jb.finished = True
            jb.failed = True
        else:
            jb.failed = False
            self.job_queue.append(jb.uuid)

    def on_job_alloc(self, jb, worker_id):
        # If worker had another job, finish it now. It failed probably
        if worker_id in self.worker_map:
            wold = self.worker_map[worker_id]
            if wold is not None:
                oldjb = self.job_entries[wold]
                self.on_job_fail(oldjb)

        self.worker_map[worker_id] = jb.uuid
        return jb

    def on_job_finished(self, uid, worker_id):
        jb = self.job_entries[uid]
        jb.finished = True
        self.worker_map[worker_id] = None

    def buid_resp_job(self, jb):
        if jb is None:
            return {'res': None}
        return {'res': jb.unit}

    async def on_ws_msg(self, websocket, path):
        logger.debug("on_msg")
        msg = await websocket.recv()
        logger.debug("Msg recv: %s" % (msg, ))

        async with self.db_lock:
            resp = self.on_msg(msg)

        resp_js = json.dumps(resp)
        await websocket.send(resp_js)

    def on_msg(self, message):
        try:
            jmsg = json.loads(message)
            if 'action' not in jmsg:
                raise ValueError("Invalid message")

            act = jmsg['action']
            wid = jmsg['uuid']
            if act == 'acquire':
                jb = self.job_get(wid)
                resp = self.buid_resp_job(jb)
                self.on_job_alloc(jb, wid)
                return resp

            elif act == 'finished':
                jid = jmsg['jid']
                self.on_job_finished(jid, wid)
                return {'resp': 'ok'}

            else:
                logger.info("Unknown action: [%s]" % (act,))

        except Exception as e:
            logger.warning("Exception in job handling: %s" % (e,), exc_info=e)
        return {'error': 'invalid'}

    async def work(self):
        # context = zmq.Context()
        # socket = context.socket(zmq.REP)
        # socket.bind("tcp://*:%s" % self.args.port)
        start_server = websockets.serve(self.on_ws_msg, "0.0.0.0", self.args.port)
        logger.info("Server started at 0.0.0.0:%s, python: %s" % (self.args.port, sys.version))
        await start_server

        # loop = asyncio.get_event_loop()
        # loop.run_forever()

    async def main(self):
        parser = self.argparse()
        self.args = parser.parse_args()

        for fl in self.args.files:
            with open(fl) as fh:
                js = json.load(fh)
            jobs = js['jobs']
            for j in jobs:
                je = JobEntry()
                je.unit = j
                je.uuid = j['uuid']
                je.idx = len(self.job_entries)
                self.job_entries[je.uuid] = je
                self.job_queue.append(je.uuid)

        logger.info("Jobs in the queue: %s" % (len(self.job_queue),))
        return await self.work()

    def argparse(self):
        parser = argparse.ArgumentParser(
            description='Job provider')

        parser.add_argument('--debug', dest='debug', action='store_const', const=True,
                            help='enables debug mode')
        parser.add_argument('--port', dest='port', type=int, default=4688,
                            help='port to bind to')

        parser.add_argument('files', nargs=argparse.ZERO_OR_MORE, default=[],
                            help='job files')

        return parser


async def amain():
    js = JobServer()
    await js.main()


def main():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(amain())
    loop.run_forever()
    loop.close()
    # asyncio.get_event_loop().run_until_complete(js.main)
    # asyncio.get_event_loop().run_forever()


if __name__ == '__main__':
    main()


# import json
# async def hello():
#     uri = "ws://localhost:4688"
#     async with websockets.connect(uri) as websocket:
#         await websocket.send(json.dumps({'test':'test'}))
#         greeting = await websocket.recv()
#         print(f"< {greeting}")
