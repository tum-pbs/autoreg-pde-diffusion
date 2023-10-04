import get_JHTDB as jhtdb
import time
import sys
import os
import signal
from multiprocessing import Process


if __name__ == '__main__':
    downloadTries = 200

    # terminate subprocess when terminated
    def signal_handler(sig, frame):
        for p in proc:
            if p:
                p.terminate()
        print('Terminated subprocesses!')
        sys.exit(0)
        os._exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


    proc = Process(target=jhtdb.download, args=(jhtdb.outDir, jhtdb.dataset, jhtdb.zSlices, jhtdb.size, jhtdb.strideSpatial, jhtdb.timeSteps))

    # poll download task every 10 seconds until it is done or fails and retry
    while downloadTries > 0 and not proc.exitcode == 0:
        if not proc.is_alive():
            print("\nCreating new downloader process...\n")
            proc = Process(target=jhtdb.download, args=(jhtdb.outDir, jhtdb.dataset, jhtdb.zSlices, jhtdb.size, jhtdb.strideSpatial, jhtdb.timeSteps))
            proc.start()
            downloadTries -= 1
            time.sleep(1)
        time.sleep(10)

    # no more open tasks, wait until remaining task is done
    proc.join()

    print("DONE - download objective COMPLETE") if proc.exitcode == 0 else print("DONE - download objective INCOMPLETE")