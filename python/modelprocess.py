import multiprocessing as mp
from typing import Callable, List, Dict, Any, Set, Union
import numpy as np
import asyncio

class ToyProcess:
    def __init__(self, init_func: Union[Callable, None], process_func: Callable, *args):
        self.init_func = init_func
        self.process_func = process_func
        self.init_args = args

        self.input_queue = mp.Queue()
        self.output_queue = mp.Queue()

        self.result_ready_event = mp.Event()
        self.stop_flag = False

        self.process: mp.Process = mp.Process(target=self._run_process)
        self.process.start()


    def _run_process(self):
        if self.init_func is not None:
            self.init_func(*self.init_args)
        
        while True:
            input_data: Dict[str, Any] = self.input_queue.get()
            if input_data is None:
                break

            result = self.process_func(**input_data)

            self.output_queue.put(result)
            self.result_ready_event.set()


    async def _wait_for_result(self):
        while not self.result_ready_event.is_set() and not self.stop_flag:
            await asyncio.sleep(0.01)


    async def get_result(self, **kwargs):
        self.result_ready_event.clear()
        self.input_queue.put(kwargs)

        try:
            await asyncio.wait_for(self._wait_for_result(), timeout=kwargs.get('timeout', None))
        except asyncio.TimeoutError:
            return None

        if self.stop_flag:
            return None
        return self.output_queue.get()


    def stop(self):
        self.stop_flag = True
        self.input_queue.put(None)
        self.process.join()
        self.input_queue.close()
        self.output_queue.close()
        self.result_ready_event.clear()