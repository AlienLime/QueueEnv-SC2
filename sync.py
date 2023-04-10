import asyncio
import queue
import threading

async def _host_game(q):
    while True:
        await asyncio.sleep(1)
        print("Running SC2..")
        item = q.get()
        q.put(f"from sc2: {item}")

def run_game(q):
    async def run_host_and_join():
        return await asyncio.gather(_host_game(q))
    asyncio.run(run_host_and_join())

class GT(threading.Thread):
    def __init__(self, *args, q=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.q = q
    def run(self):
        run_game(q)

if __name__ == "__main__":
    q = queue.Queue()
    t = GT(q=q)
    t.start()
    q.put("something! go east!")
    print(q.get())
    a = 2
    # run_game(q)
    # queue.sync_q.put("hello")
    pass