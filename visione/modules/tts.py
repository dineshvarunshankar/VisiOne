from contextlib import ExitStack, suppress
import multiprocessing as mp
import os
import queue
import threading

from config import (
    DEVICE, TTS_BASE_RATE, TTS_ENGINE, TTS_LANGUAGE, TTS_MODEL,
    TTS_RATE, TTS_SPEAKER_WAV,
)


class CoquiVoice:
    def __init__(self, config):
        from TTS.api import TTS

        self.config = config
        self.device = "cpu" if config['device'] == "mps" else config['device']
        self.tts = TTS(model_name=config['model'], progress_bar=False).to(self.device)

    def synthesize(self, text):
        return self.tts.tts(
            text=text,
            speaker_wav=self.config['speaker'],
            language=self.config['lang'],
            speed=self.config['rate'] / self.config['base_rate'],
        )


def _voice_worker(input_queue, status_queue, generation, priorities, stopped, config):
    engine = None
    sd = None
    try:
        if config['engine'] == 'ai':
            import numpy as np
            import sounddevice as sd

            engine = CoquiVoice(config)
        else:
            import pyttsx3

            engine = pyttsx3.init()
            engine.setProperty('rate', config['rate'])
            engine.startLoop(False)
        status_queue.put(None)
        while not stopped.wait(0.01):
            with generation.get_lock():
                try:
                    version, text, priority = input_queue.get_nowait()
                except queue.Empty:
                    continue
                priorities[1] = -1
                if version != generation.value:
                    continue
                priorities[0] = priority
            try:
                if config['engine'] == 'ai':
                    wav = np.asarray(engine.synthesize(text), dtype=np.float32)
                    with generation.get_lock():
                        if stopped.is_set() or version != generation.value:
                            continue
                        sd.play(wav, samplerate=24000)
                    while sd.get_stream().active and not stopped.wait(0.01):
                        if version != generation.value:
                            break
                    sd.stop()
                else:
                    engine.say(text)
                    engine.iterate()
                    while engine.isBusy() and not stopped.wait(0.01):
                        if version != generation.value:
                            break
                        engine.iterate()
                    engine.stop()
            finally:
                with generation.get_lock():
                    priorities[0] = -1
    except Exception as exc:
        status_queue.put(str(exc))
    finally:
        if sd is not None:
            sd.stop()
        elif engine is not None:
            engine.stop()
            engine.endLoop()


class VoiceGuide:
    STARTUP_TIMEOUT = 60

    def __init__(self):
        self.cfg = {
            'engine': TTS_ENGINE, 'device': DEVICE, 'model': TTS_MODEL,
            'lang': TTS_LANGUAGE, 'rate': TTS_RATE,
            'base_rate': TTS_BASE_RATE, 'speaker': os.path.abspath(TTS_SPEAKER_WAV),
        }
        if TTS_ENGINE not in {'system', 'ai'}:
            raise ValueError(f"Unknown TTS engine: {TTS_ENGINE}")
        if TTS_ENGINE == 'ai' and not os.path.isfile(self.cfg['speaker']):
            raise FileNotFoundError(f"TTS speaker recording not found: {self.cfg['speaker']}")
        context = mp.get_context('spawn')
        self.queue = context.Queue(maxsize=1)
        self.status_queue = context.Queue()
        self.generation = context.Value('L', 0)
        self.priorities = context.Array('i', [-1, -1], lock=False)
        self.stopped = context.Event()
        self.lock = threading.Lock()
        self.ready = False
        self.process = context.Process(
            target=_voice_worker,
            args=(self.queue, self.status_queue, self.generation, self.priorities, self.stopped, self.cfg),
            daemon=True,
        )
        with ExitStack() as startup:
            startup.callback(self.stop)
            self.process.start()
            try:
                error = self.status_queue.get(timeout=self.STARTUP_TIMEOUT)
            except queue.Empty as exc:
                raise TimeoutError("Voice initialization timed out") from exc
            if error is not None:
                raise RuntimeError(f"Voice initialization failed: {error}")
            self.ready = True
            startup.pop_all()

    def speak(self, text: str, interrupt: bool = False, priority=None):
        if not text:
            return False
        with self.lock:
            if not self.ready or self.stopped.is_set():
                return False
            if not self.process.is_alive():
                self.ready = False
                try:
                    error = self.status_queue.get(timeout=0.1)
                except queue.Empty:
                    error = 'voice worker exited'
                raise RuntimeError(f"Voice playback failed: {error}")
            priority = (1 if interrupt else 0) if priority is None else priority
            if not isinstance(priority, int) or priority < 0:
                raise ValueError('Speech priority must be a nonnegative integer')
            with self.generation.get_lock():
                current_priority = max(self.priorities)
                if priority < current_priority or (interrupt and priority == current_priority):
                    return False
                preempt = current_priority >= 0 and priority > current_priority
                if preempt:
                    self.generation.value += 1
                    with suppress(queue.Empty):
                        self.queue.get(timeout=0.05)
                    self.priorities[1] = -1
                try:
                    self.queue.put(
                        (self.generation.value, text, priority),
                        timeout=0.1 if preempt else 0,
                    )
                except queue.Full:
                    return False
                self.priorities[1] = priority
                return True

    def stop(self):
        with self.lock:
            if self.stopped.is_set():
                return
            self.ready = False
            self.stopped.set()
        if self.process.pid is not None:
            self.process.join(timeout=1)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=1)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=1)
            self.process.close()
        for channel in (self.queue, self.status_queue):
            channel.cancel_join_thread()
            channel.close()
