from contextlib import ExitStack, suppress
import queue
import re
import threading
import time

import numpy as np
import sounddevice as sd
import whisper

from config import DEVICE, LISTENING_TIMEOUT, MIC_DEVICE_INDEX, WAKE_WORD


class QueryListener:
    def __init__(self, speaking_event=None, enabled: bool = True):
        self.enabled = enabled
        self.stream = None
        self.worker = None
        self.stopped = threading.Event()
        self.audio_queue = queue.Queue(maxsize=2)
        self.lock = threading.Lock()
        self.query = None
        self.speaking_event = speaking_event
        self.is_waiting = False
        self.wait_start = 0
        self.blocksize = 16000 * 3
        self.stop_words = {
            'find', 'search', 'look', 'see', 'where', 'wheres', 'is', 'are',
            'the', 'a', 'an', 'my', 'me', 'you', 'can', 'do', 'did',
        }
        if enabled:
            self.model = whisper.load_model('tiny.en', device=DEVICE)

    def start(self):
        if not self.enabled or self.stream is not None:
            return
        if self.stopped.is_set():
            raise RuntimeError('Listener has been stopped')
        with ExitStack() as startup:
            startup.callback(self.stop)
            self.stream = sd.InputStream(
                device=MIC_DEVICE_INDEX, samplerate=16000, channels=1,
                blocksize=self.blocksize, callback=self._capture,
            )
            self.worker = threading.Thread(target=self._listen, daemon=True)
            self.worker.start()
            self.stream.start()
            startup.pop_all()

    def _capture(self, indata, frames, timing, status):
        if self.stopped.is_set() or (self.speaking_event and self.speaking_event.is_set()):
            return
        audio = indata[:, 0].astype(np.float32).copy()
        if np.mean(np.abs(audio)) <= 0.002:
            return
        captured_end = time.monotonic()
        captured_start = captured_end - frames / 16000
        with suppress(queue.Full):
            self.audio_queue.put_nowait((audio, captured_start, captured_end))

    def _listen(self):
        while not self.stopped.is_set():
            try:
                audio, captured_start, captured_end = self.audio_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                result = self.model.transcribe(audio, fp16=False)
                if not self.stopped.is_set():
                    self._process_text(result['text'], captured_start, captured_end)
            except Exception as exc:
                print(f'[Listener Error] {exc}', flush=True)

    def _process_text(self, text, captured_start, captured_end):
        clean = re.sub(r'[^\w\s]', '', text.lower()).strip()
        if len(clean) < 2:
            return
        if self.is_waiting and captured_start - self.wait_start > LISTENING_TIMEOUT:
            self.is_waiting = False
        target = re.search(r'\b' + re.escape(WAKE_WORD.lower()) + r'\b', clean)
        if target:
            command = clean[target.end():].strip()
            if not command:
                self.is_waiting = True
                self.wait_start = captured_end
                return
        elif self.is_waiting:
            command = clean
        else:
            return
        query = ' '.join(word for word in command.split() if word not in self.stop_words)
        if len(query) > 1:
            with self.lock:
                self.query = query
            self.is_waiting = False

    def get_query(self):
        with self.lock:
            query, self.query = self.query, None
        return query

    def stop(self):
        self.stopped.set()
        stream, self.stream = self.stream, None
        with suppress(queue.Empty):
            while True:
                self.audio_queue.get_nowait()
        with ExitStack() as cleanup:
            if self.worker is not None:
                cleanup.callback(self.worker.join, timeout=1)
            if stream is not None:
                cleanup.callback(stream.close)
                cleanup.callback(stream.stop)
