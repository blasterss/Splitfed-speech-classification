import sounddevice as sd


class CaptureAudio:
    def __init__(self):
        pass

    def _audio_callback(self):
        pass

    def run_audio_proccesing(self):

        # Запуск потокового захвата
        with sd.InputStream(
            channels=1,
            samplerate=SAMPLE_RATE,
            callback=self._audio_callback,
            blocksize=FRAME_SIZE,
        ):
            try:
                while True:
                    sd.sleep(1000)
            except KeyboardInterrupt:
                pass
