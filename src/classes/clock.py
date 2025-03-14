import threading
import time

class Clock:
    """Centralized simulation clock."""

    def __init__(self):
        self.time = 0  # Tiempo en milisegundos
        self.running = False  # Control del reloj
        self.lock = threading.Lock()  # Sincronización

    def start(self):
        """Inicia un hilo dedicado al reloj central."""
        self.running = True
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self):
        """Detiene el hilo del reloj."""
        self.running = False

    def _run(self):
        """Incrementa el reloj centralizado continuamente en milisegundos."""
        while self.running:
            with self.lock:
                self.time += 1
            time.sleep(0.001)  # 1 ms en tiempo real

    def get_current_time(self):
        """Obtiene el tiempo actual del reloj central."""
        with self.lock:
            return self.time

    def tick(self, increment=1):
        """Avanza el reloj en un número específico de milisegundos."""
        with self.lock:
            self.time += increment
            return self.time

clock = Clock()
