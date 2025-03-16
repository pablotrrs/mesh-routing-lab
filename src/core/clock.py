import threading
import time


class Clock:
    """Centralized simulation clock.

    This class provides a centralized clock for the simulation, which can be started,
    stopped, and queried for the current time. It runs in a separate thread to ensure
    continuous time updates.

    Attributes:
        time (int): Current simulation time in milliseconds.
        running (bool): Indicates whether the clock is running.
        lock (threading.Lock): Lock to ensure thread-safe access to the clock.
    """

    def __init__(self) -> None:
        """Initializes the clock with default values."""
        self.time: int = 0
        self.running: bool = False
        self.lock: threading.Lock = threading.Lock()

    def start(self) -> None:
        """Starts the clock in a dedicated thread."""
        self.running = True
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self) -> None:
        """Stops the clock."""
        self.running = False

    def _run(self) -> None:
        """Continuously increments the clock in milliseconds.

        This method runs in a separate thread and updates the clock every millisecond.
        """
        while self.running:
            with self.lock:
                self.time += 1
            time.sleep(0.001)  # Simulate 1 ms in real time

    def get_current_time(self) -> int:
        """Returns the current simulation time.

        Returns:
            int: Current simulation time in milliseconds.
        """
        with self.lock:
            return self.time

    def tick(self, increment: int = 1) -> int:
        """Advances the clock by a specified number of milliseconds.

        Args:
            increment (int): Number of milliseconds to advance the clock. Defaults to 1.

        Returns:
            int: Updated simulation time after the increment.
        """
        with self.lock:
            self.time += increment
            return self.time


# Global instance of the clock
clock: Clock = Clock()
