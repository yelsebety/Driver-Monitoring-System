import time
import winsound  # Built-in Windows sound module (no external pip install needed)


class AlertManager:
    """Manages audio and visual alert triggering with a cooldown to prevent overlapping beeps."""

    def __init__(self, sound_enabled=True, cooldown_seconds=3.0):
        self.sound_enabled = sound_enabled
        self.cooldown_seconds = cooldown_seconds
        self._last_alert_time = 0.0

    def trigger(self, state):
        """Triggers an alert if the state is Drowsy or Distracted and cooldown has passed."""
        if state == "Normal":
            return False

        now = time.monotonic()
        if now - self._last_alert_time >= self.cooldown_seconds:
            self._last_alert_time = now
            self._play_sound(state)
            return True

        return False

    def _play_sound(self, state):
        if not self.sound_enabled:
            return
        
        try:
            if state == "Drowsy":
                # High-priority urgent alarm: 3 rapid high-pitched beeps
                for _ in range(3):
                    winsound.Beep(2500, 150)
                    time.sleep(0.05)
            elif state == "Distracted":
                # Warning alarm: 2 moderate beeps
                for _ in range(2):
                    winsound.Beep(1500, 200)
                    time.sleep(0.08)
        except Exception as e:
            print(f"Audio alert error: {e}")


if __name__ == "__main__":
    # Test block
    print("Testing AlertManager...")
    manager = AlertManager(sound_enabled=True, cooldown_seconds=2.0)

    print("Triggering Drowsy alert...")
    manager.trigger("Drowsy")
    time.sleep(1)

    print("Triggering Distracted alert...")
    manager.trigger("Distracted")
    print("Done.")