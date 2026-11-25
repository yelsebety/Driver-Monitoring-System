class HoldTimer:
    """Measures how long a condition has been continuously true, but forgives
    short gaps (flicker). The timer only resets after the condition has been
    false for longer than 'grace' seconds.

    update(active, now) returns the seconds the condition has held (0.0 if not).
    """

    def __init__(self, grace=0.5):
        self.grace = grace
        self.since = None       # when the condition started
        self.last_true = None   # last time the condition was true

    def update(self, active, now):
        if active:
            if self.since is None:
                self.since = now
            self.last_true = now
        elif self.since is not None and now - self.last_true > self.grace:
            self.since = None
            self.last_true = None
        return 0.0 if self.since is None else now - self.since

    def reset(self):
        self.since = None
        self.last_true = None
