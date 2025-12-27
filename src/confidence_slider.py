
class ConfidenceSlider:
    def __init__(self, initial_threshold=0.5):
        self.threshold = initial_threshold

    def adjust_threshold(self, value):
        self.threshold = value
        return self.threshold
