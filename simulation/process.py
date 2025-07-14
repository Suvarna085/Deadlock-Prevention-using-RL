class Process:
    def __init__(self, pid):
        self.id = pid
        self.total_needs = []  # Total resources needed to complete
    
    def reset(self):
        self.total_needs = []