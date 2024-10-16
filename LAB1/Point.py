class Point:
    def __init__(self, value):
        self.value = value  # Duration or weight of the point
        self.successors = []  # List of successor indices
        self.predecessors = []  # List of predecessor indices
        self.ES = 0  # Early Start
        self.EF = 0  # Early Finish
        self.LS = float('inf')  # Late Start
        self.LF = float('inf')  # Late Finish