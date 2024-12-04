class Point:
    def __init__(self, value):
        self.value = value  # Duration or weight of the point
        self.successors = []  # List of successor indices
        self.predecessors = []  # List of predecessor indices
        self.ES = float("inf")  # Early Start
        self.EF = float("inf")  # Early Finish
        self.LS = float("inf")  # Late Start
        self.LF = float("inf") # Late Finish
        self.variance = float(0)
