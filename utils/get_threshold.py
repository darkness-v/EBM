import math

def get_threshold(r_real, r_fake):
    r_threshold = (math.acos(r_real) + math.acos(r_fake)) / 2
    threshold = 1 - math.cos(r_threshold)
    return threshold