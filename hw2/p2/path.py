import os, sys
path_hw2 = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)
path_HW2files = os.path.join(
    path_hw2,
    "HW2files"
)
sys.path.append(path_HW2files)

path_p1 = os.path.join(
    path_hw2,
    "p1"
)
sys.path.append(path_p1)

print("path.py: paths included")

