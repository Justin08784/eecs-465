import os, sys
path_hw2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path_HW2files = os.path.join(path_hw2, "HW2files")
path_p1 = os.path.join(path_hw2, "p1")

dirs = [path_hw2, path_HW2files, path_p1]

for dir in dirs:
    if dir in sys.path:
        continue
    sys.path.append(dir)

print("p2/path.py: paths included\n")

