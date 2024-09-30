import os, sys
path_HW2files = os.path.join(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    ),
    "HW2files"
)
sys.path.append(path_HW2files)
print("path included")

