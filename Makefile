
EXCLUDE = *.pdf\
	  hw2/*.pdf\
	  *.zip\
	  *.DS_Store\
	  *hw2/.DS_Store\
	  *__pycache__/*\
	  *writeup/*\
	  *.gz\
	  *.fls\
	  *.aux\
	  *.fdb_latexmk\
	  *.log\
	  *.iso

EXCLUDE_HW3 = *.pdf\
	  hw3/*.pdf\
	  *.zip\
	  *.DS_Store\
	  *hw2/.DS_Store\
	  *hw3/.DS_Store\
	  *__pycache__/*\
	  *writeup/*\
	  *.gz\
	  *.fls\
	  *.aux\
	  *.fdb_latexmk\
	  *.log\
	  *.iso\
	  hw3/2d_rr.py\
	  *.p\
	  *.npy\
.PHONY:hw2
hw2:
	zip -r hw2.zip hw2 -x $(EXCLUDE)

.PHONY:hw3
hw3:
	zip -r hw3.zip hw3 -x $(EXCLUDE_HW3)
