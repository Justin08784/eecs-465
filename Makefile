
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

.PHONY:hw2
hw2:
	zip -r hw2.zip hw2 -x $(EXCLUDE)

