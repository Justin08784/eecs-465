
EXCLUDE = *.pdf\
	  *.zip\
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

