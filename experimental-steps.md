# Experimental steps

* Use `word-freq-by-counts.py` on the concaternated authorial canons to find top 100 words across all the canons.

* Use `makeWINDOWS.py` to divide texts into rolling windows of all the plays in all the canons. The easiest way to do this is to copy all the plays to be divided into a new folder and do `dir /b > dir.txt` and then copy the resulting text into Word and replace "^p" with "',^p'" to wrap "'...'," around each title so it can be a list in `makeWINDOWS.py`.

* Use `makeWANnoprint.py` to make WANs for all the rolling windows and the individual authorial whole canons and all the individual authorial canons minus each play in turn. Do this by putting all these files in a folder and then `dir /b > dir.txt` to derive the list of files for `makeWANnoprint.py`. Remember to point to file containing the top 100 words.

* Use `makeINDICATOR.py` to find the indicator matrix for all the authorial canons considered together.

* Use `compareWANSnoprint.py` to compare each rolling window with the authorial canon of its author (minus the play this window comes from) and to all the other authorial canons. Its output is a CSV showing on each line "text1, text2, H(text1,text2)". To make the list of 'text1, text2' pairs that this script needs, use `makeBLOCKLIST.py` which needs a directory listing of the block WANs: do `dir /b *.WAN` and delete the notes that aren't blocks (which will be the ones for whole canons and whole-canon-minus-play) and wrap '...', around the directory listing using Word. (Or just pull list of `.win` files out of `makeWANnoprint.py` to get blocks.)


