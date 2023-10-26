# change directory name below
dir$ = "/Users/ivy/Desktop/libriTry"

# two input directories, one output directory and one output file
createDirectory(dir$ + "/LibriSpeech-segmented")
dir_audio$ = dir$ + "/LibriSpeech/dev-clean"
dir_text$ = dir$ + "/LibriSpeech-TextGrids/LibriSpeech/dev-clean"
dir_out$ = dir$ + "/LibriSpeech-segmented"
Create Table with column names: "myTable", 0, "segment file id startTime endTime nSample"

Create Strings as directory list: "outerList", dir_audio$
nOuter = Get number of strings

# loop through outer directories
for i to nOuter
	selectObject: "Strings outerList"
	outerDir$ = Get string: i

	# create output directory
	createDirectory(dir_out$ + "/" + outerDir$)
	
	Create Strings as directory list: "innerList", dir_audio$ + "/" + outerDir$
	nInner = Get number of strings
	
	# loop through inner directories
	for j to nInner
		selectObject: "Strings innerList"
		innerDir$ = Get string: j
		
		# create output directory
		createDirectory(dir_out$ + "/" + outerDir$ + "/" + innerDir$)
		
		Create Strings as file list: "fileList", dir_audio$ + "/" + outerDir$ + "/" + innerDir$ + "/*.flac"
		nFile = Get number of strings
		
		# loop through all files
		for k to nFile
			selectObject: "Strings fileList"
			audioName$ = Get string: k
			name$ = audioName$ - ".flac"
			textName$ = name$ + ".TextGrid"
			Read from file: dir_audio$ + "/" + outerDir$ + "/" + innerDir$ + "/" + audioName$
			Read from file: dir_text$ + "/" + outerDir$ + "/" + innerDir$ + "/" + textName$

			nInterval = Get number of intervals: 2

			# loop through all intervals
			for l to nInterval
				selectObject: "TextGrid " + name$
				label$ = Get label of interval: 2, l
				id$ = string$ (l)
				startTime = Get start time of interval: 2, l
				endTime = Get end time of interval: 2, l

				# if it's not silence
				if label$ != "sil" and label$ != "sp" and label$ != ""
					
					# extract and save audio
					selectObject: "Sound " + name$
					Extract part: startTime, endTime, "rectangular", 1, "no"
					selectObject: "Sound " + name$ + "_part"
					nSample = Get number of samples
					Save as FLAC file: dir_out$ + "/" + outerDir$ + "/" + innerDir$ + "/" + name$ + "-" + id$ + ".flac"
					
					selectObject: "Sound " + name$ + "_part"
					Remove
					
					# keep notes
					selectObject: "Table myTable"
					Insert row: 1
					Set string value: 1, "segment", label$
					Set string value: 1, "file", name$
					Set string value: 1, "id", id$
					Set numeric value: 1, "startTime", startTime
					Set numeric value: 1, "endTime", endTime
					Set numeric value: 1, "nSample", nSample
				endif
			endfor

			selectObject: "TextGrid " + name$
			plusObject: "Sound " + name$
			Remove
			
		endfor
		
		selectObject: "Strings fileList"
		Remove

	endfor
	
	selectObject: "Strings innerList"
	Remove
	
	beginPause: "pause window"
	comment: "Click continue to proceed to the next folder."
	clicked = endPause: "Continue", 1

endfor

selectObject: "Strings outerList"
Remove

# save file
selectObject: "Table myTable"
Save as comma-separated file: dir$ + "/log.csv"
