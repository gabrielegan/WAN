from math import log
import glob
import os
import sys
try:
    import cupy as cp
except ModuleNotFoundError:
    cp = None
import numpy as np

def eliminateSinks(anyMatrix):			# If any row in a square matrix is all zeroes, that indicates a
                                        #   function word not followed (with our window) by any other function
                                        #   words. The limit probability of such a 'sink' would be 1 and all
                                        #   other limit probabilities would be 0, since there is no way out of
                                        #   this state. To fix this we set all the cells in that row to 1 divided
                                        #   by the number width ( = the height) of the matrix.
    dummyMatrix=[]										# Because of the weird way Python treats "=" we need to take
    for i in range(0, len(anyMatrix[0])):				#   a dummy copy of the input matrix and work only on that
        dummyMatrix.append([0] * len(anyMatrix[0]))		# So first we create the copy and populate it with zeroes
    
    for i in range(0,len(anyMatrix[0])):							# iterate thru rows
        rowSum=0
        for j in range(0,len(anyMatrix[0])):						# iterate thru columns in this row
            rowSum = rowSum + anyMatrix[i][j]						# build rowSum
        if rowSum == 0:												# once rowSum calculated, if it's zero
            for j in range(0,len(anyMatrix[0])):					#   iterate thru same row again, setting each
                dummyMatrix[i][j] = 1 / len(anyMatrix[0])			#   cell in dummyMatrix to 1/matrix-width
        else:
            for j in range(0,len(anyMatrix[0])):					# or else just copy the anyMatrix value
                dummyMatrix[i][j] = anyMatrix[i][j]					#   over to the new dummyMatrix
    return dummyMatrix								

def limitProbabilities(anyScores, anyCounts):				# Multiply a square matrix by itself 100 times
    copyOfanyScores = anyScores								# This holds the original matrix so we can keep reusing it
    for loop in range(0,100):								# Do this 100 times ...
        product = []										# 'product' is a temporary matrix holding the result
        for i in range(0, len(anyScores[0])):				#   of (anyMatrix * copyOfanyMatrix) each time we
            product.append([0] * len(anyScores[0]))			#   go through the loop. We have to initialize it to 0s
        for i in range (0, len(anyScores[0])):				# iterate thru rows
                for j in range (0, len(anyScores[0])):		# iterate thru columns
                    for k in range (0, len(anyScores[0])):	# see 'multiply-matrices.txt' for how this loop works
                        product[i][j] = product[i][j] + (copyOfanyScores[i][k] * anyScores[k][j]) 
        anyScores=product									# once we'd done all the rows and columns, make the
                                                            #   result of the product be the new value for anyMatrix
                                                            #   while copyOfanyMatrix retains the starting value
    columnTotals = []						# Create 1D array to hold column totals after multiplication
    for z in range(0, len(anyScores)):		#   by initial probabilities based on raw counts of fword occurrences
        columnTotals.append(0)				#

    for column in range(0,len(anyScores)):	# Iterate through columns first
        for row in range(0,len(anyScores)):	# Iterate from top to bottom of column, adding to the running total this
                                            #   cell's value times the likelihood of starting the walk in this row,
                                            #   which is the count of occurrences of the fword this row represents
                                            #   (=anyCounts[row]) divided by all fword occurrences (=sum(anyCounts))
            columnTotals[column] = columnTotals[column]+anyScores[row][column]*(anyCounts[row]/sum(anyCounts))
    return columnTotals

def relativeEntropy(anyWAN1, anyWAN2, anyWAN1LimitProbs, anyIndicator):	
                                                            # For each pair of cells (same row,column) in anyWAN1
                                                            #   and anyWAN2 where both matrices' values are non-zero,
                                                            #   deduct the natural (base e) logarithm of the value in
                                                            #   anyWAN2 from the natural log of the value of the
                                                            #   in anyWAN1 and multiply by the value in anyWAN1 and
                                                            #   by the value of the corresponding limit proabability
                                                            #   in the one-dimensional list anyWAN1LimitProbs (ie the
                                                            #   element in anyWAN1LimitProbs at the index given by
                                                            #   by the row in anyWAN1)
    sigma = 0												# Our running total, added to by delta for each cell pair
    for i in range(0, len(anyWAN1[0])):
        for j in range(0,len(anyWAN1[0])):
            if (anyWAN1[i][j] != 0) and (anyWAN2[i][j] != 0) and (anyIndicator[i][j] !=0):
                delta = (log(anyWAN1[i][j]) - log(anyWAN2[i][j])) * anyWAN1[i][j] * anyWAN1LimitProbs[i]
                sigma = sigma + delta
    return sigma
            
def loadWAN(anyFileName):

    with open(anyFileName, 'r') as handle:
        inStream = handle.read()							# this reads the whole file in as one string

    asRows=inStream.split("\n")								# create a list of rows breaking the string Instream at \n
    width=len(asRows[0].split(","))							# calculate width matrix by breaking first row at its commas

    WAN=[]													# Create the WAN with 0 in each cell
    for row in range(0, width):								# the number of rows is same as width
        WAN.append([0] * width)							# because WAN is square 

    for row in range(0, width):						# iterate thru all rows except last (because width is one less)
        thisRow=asRows[row].split(",")				# break present row at commas to make a list of cell values
        for column in range(0,width):				# iterate through columns
            WAN[row][column] = float(thisRow[column])	# assign cell to the current element in the list thisRow

    textcounts=[0] * width						# recover raw word counts from final row of matrix
    lastRow=asRows[width].split(",")			# turn final row string {which is asRows[width]) into list split on commas
    for column in range(0,width):				# iterate through that list and stuff each item
        textcounts[column]=int(lastRow[column])#   in that list into the empty list text1counts as integers

    return (WAN, textcounts)

def limitProbabilities_gpu(anyScores, anyCounts):
    if cp is None:
        return limitProbabilities(anyScores, anyCounts)
    """GPU-accelerated version using CuPy for matrix exponentiation"""
    # Convert input to CuPy array (move to GPU)
    gpu_matrix = cp.asarray(anyScores, dtype=cp.float64)
    
    # Matrix exponentiation: raise to the 100th power
    # This is MUCH faster than looping 100 times
    gpu_result = cp.linalg.matrix_power(gpu_matrix, 100)
    
    # Calculate column totals weighted by initial probabilities
    # Convert anyCounts to GPU array
    gpu_counts = cp.asarray(anyCounts, dtype=cp.float64)
    total_counts = cp.sum(gpu_counts)
    
    # Weight each row by its probability (count/total)
    row_weights = gpu_counts / total_counts
    
    # Compute weighted column sums
    # For each column, sum: matrix[row][col] * (counts[row] / sum(counts))
    columnTotals_gpu = cp.sum(gpu_result * row_weights[:, cp.newaxis], axis=0)
    
    # Convert result back to CPU (NumPy array)
    columnTotals = cp.asnumpy(columnTotals_gpu)
    
    return columnTotals.tolist()

############### END OF FUNCTION DEFINITIONS #################

# Default indicator file (can be overridden with -i/--indicator on the CLI)
DEFAULT_INDICATOR_FILE = "6-authors-whole-plays-top-100-words.IND"


def _parse_args(argv):
    """
    CLI:
      python compareWANSnoprint.py [-i INDICATOR_FILE] <wan_pairs_file>
    """
    indicator_file = DEFAULT_INDICATOR_FILE
    positional = []

    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in ("-i", "--indicator"):
            if i + 1 >= len(argv):
                print("Error: -i/--indicator requires a value", file=sys.stderr)
                sys.exit(2)
            indicator_file = argv[i + 1]
            i += 2
            continue

        positional.append(arg)
        i += 1

    if len(positional) != 1:
        print(
            "Usage: python compareWANSnoprint.py [-i INDICATOR_FILE] <wan_pairs_file>",
            file=sys.stderr,
        )
        sys.exit(1)

    return indicator_file, positional[0]


indicatorFileName, wanPairsFile = _parse_args(sys.argv[1:])

with open(wanPairsFile, 'r') as handle:
    listOfWANPairs = [line.strip() for line in handle if line.strip()]

(indicator, throwaway) = loadWAN(indicatorFileName)

for pair in listOfWANPairs:
    (textFile1, throwAway) = pair.split(",")
    (throwAway, textFile2) = pair.split(",")
    (WAN1, text1counts) = loadWAN(textFile1)
    (WAN2, text2counts) = loadWAN(textFile2)

    # Use GPU-accelerated version (or fall back to CPU version if CuPy is not available)
    WAN1LimitProbs = limitProbabilities_gpu((eliminateSinks(WAN1)), text1counts)
    print(textFile1 + "," + textFile2 + "," + str(round(100 * relativeEntropy(WAN1, WAN2, WAN1LimitProbs, indicator), 2)))
