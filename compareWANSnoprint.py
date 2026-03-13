from math import log
import cupy as cp
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

# indicatorFileName = 'all-8-alls.IND'
indicatorFileName = 'all-1s.IND'

listOfWANPairs = [

'AF.WAN,all-Chapman-minus-AF.WAN',
'AF.WAN,all-Fletcher.WAN',
'AF.WAN,all-Greene.WAN',
'AF.WAN,all-Jonson.WAN',
'AF.WAN,all-Marlowe.WAN',
'AF.WAN,all-Middleton.WAN',
'AF.WAN,all-Peele.WAN',
'AF.WAN,all-Shakespeare.WAN',

'BBA.WAN,all-Chapman-minus-BBA.WAN',
'BBA.WAN,all-Fletcher.WAN',
'BBA.WAN,all-Greene.WAN',
'BBA.WAN,all-Jonson.WAN',
'BBA.WAN,all-Marlowe.WAN',
'BBA.WAN,all-Middleton.WAN',
'BBA.WAN,all-Peele.WAN',
'BBA.WAN,all-Shakespeare.WAN',

'BUS.WAN,all-Chapman-minus-BUS.WAN',
'BUS.WAN,all-Fletcher.WAN',
'BUS.WAN,all-Greene.WAN',
'BUS.WAN,all-Jonson.WAN',
'BUS.WAN,all-Marlowe.WAN',
'BUS.WAN,all-Middleton.WAN',
'BUS.WAN,all-Peele.WAN',
'BUS.WAN,all-Shakespeare.WAN',

'BYR.WAN,all-Chapman-minus-BYR.WAN',
'BYR.WAN,all-Fletcher.WAN',
'BYR.WAN,all-Greene.WAN',
'BYR.WAN,all-Jonson.WAN',
'BYR.WAN,all-Marlowe.WAN',
'BYR.WAN,all-Middleton.WAN',
'BYR.WAN,all-Peele.WAN',
'BYR.WAN,all-Shakespeare.WAN',

'CAE.WAN,all-Chapman-minus-CAE.WAN',
'CAE.WAN,all-Fletcher.WAN',
'CAE.WAN,all-Greene.WAN',
'CAE.WAN,all-Jonson.WAN',
'CAE.WAN,all-Marlowe.WAN',
'CAE.WAN,all-Middleton.WAN',
'CAE.WAN,all-Peele.WAN',
'CAE.WAN,all-Shakespeare.WAN',

'GOO.WAN,all-Chapman-minus-GOO.WAN',
'GOO.WAN,all-Fletcher.WAN',
'GOO.WAN,all-Greene.WAN',
'GOO.WAN,all-Jonson.WAN',
'GOO.WAN,all-Marlowe.WAN',
'GOO.WAN,all-Middleton.WAN',
'GOO.WAN,all-Peele.WAN',
'GOO.WAN,all-Shakespeare.WAN',

'GU.WAN,all-Chapman-minus-GU.WAN',
'GU.WAN,all-Fletcher.WAN',
'GU.WAN,all-Greene.WAN',
'GU.WAN,all-Jonson.WAN',
'GU.WAN,all-Marlowe.WAN',
'GU.WAN,all-Middleton.WAN',
'GU.WAN,all-Peele.WAN',
'GU.WAN,all-Shakespeare.WAN',

'HUM.WAN,all-Chapman-minus-HUM.WAN',
'HUM.WAN,all-Fletcher.WAN',
'HUM.WAN,all-Greene.WAN',
'HUM.WAN,all-Jonson.WAN',
'HUM.WAN,all-Marlowe.WAN',
'HUM.WAN,all-Middleton.WAN',
'HUM.WAN,all-Peele.WAN',
'HUM.WAN,all-Shakespeare.WAN',

'MAY.WAN,all-Chapman-minus-MAY.WAN',
'MAY.WAN,all-Fletcher.WAN',
'MAY.WAN,all-Greene.WAN',
'MAY.WAN,all-Jonson.WAN',
'MAY.WAN,all-Marlowe.WAN',
'MAY.WAN,all-Middleton.WAN',
'MAY.WAN,all-Peele.WAN',
'MAY.WAN,all-Shakespeare.WAN',

'OLI.WAN,all-Chapman-minus-OLI.WAN',
'OLI.WAN,all-Fletcher.WAN',
'OLI.WAN,all-Greene.WAN',
'OLI.WAN,all-Jonson.WAN',
'OLI.WAN,all-Marlowe.WAN',
'OLI.WAN,all-Middleton.WAN',
'OLI.WAN,all-Peele.WAN',
'OLI.WAN,all-Shakespeare.WAN',

'RBD.WAN,all-Chapman-minus-RBD.WAN',
'RBD.WAN,all-Fletcher.WAN',
'RBD.WAN,all-Greene.WAN',
'RBD.WAN,all-Jonson.WAN',
'RBD.WAN,all-Marlowe.WAN',
'RBD.WAN,all-Middleton.WAN',
'RBD.WAN,all-Peele.WAN',
'RBD.WAN,all-Shakespeare.WAN',

'WID.WAN,all-Chapman-minus-WID.WAN',
'WID.WAN,all-Fletcher.WAN',
'WID.WAN,all-Greene.WAN',
'WID.WAN,all-Jonson.WAN',
'WID.WAN,all-Marlowe.WAN',
'WID.WAN,all-Middleton.WAN',
'WID.WAN,all-Peele.WAN',
'WID.WAN,all-Shakespeare.WAN',

'BON.WAN,all-Chapman.WAN',
'BON.WAN,all-Fletcher-minus-BON.WAN',
'BON.WAN,all-Greene.WAN',
'BON.WAN,all-Jonson.WAN',
'BON.WAN,all-Marlowe.WAN',
'BON.WAN,all-Middleton.WAN',
'BON.WAN,all-Peele.WAN',
'BON.WAN,all-Shakespeare.WAN',

'CHA.WAN,all-Chapman.WAN',
'CHA.WAN,all-Fletcher-minus-CHA.WAN',
'CHA.WAN,all-Greene.WAN',
'CHA.WAN,all-Jonson.WAN',
'CHA.WAN,all-Marlowe.WAN',
'CHA.WAN,all-Middleton.WAN',
'CHA.WAN,all-Peele.WAN',
'CHA.WAN,all-Shakespeare.WAN',

'FAI.WAN,all-Chapman.WAN',
'FAI.WAN,all-Fletcher-minus-FAI.WAN',
'FAI.WAN,all-Greene.WAN',
'FAI.WAN,all-Jonson.WAN',
'FAI.WAN,all-Marlowe.WAN',
'FAI.WAN,all-Middleton.WAN',
'FAI.WAN,all-Peele.WAN',
'FAI.WAN,all-Shakespeare.WAN',

'HML.WAN,all-Chapman.WAN',
'HML.WAN,all-Fletcher-minus-HML.WAN',
'HML.WAN,all-Greene.WAN',
'HML.WAN,all-Jonson.WAN',
'HML.WAN,all-Marlowe.WAN',
'HML.WAN,all-Middleton.WAN',
'HML.WAN,all-Peele.WAN',
'HML.WAN,all-Shakespeare.WAN',

'ISL.WAN,all-Chapman.WAN',
'ISL.WAN,all-Fletcher-minus-ISL.WAN',
'ISL.WAN,all-Greene.WAN',
'ISL.WAN,all-Jonson.WAN',
'ISL.WAN,all-Marlowe.WAN',
'ISL.WAN,all-Middleton.WAN',
'ISL.WAN,all-Peele.WAN',
'ISL.WAN,all-Shakespeare.WAN',

'LOY.WAN,all-Chapman.WAN',
'LOY.WAN,all-Fletcher-minus-LOY.WAN',
'LOY.WAN,all-Greene.WAN',
'LOY.WAN,all-Jonson.WAN',
'LOY.WAN,all-Marlowe.WAN',
'LOY.WAN,all-Middleton.WAN',
'LOY.WAN,all-Peele.WAN',
'LOY.WAN,all-Shakespeare.WAN',

'MAD.WAN,all-Chapman.WAN',
'MAD.WAN,all-Fletcher-minus-MAD.WAN',
'MAD.WAN,all-Greene.WAN',
'MAD.WAN,all-Jonson.WAN',
'MAD.WAN,all-Marlowe.WAN',
'MAD.WAN,all-Middleton.WAN',
'MAD.WAN,all-Peele.WAN',
'MAD.WAN,all-Shakespeare.WAN',

'MON.WAN,all-Chapman.WAN',
'MON.WAN,all-Fletcher-minus-MON.WAN',
'MON.WAN,all-Greene.WAN',
'MON.WAN,all-Jonson.WAN',
'MON.WAN,all-Marlowe.WAN',
'MON.WAN,all-Middleton.WAN',
'MON.WAN,all-Peele.WAN',
'MON.WAN,all-Shakespeare.WAN',

'PIL.WAN,all-Chapman.WAN',
'PIL.WAN,all-Fletcher-minus-PIL.WAN',
'PIL.WAN,all-Greene.WAN',
'PIL.WAN,all-Jonson.WAN',
'PIL.WAN,all-Marlowe.WAN',
'PIL.WAN,all-Middleton.WAN',
'PIL.WAN,all-Peele.WAN',
'PIL.WAN,all-Shakespeare.WAN',

'RUL.WAN,all-Chapman.WAN',
'RUL.WAN,all-Fletcher-minus-RUL.WAN',
'RUL.WAN,all-Greene.WAN',
'RUL.WAN,all-Jonson.WAN',
'RUL.WAN,all-Marlowe.WAN',
'RUL.WAN,all-Middleton.WAN',
'RUL.WAN,all-Peele.WAN',
'RUL.WAN,all-Shakespeare.WAN',

'VAL.WAN,all-Chapman.WAN',
'VAL.WAN,all-Fletcher-minus-VAL.WAN',
'VAL.WAN,all-Greene.WAN',
'VAL.WAN,all-Jonson.WAN',
'VAL.WAN,all-Marlowe.WAN',
'VAL.WAN,all-Middleton.WAN',
'VAL.WAN,all-Peele.WAN',
'VAL.WAN,all-Shakespeare.WAN',

'WFAM.WAN,all-Chapman.WAN',
'WFAM.WAN,all-Fletcher-minus-WFAM.WAN',
'WFAM.WAN,all-Greene.WAN',
'WFAM.WAN,all-Jonson.WAN',
'WFAM.WAN,all-Marlowe.WAN',
'WFAM.WAN,all-Middleton.WAN',
'WFAM.WAN,all-Peele.WAN',
'WFAM.WAN,all-Shakespeare.WAN',

'WIL.WAN,all-Chapman.WAN',
'WIL.WAN,all-Fletcher-minus-WIL.WAN',
'WIL.WAN,all-Greene.WAN',
'WIL.WAN,all-Jonson.WAN',
'WIL.WAN,all-Marlowe.WAN',
'WIL.WAN,all-Middleton.WAN',
'WIL.WAN,all-Peele.WAN',
'WIL.WAN,all-Shakespeare.WAN',

'WPL.WAN,all-Chapman.WAN',
'WPL.WAN,all-Fletcher-minus-WPL.WAN',
'WPL.WAN,all-Greene.WAN',
'WPL.WAN,all-Jonson.WAN',
'WPL.WAN,all-Marlowe.WAN',
'WPL.WAN,all-Middleton.WAN',
'WPL.WAN,all-Peele.WAN',
'WPL.WAN,all-Shakespeare.WAN',

'WPR.WAN,all-Chapman.WAN',
'WPR.WAN,all-Fletcher-minus-WPR.WAN',
'WPR.WAN,all-Greene.WAN',
'WPR.WAN,all-Jonson.WAN',
'WPR.WAN,all-Marlowe.WAN',
'WPR.WAN,all-Middleton.WAN',
'WPR.WAN,all-Peele.WAN',
'WPR.WAN,all-Shakespeare.WAN',

'ALP.WAN,all-Chapman.WAN',
'ALP.WAN,all-Fletcher.WAN',
'ALP.WAN,all-Greene-minus-ALP.WAN',
'ALP.WAN,all-Jonson.WAN',
'ALP.WAN,all-Marlowe.WAN',
'ALP.WAN,all-Middleton.WAN',
'ALP.WAN,all-Peele.WAN',
'ALP.WAN,all-Shakespeare.WAN',

'FRI.WAN,all-Chapman.WAN',
'FRI.WAN,all-Fletcher.WAN',
'FRI.WAN,all-Greene-minus-FRI.WAN',
'FRI.WAN,all-Jonson.WAN',
'FRI.WAN,all-Marlowe.WAN',
'FRI.WAN,all-Middleton.WAN',
'FRI.WAN,all-Peele.WAN',
'FRI.WAN,all-Shakespeare.WAN',

'J4.WAN,all-Chapman.WAN',
'J4.WAN,all-Fletcher.WAN',
'J4.WAN,all-Greene-minus-J4.WAN',
'J4.WAN,all-Jonson.WAN',
'J4.WAN,all-Marlowe.WAN',
'J4.WAN,all-Middleton.WAN',
'J4.WAN,all-Peele.WAN',
'J4.WAN,all-Shakespeare.WAN',

'ORL.WAN,all-Chapman.WAN',
'ORL.WAN,all-Fletcher.WAN',
'ORL.WAN,all-Greene-minus-ORL.WAN',
'ORL.WAN,all-Jonson.WAN',
'ORL.WAN,all-Marlowe.WAN',
'ORL.WAN,all-Middleton.WAN',
'ORL.WAN,all-Peele.WAN',
'ORL.WAN,all-Shakespeare.WAN',

'ALC.WAN,all-Chapman.WAN',
'ALC.WAN,all-Fletcher.WAN',
'ALC.WAN,all-Greene.WAN',
'ALC.WAN,all-Jonson-minus-ALC.WAN',
'ALC.WAN,all-Marlowe.WAN',
'ALC.WAN,all-Middleton.WAN',
'ALC.WAN,all-Peele.WAN',
'ALC.WAN,all-Shakespeare.WAN',

'BF.WAN,all-Chapman.WAN',
'BF.WAN,all-Fletcher.WAN',
'BF.WAN,all-Greene.WAN',
'BF.WAN,all-Jonson-minus-BF.WAN',
'BF.WAN,all-Marlowe.WAN',
'BF.WAN,all-Middleton.WAN',
'BF.WAN,all-Peele.WAN',
'BF.WAN,all-Shakespeare.WAN',

'CAT.WAN,all-Chapman.WAN',
'CAT.WAN,all-Fletcher.WAN',
'CAT.WAN,all-Greene.WAN',
'CAT.WAN,all-Jonson-minus-CAT.WAN',
'CAT.WAN,all-Marlowe.WAN',
'CAT.WAN,all-Middleton.WAN',
'CAT.WAN,all-Peele.WAN',
'CAT.WAN,all-Shakespeare.WAN',

'CR.WAN,all-Chapman.WAN',
'CR.WAN,all-Fletcher.WAN',
'CR.WAN,all-Greene.WAN',
'CR.WAN,all-Jonson-minus-CR.WAN',
'CR.WAN,all-Marlowe.WAN',
'CR.WAN,all-Middleton.WAN',
'CR.WAN,all-Peele.WAN',
'CR.WAN,all-Shakespeare.WAN',

'DIAA.WAN,all-Chapman.WAN',
'DIAA.WAN,all-Fletcher.WAN',
'DIAA.WAN,all-Greene.WAN',
'DIAA.WAN,all-Jonson-minus-DIAA.WAN',
'DIAA.WAN,all-Marlowe.WAN',
'DIAA.WAN,all-Middleton.WAN',
'DIAA.WAN,all-Peele.WAN',
'DIAA.WAN,all-Shakespeare.WAN',

'EMI.WAN,all-Chapman.WAN',
'EMI.WAN,all-Fletcher.WAN',
'EMI.WAN,all-Greene.WAN',
'EMI.WAN,all-Jonson-minus-EMI.WAN',
'EMI.WAN,all-Marlowe.WAN',
'EMI.WAN,all-Middleton.WAN',
'EMI.WAN,all-Peele.WAN',
'EMI.WAN,all-Shakespeare.WAN',

'EMO.WAN,all-Chapman.WAN',
'EMO.WAN,all-Fletcher.WAN',
'EMO.WAN,all-Greene.WAN',
'EMO.WAN,all-Jonson-minus-EMO.WAN',
'EMO.WAN,all-Marlowe.WAN',
'EMO.WAN,all-Middleton.WAN',
'EMO.WAN,all-Peele.WAN',
'EMO.WAN,all-Shakespeare.WAN',

'EPI.WAN,all-Chapman.WAN',
'EPI.WAN,all-Fletcher.WAN',
'EPI.WAN,all-Greene.WAN',
'EPI.WAN,all-Jonson-minus-EPI.WAN',
'EPI.WAN,all-Marlowe.WAN',
'EPI.WAN,all-Middleton.WAN',
'EPI.WAN,all-Peele.WAN',
'EPI.WAN,all-Shakespeare.WAN',

'MAG.WAN,all-Chapman.WAN',
'MAG.WAN,all-Fletcher.WAN',
'MAG.WAN,all-Greene.WAN',
'MAG.WAN,all-Jonson-minus-MAG.WAN',
'MAG.WAN,all-Marlowe.WAN',
'MAG.WAN,all-Middleton.WAN',
'MAG.WAN,all-Peele.WAN',
'MAG.WAN,all-Shakespeare.WAN',

'NI.WAN,all-Chapman.WAN',
'NI.WAN,all-Fletcher.WAN',
'NI.WAN,all-Greene.WAN',
'NI.WAN,all-Jonson-minus-NI.WAN',
'NI.WAN,all-Marlowe.WAN',
'NI.WAN,all-Middleton.WAN',
'NI.WAN,all-Peele.WAN',
'NI.WAN,all-Shakespeare.WAN',

'POE.WAN,all-Chapman.WAN',
'POE.WAN,all-Fletcher.WAN',
'POE.WAN,all-Greene.WAN',
'POE.WAN,all-Jonson-minus-POE.WAN',
'POE.WAN,all-Marlowe.WAN',
'POE.WAN,all-Middleton.WAN',
'POE.WAN,all-Peele.WAN',
'POE.WAN,all-Shakespeare.WAN',

'SEJ.WAN,all-Chapman.WAN',
'SEJ.WAN,all-Fletcher.WAN',
'SEJ.WAN,all-Greene.WAN',
'SEJ.WAN,all-Jonson-minus-SEJ.WAN',
'SEJ.WAN,all-Marlowe.WAN',
'SEJ.WAN,all-Middleton.WAN',
'SEJ.WAN,all-Peele.WAN',
'SEJ.WAN,all-Shakespeare.WAN',

'SS.WAN,all-Chapman.WAN',
'SS.WAN,all-Fletcher.WAN',
'SS.WAN,all-Greene.WAN',
'SS.WAN,all-Jonson-minus-SS.WAN',
'SS.WAN,all-Marlowe.WAN',
'SS.WAN,all-Middleton.WAN',
'SS.WAN,all-Peele.WAN',
'SS.WAN,all-Shakespeare.WAN',

'STAP.WAN,all-Chapman.WAN',
'STAP.WAN,all-Fletcher.WAN',
'STAP.WAN,all-Greene.WAN',
'STAP.WAN,all-Jonson-minus-STAP.WAN',
'STAP.WAN,all-Marlowe.WAN',
'STAP.WAN,all-Middleton.WAN',
'STAP.WAN,all-Peele.WAN',
'STAP.WAN,all-Shakespeare.WAN',

'TUB.WAN,all-Chapman.WAN',
'TUB.WAN,all-Fletcher.WAN',
'TUB.WAN,all-Greene.WAN',
'TUB.WAN,all-Jonson-minus-TUB.WAN',
'TUB.WAN,all-Marlowe.WAN',
'TUB.WAN,all-Middleton.WAN',
'TUB.WAN,all-Peele.WAN',
'TUB.WAN,all-Shakespeare.WAN',

'VOLP.WAN,all-Chapman.WAN',
'VOLP.WAN,all-Fletcher.WAN',
'VOLP.WAN,all-Greene.WAN',
'VOLP.WAN,all-Jonson-minus-VOLP.WAN',
'VOLP.WAN,all-Marlowe.WAN',
'VOLP.WAN,all-Middleton.WAN',
'VOLP.WAN,all-Peele.WAN',
'VOLP.WAN,all-Shakespeare.WAN',

'1TAM.WAN,all-Chapman.WAN',
'1TAM.WAN,all-Fletcher.WAN',
'1TAM.WAN,all-Greene.WAN',
'1TAM.WAN,all-Jonson.WAN',
'1TAM.WAN,all-Marlowe-minus-1TAM.WAN',
'1TAM.WAN,all-Middleton.WAN',
'1TAM.WAN,all-Peele.WAN',
'1TAM.WAN,all-Shakespeare.WAN',

'2TAM.WAN,all-Chapman.WAN',
'2TAM.WAN,all-Fletcher.WAN',
'2TAM.WAN,all-Greene.WAN',
'2TAM.WAN,all-Jonson.WAN',
'2TAM.WAN,all-Marlowe-minus-2TAM.WAN',
'2TAM.WAN,all-Middleton.WAN',
'2TAM.WAN,all-Peele.WAN',
'2TAM.WAN,all-Shakespeare.WAN',

'E2.WAN,all-Chapman.WAN',
'E2.WAN,all-Fletcher.WAN',
'E2.WAN,all-Greene.WAN',
'E2.WAN,all-Jonson.WAN',
'E2.WAN,all-Marlowe-minus-E2.WAN',
'E2.WAN,all-Middleton.WAN',
'E2.WAN,all-Peele.WAN',
'E2.WAN,all-Shakespeare.WAN',

'FAU.WAN,all-Chapman.WAN',
'FAU.WAN,all-Fletcher.WAN',
'FAU.WAN,all-Greene.WAN',
'FAU.WAN,all-Jonson.WAN',
'FAU.WAN,all-Marlowe-minus-FAU.WAN',
'FAU.WAN,all-Middleton.WAN',
'FAU.WAN,all-Peele.WAN',
'FAU.WAN,all-Shakespeare.WAN',

'JOM.WAN,all-Chapman.WAN',
'JOM.WAN,all-Fletcher.WAN',
'JOM.WAN,all-Greene.WAN',
'JOM.WAN,all-Jonson.WAN',
'JOM.WAN,all-Marlowe-minus-JOM.WAN',
'JOM.WAN,all-Middleton.WAN',
'JOM.WAN,all-Peele.WAN',
'JOM.WAN,all-Shakespeare.WAN',

'MASS.WAN,all-Chapman.WAN',
'MASS.WAN,all-Fletcher.WAN',
'MASS.WAN,all-Greene.WAN',
'MASS.WAN,all-Jonson.WAN',
'MASS.WAN,all-Marlowe-minus-MASS.WAN',
'MASS.WAN,all-Middleton.WAN',
'MASS.WAN,all-Peele.WAN',
'MASS.WAN,all-Shakespeare.WAN',

'2MT.WAN,all-Chapman.WAN',
'2MT.WAN,all-Fletcher.WAN',
'2MT.WAN,all-Greene.WAN',
'2MT.WAN,all-Jonson.WAN',
'2MT.WAN,all-Marlowe.WAN',
'2MT.WAN,all-Middleton-minus-2MT.WAN',
'2MT.WAN,all-Peele.WAN',
'2MT.WAN,all-Shakespeare.WAN',

'CHASTE.WAN,all-Chapman.WAN',
'CHASTE.WAN,all-Fletcher.WAN',
'CHASTE.WAN,all-Greene.WAN',
'CHASTE.WAN,all-Jonson.WAN',
'CHASTE.WAN,all-Marlowe.WAN',
'CHASTE.WAN,all-Middleton-minus-CHASTE.WAN',
'CHASTE.WAN,all-Peele.WAN',
'CHASTE.WAN,all-Shakespeare.WAN',

'GAM.WAN,all-Chapman.WAN',
'GAM.WAN,all-Fletcher.WAN',
'GAM.WAN,all-Greene.WAN',
'GAM.WAN,all-Jonson.WAN',
'GAM.WAN,all-Marlowe.WAN',
'GAM.WAN,all-Middleton-minus-GAM.WAN',
'GAM.WAN,all-Peele.WAN',
'GAM.WAN,all-Shakespeare.WAN',

'HENG.WAN,all-Chapman.WAN',
'HENG.WAN,all-Fletcher.WAN',
'HENG.WAN,all-Greene.WAN',
'HENG.WAN,all-Jonson.WAN',
'HENG.WAN,all-Marlowe.WAN',
'HENG.WAN,all-Middleton-minus-HENG.WAN',
'HENG.WAN,all-Peele.WAN',
'HENG.WAN,all-Shakespeare.WAN',

'MDBW.WAN,all-Chapman.WAN',
'MDBW.WAN,all-Fletcher.WAN',
'MDBW.WAN,all-Greene.WAN',
'MDBW.WAN,all-Jonson.WAN',
'MDBW.WAN,all-Marlowe.WAN',
'MDBW.WAN,all-Middleton-minus-MDBW.WAN',
'MDBW.WAN,all-Peele.WAN',
'MDBW.WAN,all-Shakespeare.WAN',

'MIC.WAN,all-Chapman.WAN',
'MIC.WAN,all-Fletcher.WAN',
'MIC.WAN,all-Greene.WAN',
'MIC.WAN,all-Jonson.WAN',
'MIC.WAN,all-Marlowe.WAN',
'MIC.WAN,all-Middleton-minus-MIC.WAN',
'MIC.WAN,all-Peele.WAN',
'MIC.WAN,all-Shakespeare.WAN',

'MWMM.WAN,all-Chapman.WAN',
'MWMM.WAN,all-Fletcher.WAN',
'MWMM.WAN,all-Greene.WAN',
'MWMM.WAN,all-Jonson.WAN',
'MWMM.WAN,all-Marlowe.WAN',
'MWMM.WAN,all-Middleton-minus-MWMM.WAN',
'MWMM.WAN,all-Peele.WAN',
'MWMM.WAN,all-Shakespeare.WAN',

'NWNH.WAN,all-Chapman.WAN',
'NWNH.WAN,all-Fletcher.WAN',
'NWNH.WAN,all-Greene.WAN',
'NWNH.WAN,all-Jonson.WAN',
'NWNH.WAN,all-Marlowe.WAN',
'NWNH.WAN,all-Middleton-minus-NWNH.WAN',
'NWNH.WAN,all-Peele.WAN',
'NWNH.WAN,all-Shakespeare.WAN',

'PHOE.WAN,all-Chapman.WAN',
'PHOE.WAN,all-Fletcher.WAN',
'PHOE.WAN,all-Greene.WAN',
'PHOE.WAN,all-Jonson.WAN',
'PHOE.WAN,all-Marlowe.WAN',
'PHOE.WAN,all-Middleton-minus-PHOE.WAN',
'PHOE.WAN,all-Peele.WAN',
'PHOE.WAN,all-Shakespeare.WAN',

'PUR.WAN,all-Chapman.WAN',
'PUR.WAN,all-Fletcher.WAN',
'PUR.WAN,all-Greene.WAN',
'PUR.WAN,all-Jonson.WAN',
'PUR.WAN,all-Marlowe.WAN',
'PUR.WAN,all-Middleton-minus-PUR.WAN',
'PUR.WAN,all-Peele.WAN',
'PUR.WAN,all-Shakespeare.WAN',

'RT.WAN,all-Chapman.WAN',
'RT.WAN,all-Fletcher.WAN',
'RT.WAN,all-Greene.WAN',
'RT.WAN,all-Jonson.WAN',
'RT.WAN,all-Marlowe.WAN',
'RT.WAN,all-Middleton-minus-RT.WAN',
'RT.WAN,all-Peele.WAN',
'RT.WAN,all-Shakespeare.WAN',

'TRI.WAN,all-Chapman.WAN',
'TRI.WAN,all-Fletcher.WAN',
'TRI.WAN,all-Greene.WAN',
'TRI.WAN,all-Jonson.WAN',
'TRI.WAN,all-Marlowe.WAN',
'TRI.WAN,all-Middleton-minus-TRI.WAN',
'TRI.WAN,all-Peele.WAN',
'TRI.WAN,all-Shakespeare.WAN',

'WBW.WAN,all-Chapman.WAN',
'WBW.WAN,all-Fletcher.WAN',
'WBW.WAN,all-Greene.WAN',
'WBW.WAN,all-Jonson.WAN',
'WBW.WAN,all-Marlowe.WAN',
'WBW.WAN,all-Middleton-minus-WBW.WAN',
'WBW.WAN,all-Peele.WAN',
'WBW.WAN,all-Shakespeare.WAN',

'WDO.WAN,all-Chapman.WAN',
'WDO.WAN,all-Fletcher.WAN',
'WDO.WAN,all-Greene.WAN',
'WDO.WAN,all-Jonson.WAN',
'WDO.WAN,all-Marlowe.WAN',
'WDO.WAN,all-Middleton-minus-WDO.WAN',
'WDO.WAN,all-Peele.WAN',
'WDO.WAN,all-Shakespeare.WAN',

'WIT.WAN,all-Chapman.WAN',
'WIT.WAN,all-Fletcher.WAN',
'WIT.WAN,all-Greene.WAN',
'WIT.WAN,all-Jonson.WAN',
'WIT.WAN,all-Marlowe.WAN',
'WIT.WAN,all-Middleton-minus-WIT.WAN',
'WIT.WAN,all-Peele.WAN',
'WIT.WAN,all-Shakespeare.WAN',

'Y5G.WAN,all-Chapman.WAN',
'Y5G.WAN,all-Fletcher.WAN',
'Y5G.WAN,all-Greene.WAN',
'Y5G.WAN,all-Jonson.WAN',
'Y5G.WAN,all-Marlowe.WAN',
'Y5G.WAN,all-Middleton-minus-Y5G.WAN',
'Y5G.WAN,all-Peele.WAN',
'Y5G.WAN,all-Shakespeare.WAN',

'BA.WAN,all-Chapman.WAN',
'BA.WAN,all-Fletcher.WAN',
'BA.WAN,all-Greene.WAN',
'BA.WAN,all-Jonson.WAN',
'BA.WAN,all-Marlowe.WAN',
'BA.WAN,all-Middleton.WAN',
'BA.WAN,all-Peele-minus-BA.WAN',
'BA.WAN,all-Shakespeare.WAN',

'DAV.WAN,all-Chapman.WAN',
'DAV.WAN,all-Fletcher.WAN',
'DAV.WAN,all-Greene.WAN',
'DAV.WAN,all-Jonson.WAN',
'DAV.WAN,all-Marlowe.WAN',
'DAV.WAN,all-Middleton.WAN',
'DAV.WAN,all-Peele-minus-DAV.WAN',
'DAV.WAN,all-Shakespeare.WAN',

'E1.WAN,all-Chapman.WAN',
'E1.WAN,all-Fletcher.WAN',
'E1.WAN,all-Greene.WAN',
'E1.WAN,all-Jonson.WAN',
'E1.WAN,all-Marlowe.WAN',
'E1.WAN,all-Middleton.WAN',
'E1.WAN,all-Peele-minus-E1.WAN',
'E1.WAN,all-Shakespeare.WAN',

'OWT.WAN,all-Chapman.WAN',
'OWT.WAN,all-Fletcher.WAN',
'OWT.WAN,all-Greene.WAN',
'OWT.WAN,all-Jonson.WAN',
'OWT.WAN,all-Marlowe.WAN',
'OWT.WAN,all-Middleton.WAN',
'OWT.WAN,all-Peele-minus-OWT.WAN',
'OWT.WAN,all-Shakespeare.WAN',

'PAR.WAN,all-Chapman.WAN',
'PAR.WAN,all-Fletcher.WAN',
'PAR.WAN,all-Greene.WAN',
'PAR.WAN,all-Jonson.WAN',
'PAR.WAN,all-Marlowe.WAN',
'PAR.WAN,all-Middleton.WAN',
'PAR.WAN,all-Peele-minus-PAR.WAN',
'PAR.WAN,all-Shakespeare.WAN',

'1H4.WAN,all-Chapman.WAN',
'1H4.WAN,all-Fletcher.WAN',
'1H4.WAN,all-Greene.WAN',
'1H4.WAN,all-Jonson.WAN',
'1H4.WAN,all-Marlowe.WAN',
'1H4.WAN,all-Middleton.WAN',
'1H4.WAN,all-Peele.WAN',
'1H4.WAN,all-Shakespeare-minus-1H4.WAN',

'2H4.WAN,all-Chapman.WAN',
'2H4.WAN,all-Fletcher.WAN',
'2H4.WAN,all-Greene.WAN',
'2H4.WAN,all-Jonson.WAN',
'2H4.WAN,all-Marlowe.WAN',
'2H4.WAN,all-Middleton.WAN',
'2H4.WAN,all-Peele.WAN',
'2H4.WAN,all-Shakespeare-minus-2H4.WAN',

'ADO.WAN,all-Chapman.WAN',
'ADO.WAN,all-Fletcher.WAN',
'ADO.WAN,all-Greene.WAN',
'ADO.WAN,all-Jonson.WAN',
'ADO.WAN,all-Marlowe.WAN',
'ADO.WAN,all-Middleton.WAN',
'ADO.WAN,all-Peele.WAN',
'ADO.WAN,all-Shakespeare-minus-ADO.WAN',

'ANT.WAN,all-Chapman.WAN',
'ANT.WAN,all-Fletcher.WAN',
'ANT.WAN,all-Greene.WAN',
'ANT.WAN,all-Jonson.WAN',
'ANT.WAN,all-Marlowe.WAN',
'ANT.WAN,all-Middleton.WAN',
'ANT.WAN,all-Peele.WAN',
'ANT.WAN,all-Shakespeare-minus-ANT.WAN',

'AWW.WAN,all-Chapman.WAN',
'AWW.WAN,all-Fletcher.WAN',
'AWW.WAN,all-Greene.WAN',
'AWW.WAN,all-Jonson.WAN',
'AWW.WAN,all-Marlowe.WAN',
'AWW.WAN,all-Middleton.WAN',
'AWW.WAN,all-Peele.WAN',
'AWW.WAN,all-Shakespeare-minus-AWW.WAN',

'AYLI.WAN,all-Chapman.WAN',
'AYLI.WAN,all-Fletcher.WAN',
'AYLI.WAN,all-Greene.WAN',
'AYLI.WAN,all-Jonson.WAN',
'AYLI.WAN,all-Marlowe.WAN',
'AYLI.WAN,all-Middleton.WAN',
'AYLI.WAN,all-Peele.WAN',
'AYLI.WAN,all-Shakespeare-minus-AYLI.WAN',

'COR.WAN,all-Chapman.WAN',
'COR.WAN,all-Fletcher.WAN',
'COR.WAN,all-Greene.WAN',
'COR.WAN,all-Jonson.WAN',
'COR.WAN,all-Marlowe.WAN',
'COR.WAN,all-Middleton.WAN',
'COR.WAN,all-Peele.WAN',
'COR.WAN,all-Shakespeare-minus-COR.WAN',

'CYM.WAN,all-Chapman.WAN',
'CYM.WAN,all-Fletcher.WAN',
'CYM.WAN,all-Greene.WAN',
'CYM.WAN,all-Jonson.WAN',
'CYM.WAN,all-Marlowe.WAN',
'CYM.WAN,all-Middleton.WAN',
'CYM.WAN,all-Peele.WAN',
'CYM.WAN,all-Shakespeare-minus-CYM.WAN',

'ERR.WAN,all-Chapman.WAN',
'ERR.WAN,all-Fletcher.WAN',
'ERR.WAN,all-Greene.WAN',
'ERR.WAN,all-Jonson.WAN',
'ERR.WAN,all-Marlowe.WAN',
'ERR.WAN,all-Middleton.WAN',
'ERR.WAN,all-Peele.WAN',
'ERR.WAN,all-Shakespeare-minus-ERR.WAN',

'H5.WAN,all-Chapman.WAN',
'H5.WAN,all-Fletcher.WAN',
'H5.WAN,all-Greene.WAN',
'H5.WAN,all-Jonson.WAN',
'H5.WAN,all-Marlowe.WAN',
'H5.WAN,all-Middleton.WAN',
'H5.WAN,all-Peele.WAN',
'H5.WAN,all-Shakespeare-minus-H5.WAN',

'HAM.WAN,all-Chapman.WAN',
'HAM.WAN,all-Fletcher.WAN',
'HAM.WAN,all-Greene.WAN',
'HAM.WAN,all-Jonson.WAN',
'HAM.WAN,all-Marlowe.WAN',
'HAM.WAN,all-Middleton.WAN',
'HAM.WAN,all-Peele.WAN',
'HAM.WAN,all-Shakespeare-minus-HAM.WAN',

'JC.WAN,all-Chapman.WAN',
'JC.WAN,all-Fletcher.WAN',
'JC.WAN,all-Greene.WAN',
'JC.WAN,all-Jonson.WAN',
'JC.WAN,all-Marlowe.WAN',
'JC.WAN,all-Middleton.WAN',
'JC.WAN,all-Peele.WAN',
'JC.WAN,all-Shakespeare-minus-JC.WAN',

'LLL.WAN,all-Chapman.WAN',
'LLL.WAN,all-Fletcher.WAN',
'LLL.WAN,all-Greene.WAN',
'LLL.WAN,all-Jonson.WAN',
'LLL.WAN,all-Marlowe.WAN',
'LLL.WAN,all-Middleton.WAN',
'LLL.WAN,all-Peele.WAN',
'LLL.WAN,all-Shakespeare-minus-LLL.WAN',

'LR.WAN,all-Chapman.WAN',
'LR.WAN,all-Fletcher.WAN',
'LR.WAN,all-Greene.WAN',
'LR.WAN,all-Jonson.WAN',
'LR.WAN,all-Marlowe.WAN',
'LR.WAN,all-Middleton.WAN',
'LR.WAN,all-Peele.WAN',
'LR.WAN,all-Shakespeare-minus-LR.WAN',

'MND.WAN,all-Chapman.WAN',
'MND.WAN,all-Fletcher.WAN',
'MND.WAN,all-Greene.WAN',
'MND.WAN,all-Jonson.WAN',
'MND.WAN,all-Marlowe.WAN',
'MND.WAN,all-Middleton.WAN',
'MND.WAN,all-Peele.WAN',
'MND.WAN,all-Shakespeare-minus-MND.WAN',

'MV.WAN,all-Chapman.WAN',
'MV.WAN,all-Fletcher.WAN',
'MV.WAN,all-Greene.WAN',
'MV.WAN,all-Jonson.WAN',
'MV.WAN,all-Marlowe.WAN',
'MV.WAN,all-Middleton.WAN',
'MV.WAN,all-Peele.WAN',
'MV.WAN,all-Shakespeare-minus-MV.WAN',

'OTH.WAN,all-Chapman.WAN',
'OTH.WAN,all-Fletcher.WAN',
'OTH.WAN,all-Greene.WAN',
'OTH.WAN,all-Jonson.WAN',
'OTH.WAN,all-Marlowe.WAN',
'OTH.WAN,all-Middleton.WAN',
'OTH.WAN,all-Peele.WAN',
'OTH.WAN,all-Shakespeare-minus-OTH.WAN',

'R2.WAN,all-Chapman.WAN',
'R2.WAN,all-Fletcher.WAN',
'R2.WAN,all-Greene.WAN',
'R2.WAN,all-Jonson.WAN',
'R2.WAN,all-Marlowe.WAN',
'R2.WAN,all-Middleton.WAN',
'R2.WAN,all-Peele.WAN',
'R2.WAN,all-Shakespeare-minus-R2.WAN',

'R3.WAN,all-Chapman.WAN',
'R3.WAN,all-Fletcher.WAN',
'R3.WAN,all-Greene.WAN',
'R3.WAN,all-Jonson.WAN',
'R3.WAN,all-Marlowe.WAN',
'R3.WAN,all-Middleton.WAN',
'R3.WAN,all-Peele.WAN',
'R3.WAN,all-Shakespeare-minus-R3.WAN',

'ROM.WAN,all-Chapman.WAN',
'ROM.WAN,all-Fletcher.WAN',
'ROM.WAN,all-Greene.WAN',
'ROM.WAN,all-Jonson.WAN',
'ROM.WAN,all-Marlowe.WAN',
'ROM.WAN,all-Middleton.WAN',
'ROM.WAN,all-Peele.WAN',
'ROM.WAN,all-Shakespeare-minus-ROM.WAN',

'SHR.WAN,all-Chapman.WAN',
'SHR.WAN,all-Fletcher.WAN',
'SHR.WAN,all-Greene.WAN',
'SHR.WAN,all-Jonson.WAN',
'SHR.WAN,all-Marlowe.WAN',
'SHR.WAN,all-Middleton.WAN',
'SHR.WAN,all-Peele.WAN',
'SHR.WAN,all-Shakespeare-minus-SHR.WAN',

'TGV.WAN,all-Chapman.WAN',
'TGV.WAN,all-Fletcher.WAN',
'TGV.WAN,all-Greene.WAN',
'TGV.WAN,all-Jonson.WAN',
'TGV.WAN,all-Marlowe.WAN',
'TGV.WAN,all-Middleton.WAN',
'TGV.WAN,all-Peele.WAN',
'TGV.WAN,all-Shakespeare-minus-TGV.WAN',

'TMP.WAN,all-Chapman.WAN',
'TMP.WAN,all-Fletcher.WAN',
'TMP.WAN,all-Greene.WAN',
'TMP.WAN,all-Jonson.WAN',
'TMP.WAN,all-Marlowe.WAN',
'TMP.WAN,all-Middleton.WAN',
'TMP.WAN,all-Peele.WAN',
'TMP.WAN,all-Shakespeare-minus-TMP.WAN',

'TN.WAN,all-Chapman.WAN',
'TN.WAN,all-Fletcher.WAN',
'TN.WAN,all-Greene.WAN',
'TN.WAN,all-Jonson.WAN',
'TN.WAN,all-Marlowe.WAN',
'TN.WAN,all-Middleton.WAN',
'TN.WAN,all-Peele.WAN',
'TN.WAN,all-Shakespeare-minus-TN.WAN',

'TRO.WAN,all-Chapman.WAN',
'TRO.WAN,all-Fletcher.WAN',
'TRO.WAN,all-Greene.WAN',
'TRO.WAN,all-Jonson.WAN',
'TRO.WAN,all-Marlowe.WAN',
'TRO.WAN,all-Middleton.WAN',
'TRO.WAN,all-Peele.WAN',
'TRO.WAN,all-Shakespeare-minus-TRO.WAN',

'WIV.WAN,all-Chapman.WAN',
'WIV.WAN,all-Fletcher.WAN',
'WIV.WAN,all-Greene.WAN',
'WIV.WAN,all-Jonson.WAN',
'WIV.WAN,all-Marlowe.WAN',
'WIV.WAN,all-Middleton.WAN',
'WIV.WAN,all-Peele.WAN',
'WIV.WAN,all-Shakespeare-minus-WIV.WAN',

'WT.WAN,all-Chapman.WAN',
'WT.WAN,all-Fletcher.WAN',
'WT.WAN,all-Greene.WAN',
'WT.WAN,all-Jonson.WAN',
'WT.WAN,all-Marlowe.WAN',
'WT.WAN,all-Middleton.WAN',
'WT.WAN,all-Peele.WAN',
'WT.WAN,all-Shakespeare-minus-WT.WAN'
]

(indicator, throwaway) = loadWAN(indicatorFileName)

for pair in listOfWANPairs:
    (textFile1, throwAway) = pair.split(",")
    (throwAway, textFile2) = pair.split(",")
    (WAN1, text1counts) = loadWAN(textFile1)
    (WAN2, text2counts) = loadWAN(textFile2)

    # Use CPU version
    # WAN1LimitProbs = limitProbabilities((eliminateSinks(WAN1)),text1counts)

    # Use GPU-accelerated version
    WAN1LimitProbs = limitProbabilities_gpu((eliminateSinks(WAN1)), text1counts)
    print(textFile1+","+textFile2+","+str(round(100 * relativeEntropy(WAN1, WAN2, WAN1LimitProbs, indicator),2)))
