# Workflow

The Python scripts form a **pipeline** that must be run manually in sequence, with each script's output serving as input to the next script in the workflow.

## Main Scripts (in execution order)

1. **`word-freq-by-counts.py`** - First script to run
   - **Purpose**: Finds the top 100 most frequent words across all concatenated authorial canons
   - **Input**: `all.txt` (concatenated texts from all authors)
   - **Output**: Word frequency list (which becomes `top-100-words.txt`)
   - **Called by**: None (run manually first)

2. **`makeWINDOWS.py`** - Second script to run
   - **Purpose**: Divides each play into rolling windows of 2000 words, advancing by 500 words
   - **Input**: Individual play text files (e.g., `Shakespeare-HAM.txt`)
   - **Output**: Window files with `.win` extension (e.g., `Shakespeare-HAM-01.win`, `Shakespeare-HAM-02.win`, etc.)
   - **Called by**: None (run manually second)

3. **`makeWANnoprint.py`** - Third script to run
   - **Purpose**: Creates WAN (Word Adjacency Network) matrices for all rolling windows and complete canons
   - **Input**: 
     - Window files (`.win` files from `makeWINDOWS.py`)
     - `top-100-words.txt` (from `word-freq-by-counts.py`)
   - **Output**: WAN files with `.WAN` extension
   - **Called by**: None (run manually third)

4. **`makeINDICATOR.py`** - Fourth script to run
   - **Purpose**: Creates an indicator matrix showing which word transitions occur in ALL authorial canons
   - **Input**: Complete canon WAN files (e.g., `Chapman.WAN`, `Shakespeare.WAN`, etc.)
   - **Output**: Indicator file with `.IND` extension (e.g., `8-authors-2k-0.5k-top-100-words.IND`)
   - **Called by**: None (run manually fourth)

5. **`compareWANSnoprint.py`** - Fifth (final) script to run
   - **Purpose**: Compares each rolling window with authorial canons using relative entropy
   - **Input**: 
     - WAN files from `makeWANnoprint.py`
     - Indicator file from `makeINDICATOR.py`
   - **Output**: CSV with format `text1, text2, H(text1,text2)` showing relative entropy values
   - **Called by**: None (run manually last)

6. **`makeBLOCKLIST.py`** - Utility script
   - **Purpose**: Generates the list of WAN pairs for `compareWANSnoprint.py` (currently empty)
   - **Note**: According to `experimental-steps.txt`, you can manually create the list of `.win` files out of `makeWANnoprint.py` to get blocks
   - **Called by**: None (utility for generating the `listOfWANPairs` variable)

## Current State of the Codebase

At the moment, the following files are missing (in order to run all the Python scripts, not just the final one `compareWANSnoprint.py`):

- Concatenated texts from all authors (for example, `all.txt`) used as input to generate word frequency list (such as `top-100-words.txt`)
- Individual play text files (e.g., `Shakespeare-HAM.txt`)

These input files will enable running the 3 first scripts which will generate all intermediate files as input for each next script in the pipeline.

## Performance pre-assessment

The most likely performance bottlenecks in the pipeline are:

1. **`makeWANnoprint.py`** - Creating thousands of WAN matrices for all rolling windows (3000-4000 `.win` files from around 100 plays + 8 author canons + about 100 canon-minus-play files in the working example) and writing each of them into new `.WAN` files. Likely, file I/O overhead.

2. **`compareWANSnoprint.py`** - Has a very large `listOfWANPairs` containing approximately 1,000+ pairs of WANs to compare (each play's window compared against 8 authorial canons) and performs relative entropy calculations with 100 matrix multiplications in the `limitProbabilities()` function. Likely, computationally expensive operations per pair due to expensive matrix multiplication in `limitProbabilities()`.

### GPU porting pre-assessment

Based on the bottleneck analysis:

**`compareWANSnoprint.py` - Strong candidate for GPU acceleration:**
- The [`limitProbabilities()`](compareWANSnoprint.py) function performs 100 matrix multiplications (raising a 100x100 matrix to the 100th power) for each WAN pair
- With ~1,000+ WAN pairs to process, this represents significant computational work
- Matrix multiplication is highly parallelizable and well-suited for GPU computation
- **Recommendation**: Port [`limitProbabilities()`](compareWANSnoprint.py) and [`relativeEntropy()`](compareWANSnoprint.py) to use GPU libraries (maybe, **CuPy**?!)
- Expected speedup: Significant, especially for large-scale comparisons

**`makeWANnoprint.py` - Poor candidate for GPU acceleration:**. The primary bottleneck is file I/O, but GPU acceleration doesn't help with disk operations.

**Implementation priority**: Focus GPU porting efforts on [`compareWANSnoprint.py`](compareWANSnoprint.py) first, as it will provide the most significant performance improvement.