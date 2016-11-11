# Combiom: an algorithm to search for combinatorial biomarkers

## Implementation

The algorithm to search for combinatorial biomarkers is incorporated in `Combiom. searching for combinatorial biomarkers.ipynb` notebook and available as a separate `combiom.py` file. An example of data analysis is available in `Combiom. Statistical analysis, cross validation and plotting.ipynb`.

## Usage

Combiom can be imported by:
```python
import combiom as cb
```

Firstly, load some data and convert it to numpy arrays. Then call `init_iterators()` and `search()` functions:

```python
# Initializing with 5 parameters
iters = cb.init_iterators(5)

# Searching for combinatorial biomarkers
results = cb.search(bio_marker_names, bio_marker_data, bio_target_names.size, bio_target_data, bio_target_names, iters)
```

## Results

Resulting [Pandas](http://pandas.pydata.org/) DataFrames are stored as *pickle* files. Excel files were created with Pandas and [XlsxWriter](http://xlsxwriter.readthedocs.io/). See Jupyter notebooks for details.

