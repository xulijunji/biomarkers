# Combiom: an algorithm to search for combinatorial biomarkers

## Implementation

The algorithm to search for combinatorial biomarkers is incorporated in `Combiom. searching for combinatorial biomarkers.ipynb` notebook and available as a separate `combiom.py` file. An example of data analysis is available in `Combiom. Statistical analysis, cross validation and plotting.ipynb`.

## Usage

Combiom can be imported by:
```python
import combiom as cb
```

Then instantiate Combiom class:

```python
# 5 parameters, 1 observation and 10 participants
cbm = cb.Combiom(5, 1, 10)
```

Now, we must load some data:

```python
# Shape of bio_marker_names: (5, )
# Shape of bio_marker_data: (5, 10)
cmb.load_marker_data(bio_marker_names, bio_marker_data)

# Shape of bio_target_names: (6, )
# Shape of bio_target_names: (6, 10)
# The second shape dimension of bio_target_names must be equal to 
# the second dimension of bio_marker_data
cmb.load_target_data(bio_target_names, bio_target_data)
```

We begin with calling `search()` method:

```python
results = cmb.search('all')
```

## Results

Resulting [Pandas](http://pandas.pydata.org/) DataFrames are stored as *pickle* files. Excel files were created with Pandas and [XlsxWriter](http://xlsxwriter.readthedocs.io/). See Jupyter notebooks for details.

