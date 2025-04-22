# CS879Project
CS879 Reusable Learning Objective


## Context
**COMPSCI 446 (Search Engines):** is an undergraduate level course on information retrieval at UMass, where students get to learn about different retrieval models and how the design decisions made in constructing those retrieval models affect ranking.

**Our Project:** We plan to create a interactive Jupyter notebook that allows students to visualize key topics from the course.

## Goals
WHAT skills will our RLO promote?
- Visual learning.
- Interactive learning.
- Immersive experience.
HOW will it promote it?
- Offer to students to prime for assignments and exams.
- Demonstration in class.
- It may be part of the third major programming assignment


## Accessibility
In consideration of accessiblity, we are considering color blindness in our models. 

- Colorblind-friendly palettes: Use high-contrast, colorblind-friendly color schemes (e.g., Viridis, Color Universal Design).
- Alternative visual encodings: Use patterns, shapes, or text labels in addition to color to convey information.

## Technical Stuff

## Usage

### Creating Data

To download and process the required data please run `python msmarco.py`, which will create `data/data.json`, which contains all of the necessary data.  The data will then be stored in `data/data.json`, a JSON file containing the following keys:
- `queries: dict[str, list[str]]`  
  Mapping each query ID to its raw tokenized text (list of words)

- `queries_stopped: dict[str, list[str]]`  
  Mapping each query ID to its tokenized text after stopword removal

- `queries_stemmed: dict[str, list[str]]`  
  Mapping each query ID to its tokenized text after stemming

- `queries_stopped_stemmed: dict[str, list[str]]`  
  Mapping each query ID to its tokenized text after stopword removal and stemming

- `documents: dict[str, list[str]]`  
  Mapping each document ID to its raw tokenized text (list of words)

- `documents_stopped: dict[str, list[str]]`  
  Mapping each document ID to its tokenized text after stopword removal

- `documents_stemmed: dict[str, list[str]]`  
  Mapping each document ID to its tokenized text after stemming

- `documents_stopped_stemmed: dict[str, list[str]]`  
  Mapping each document ID to its tokenized text after stopword removal and stemming

- `qrels: dict[str, dict[str, int]]`  
  Mapping each query ID to a dictonary mapping document IDs to their revelance scores.  Not complete (assume missing values are 0)
### Project Hierarchy

- `data/` is where processed data is stored
- `indexes/` is a cache directory for storing built bm25s indexes
- `notebooks/` contains miscellaneous `.ipynb` notebooks to test and develop this project
- `msmarco.py` contains the code to download, process and save the necessary data for this project.
- `retrieve.py` contains implementations of retrieval models

### Requirements
- Python 3.12
- ir_datasets
- numpy
- PyStemmer
- bm25s
- nltk
- Pytrec_eval
