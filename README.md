SfM ToAEval
==============================

**Trade-off Aware Evaluation of Feature Extraction Algorithms in Structure from Motion**

SfM ToAEval is a framework for evaluating feature extraction algorithms in 3D reconstruction of objects and scenes from image sequences using Structure from Motion (SfM) pipeline. SfM ToAEval allows the user to automatically evaluate different combinations of feature detectors and feature descriptors on a given image sequence. In addition, SfM ToAEval does not require having a ground truth. Moreover, SfM ToAEval is aware of the trade-off between the reconstruction density and accuracy, and it allows the user to visualize it in order to transparently decide the best reconstruction. The complete source code of SfM ToAEval as well as a [Jupyter Notebook](./notebooks/demo.ipynb) demonstrating how to use it are released under the MIT license.

The first stage in SfM pipeline is extracting features in the images and matching the corresponding features in pairs of images. The 2D positions of the corresponding features in the images are used in subsequent stages for estimating the position of the 3D points in the sparse point cloud. Accordingly, deciding which feature detector and feature descriptor to use directly affects the accuracy of the 3D reconstruction as feature detector is responsible for the accuracy of the 2D position of image features while feature descriptor is responsible for the accuracy of matching the detected features. Therefore, finding good combinations of feature detectors and feature descriptors has been a hot area of research during the past decade. Considering the large number of published feature extraction algorithms (for example, there are about 30 feature extraction algorithms implemented in OpenCV) and the growing interest in evaluating different combinations of feature detectors and feature descriptors in SfM motivated the development of SfM ToAEval.

Requirements
------------

SfM ToAEval relies on OpenCV for the first stage of the SfM pipeline (feature extraction as well as feature matching) and COLMAP for the rest of the SfM pipeline. Therefore, OpenCV 4.6.0 compiled with OPENCV_ENABLE_NONFREE flag and COLMAP 3.8 compiled with the default parameters are required. Other requirements can be found in [requirements.txt](./requirements.txt).

USAGE
------------

This is a minimal example showing how to use different functions of SfM ToAEval.

```python
dataset = Dataset(dataset_path, K, 'jpg')
dataset.load()
database = Database(database_path)
dataset.evaluate(database, databases_path, point_clouds_path,
    ['SIFT', 'SURF', 'FAST'],
    ['SIFT', 'SURF', 'DAISY'])
database.close()
analyzer = Analyzer(database_path, point_clouds_path, analysis_database_path)
analyzer.analyze()
visualizer = Visualizer(analysis_database_path)
fig = visualizer.visualize_density_accuracy()
display(fig)
fig = visualizer.visualize_trade_off()
display(fig)
```

Project Organization
------------

    ├── data
    │   ├── databases
    │   │   ├── analysis.sqlite
    │   │   ├── main.sqlite
    │   │   ├── SEQ1_DETECT1_DESC1.sqlite
    │   │   ├── SEQ1_...
    │   │   └── ...
    │   ├── point_clouds
    │   │   ├── SEQ1_DETECT1_DESC1
    │   │   │   ├── 0
    │   │   │   │   ├── cameras.bin
    │   │   │   │   ├── images.bin
    │   │   │   │   ├── points3D.bin
    │   │   │   │   └── project.ini
    │   │   │   └── 0.ply
    │   │   ├── SEQ1_...
    │   │   │   └── ...
    │   │   └── ...
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   │   ├── SEQ1
    │   │   │   └── images
    │   │   │       ├── *.jpg|png|tif|...
    │   │   │       └── ...
    │   │   ├── SEQ2
    │   │   │   └── ...
    │   │   └── ...
    │   └── raw            <- The original, immutable data dump.
    │       └── ...
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │   ├── commands.rst
    │   ├── conf.py
    │   ├── getting-started.rst
    │   ├── index.rst
    │   ├── make.bat
    │   └── Makefile
    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │   │                     the creator's initials, and a short `-` delimited description, e.g.
    │   │                     `1.0-jqp-initial-data-exploration`.
    │   └── demo.ipynb
    ├── README.md          <- The top-level README for developers using this project.
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │       ├── ERD.svg
    │       ├── radar_chart.svg
    │       └── size-error_curves.svg
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   ├── analysis
    │   │   ├── __init__.py
    │   │   └── analyzer.py
    │   ├── colmap
    │   │   ├── __init__.py
    │   │   ├── database.py
    │   │   └── read_write_model.py
    │   ├── data           <- Scripts to download or generate data
    │   │   ├── __init__.py
    │   │   ├── database.py
    │   │   └── dataset.py
    │   ├── experiment.py
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   ├── __init__.py
    │   │   ├── extractor.py
    │   │   ├── matcher.py
    │   │   └── view.py
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       ├── __init__.py
    │       ├── radar_chart.py
    │       └── visualizer.py
    ├── test_environment.py
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
