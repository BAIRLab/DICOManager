<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Thanks again! Now go create something AMAZING! :D
***
***
***
*** To avoid retyping too much info. Do a search and replace for the following:
*** github_username, repo_name, twitter_handle, email, project_title, project_description
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![BSD-3-Clause License][license-shield]][license-url]
[![3.8][python-shield]][python-url]



<!-- PROJECT LOGO -->
<br />
<p align="center">
  <h3 align="center">DICOManager</h3>

  <p align="center">
    A DICOM Sorting and Processing Toolkit written in Python
    <br />
    <a href="https://github.com/BAIRLab/DICOManager/"><strong>Explore the docs »</strong></a>
    <br />
    <a href="https://github.com/BAIRLab/DICOManager/blob/dev/example.py">View Demo</a>
    ·
    <a href="https://github.com/BAIRLab/DICOManager/issues">Report Bug</a>
    ·
    <a href="https://github.com/BAIRLab/DICOManager/issues">Request Feature</a>
  </p>
</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project
DICOManager is designed around building a DICOM file tree, from which multi-threaded reconstruction, processing and saving can be conducted. For systems with limited memory, file trees contain pointers to disk allowing reconstruction to be run on as little as 8 GiB memory. For systems with limited write access, all operations can be conducted in memory.

### Built With

* [anytree](https://anytree.readthedocs.io/en/latest/)
* [pydicom](https://pydicom.github.io)
* [Numpy](https://numpy.org)
* [scipy.ndimage](https://docs.scipy.org/doc/scipy/reference/ndimage.html)
* [xarray](http://xarray.pydata.org/en/stable/)


<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

This package requires python version 3.8 or greater.
### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/BAIRLab/DICOManager.git
   ```
2. Install PIP packages
   ```sh
   python -m pip install -r requirements.txt
   ```

<!-- USAGE EXAMPLES -->
## Usage

DICOManager is designed around building DICOM trees and conducting multithreaded operations on the trees. The basic tree design has the following heirachy:
1. Cohort
2. PatientID
3. FrameOfReferenceUID
4. StudyInstanceUID
5. SeriesInstanceUID
6. Modality (contains individual dicom and reconstructed files)

The file tree can then be saved, reconstructed into volumes, filtered, sorted or manipulated.

**Building a tree**
```
from DICOManager import groupings
from glob import glob

files = glob('/path/to/unsorted/files/**/*.dcm', recursive=True)
project = groupings.Cohort(files=files, name='Example')
print(project)
```

**Saving DICOM tree**
```
project.save_tree('/path/to/save/tree')
```

**Reconstructing DICOMs**
<br />
To reduce memory usage, the default behavior for reconstruction is to write the volumes to disk and only store the paths to the volumes in memory.
```
project.recon(path='/path/to/save/pointers')
```

For more examples, please refer to the [jupyter notebook](https://github.com/BAIRLab/DICOManager/blob/dev/tutorial.ipynb) or the example script [example.py](https://github.com/BAIRLab/DICOManager/blob/dev/example.py).

<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/BAIRLab/DICOManager/issues) for a list of proposed features (and known issues).

The remaining tasks are in the process of being completed:
* Support for data generators for pytorch / tensorflow
* Reading saved directory trees without re-sorting
* Checking loaded tree for validity
* Updating deconstruction for new data structures
* Support for saving as NIFTI format or x-array
* Improving documentation and creation of wiki
* Formatting for pip install

<!-- CONTRIBUTING -->
## Contributing

Any contributions are **greatly appreciated**. Please submit a pull request as follows:

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


<!-- LICENSE -->
## License

Distributed under the BSD-3-Clause License. See `LICENSE` for more information.


<!-- CONTACT -->
## Contact

Evan Porter - evan.porter(at)beaumont.org

<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

* [NEMA DICOM Standard](http://dicom.nema.org/medical/dicom/current/output/html/part01.html)
* [anytree](https://anytree.readthedocs.io/en/latest/)
* [pydicom](https://pydicom.github.io)
* [Numpy](https://numpy.org)
* [scipy.ndimage](https://docs.scipy.org/doc/scipy/reference/ndimage.html)
* [xarray](http://xarray.pydata.org/en/stable/)
* [scikit-image](https://scikit-image.org/docs/stable/api/skimage.html)
* [README design template](https://github.com/othneildrew/Best-README-Template)


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/BAIRLab/DICOManager.svg?style=for-the-badge
[contributors-url]: https://github.com/BAIRLab/DICOManager/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/BAIRLab/DICOManager.svg?style=for-the-badge
[forks-url]: https://github.com/BAIRLab/DICOManager/network/members
[stars-shield]: https://img.shields.io/github/stars/BAIRLab/DICOManager.svg?style=for-the-badge
[stars-url]: https://github.com/BAIRLab/DICOManager/stargazers
[issues-shield]: https://img.shields.io/github/issues/BAIRLab/DICOManager.svg?style=for-the-badge
[issues-url]: https://github.com/BAIRLab/DICOManager/issues
[license-shield]: https://img.shields.io/github/license/BAIRLab/DICOManager.svg?style=for-the-badge
[license-url]: https://github.com/BAIRLab/DICOManager/blob/dev/LICENSE.txt
[python-shield]: https://img.shields.io/badge/python-3.8-blue.svg?style=for-the-badge
[python-url]: https://www.python.org/downloads/release/python-360/
