# Deep Learning-Based ROI Recommendation Tool for Brain Slice Imaging

## Overview

This repository contains the source code for a deep learning-based imaging assistant tool designed to automatically recommend regions of interest (ROIs) in mouse brain slice images.

The tool was developed to assist brain imaging experiments in which researchers need to identify specific brain regions based on a reference brain atlas. 

Conventional ROI selection relies heavily on manual identification by researchers, which may introduce variability depending on the user's experience.

To improve the consistency and efficiency of ROI selection, this tool integrates DeepSlice-based brain atlas registration with image processing and feature tracking. 

The program automatically aligns a brain slice image with a reference atlas, identifies the target brain region, and recommends an appropriate imaging area.

The recommended ROI can then be tracked while the microscope stage is moved, allowing the researcher to navigate toward the target region before acquiring high-magnification images.


## Workflow

The imaging assistant follows the workflow below:

1. Capture an image from the microscope.
2. Define the image scale using a known scale bar.
3. Crop the image to the brain section.
4. Remove the image background.
5. Register the brain section to the reference brain atlas using DeepSlice.
6. Identify the target brain region from the registered atlas.
7. Calculate and display the recommended ROI.
8. Track the ROI using ORB feature matching when the microscope stage is moved.
9. Adjust the displayed ROI according to microscope magnification.

<img width="1146" height="1361" alt="image" src="https://github.com/user-attachments/assets/d6648da3-0325-43e7-a219-6be0b9b05480" />

## Main Features

### Image Preprocessing

The program captures a frame from the microscope and allows the user to define the image scale and crop the brain section.
The background of the cropped brain slice image is removed before atlas registration. This preprocessing step improves the registration performance of DeepSlice.

### Brain Atlas Registration

Brain slice registration is performed using DeepSlice, a deep learning-based tool for automatically registering mouse brain histological images to a three-dimensional reference atlas.
The program uses the `DSModel` class and its `predict` method to estimate the spatial alignment of the brain slice.
The resulting alignment information is used to map the input image to atlas coordinates.

### Automatic ROI Recommendation

After atlas registration, the corresponding atlas image is overlaid onto the brain slice image.
The target brain region is identified from the segmented atlas, and its coordinates are used to calculate a recommended imaging ROI.
The recommended ROI is displayed as a bounding box on the microscope image.
The color of the box indicates whether the target region can remain within the microscope field of view after switching to 10X magnification:

- Blue box: the ROI remains within the expected 10X field of view.
- Red box: the ROI would fall outside the expected 10X field of view.

### ROI Tracking

When the microscope stage is moved, the position of the recommended ROI is updated using ORB (Oriented FAST and Rotated BRIEF) feature detection and matching.
Feature matching between consecutive images is used to estimate image displacement and move the recommended ROI accordingly.

### Magnification Support

The program supports visualization of the expected imaging area at:

- 1.25X
- 4X
- 10X

The ROI is scaled relative to the center of the microscope image to approximate the change in field of view produced by switching objective lenses.


## Performance

The performance of the ROI recommendation algorithm was evaluated by comparing automatically recommended ROIs with ROIs manually selected by an experienced researcher.
The mean Intersection over Union (IoU) between the expert-defined and automatically recommended ROIs was approximately **0.7**, indicating substantial spatial agreement between the two methods.


## Requirements

The program is implemented in Python and primarily uses:

- DeepSlice
- OpenCV
- rembg
- NumPy

Additional dependencies may be required depending on the DeepSlice installation and execution environment.


## Reference

This tool was developed based on the DeepSlice framework:
Carey, H., Pegios, M., Martin, L., et al. (2023). DeepSlice: Rapid fully automatic registration of mouse brain imaging to a volumetric atlas. *Nature Communications, 14*, 5884. https://doi.org/10.1038/s41467-023-41645-4

The atlas registration approach used by DeepSlice is related to QuickNII:
Puchades, M. A., Csucs, G., Ledergerber, D., Leergaard, T. B., & Bjaalie, J. G. (2019). Spatial registration of serial microscopic brain images to three-dimensional reference atlases with the QuickNII tool. *PLOS ONE, 14*(5), e0216796. https://doi.org/10.1371/journal.pone.0216796


## License

Will be Uploaded later.


## Contact

For questions, please send an e-mail to me at minhyo331@gmail.com or create an Issue on GitHub.
