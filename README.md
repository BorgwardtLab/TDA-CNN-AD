# TDA-CNN-AD
Model combining topological descriptors with patch based MR imaging features

This is a work in progress repositroy by F. Hensel and S. Brueningk according to the initial description in https://arxiv.org/abs/2011.06531. This code will be further imporved and hence may sightly deviate from the original description. 

In out analysis we used T1-weighted MR images for AD and CN subjects from the Alzheimer's Disease Neuroimaging Initiative (http://adni.loni.usc.edu). Data was preprocessed as described in the archive article using the \textit{fmriprep} pipeline. For the creation of persisitence images the following packages were used: 

The function run.py contains the code to run the image-patch-based 3D-CNN, the TDA 2D-CNN, and a combined model using both topoligical descriptors and a 3D image patch. The information for all 216 patched can be combined in a logistic regression model (ensemble model 1), whereas the preclassification layer encodings of the TDA 2D-CNN and single patch 3D-CNN can used as features for a single dense layer (ensemble model 2). 




