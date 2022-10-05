# Cursive Style Transfer
### Files:
* classifier.py: classifier that can classify input data either into cursive/standard, or by the script type.
* config.py: global settings for classifier and CycleGAN
* cyclegan.py: train CycleGAN
* imageloader.py: load data and preprocess for classifier, and build tf datasets for CycleGAN
* layers.py: layers for CycleGAN models
* models.py: discriminator, generator, and CycleGAN models
* utils.py: functions for classifier accuracy, CycleGAN preprocessing and saving images.

For more information, read the report at: https://docs.google.com/document/d/1P6rplTh_iS6wiTAIlJl_AcRzP0edkj6eciUMn__P8Ls/edit?usp=sharing
Note: it seems my report is permanently lost with no backup I can find. In general, the results were that cyclegan was able to quite effectively convert the print latin script to a cursive one but was unsuccessful in other respects.
