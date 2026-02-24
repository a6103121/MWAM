# Official MWAM in PyTorch

Here is the official PyTorch implementation of MWAM.

### MWAM is a plug-and-play module. Due to the varying structures of different projects, we will only demonstrate the process of embedding MWAM into MMANet here.

The project is being modified on the official MMANet code repository. The link to MMANet is: [MMANet](https://github.com/shicaiwei123/MMANet-CVPR2023)


## The process of embedding MWAM on MMANet is as follows:

 - Copy our frequency domain information extraction module to the project directory ```./src/freq_n.py```
 - Instantiate this class in the existing model in line 558 of ```./models/surf_baseline.py```
 - Define the FRM bank for each modality, and define the weight allocation function and FRM bank update function within the model class in ```./models/surf_baseline.py```.
```
FRM bank is defined in lines 12-14.
Two new functions are defined in lines 597-666.
```
 - Integrating these computational processes into the forward derivation process, as detailed in lines 561–571 of ```./models/surf_baseline.py```.
 - Add weights from different modalities to the output of the model's forward inference process in the line 595 of ```./models/surf_baseline.py```.
 - In the training function, utilize weights for gradient editing or loss weighting. Relevant modifications are located in lines 1166, 1214-1220  of the ```./lib/model_develop.py```

For other training and inference procedures, refer to [MMANet](https://github.com/shicaiwei123/MMANet-CVPR2023).


## Acknowledgments
[MMANet](https://github.com/shicaiwei123/MMANet-CVPR2023)'s official code

## If this project has been helpful to you, please cite the article below.
```
a
```
