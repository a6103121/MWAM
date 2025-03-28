
_Figure2_. Schematic diagrams of the modal weight allocation module. **(a):** Main structure of MWAM. **(b):** FRM bank for handling exception inputs and its update rule is shown in Eq. 1. **(c):** Simple example of embedding MWAM in an underlying model. The calculation rules of FRM follow Eq. 3, which requires flipping and aligning the high-frequency components.
![图片2](https://github.com/user-attachments/assets/4e0a44e7-2228-4c8e-8408-6938575c076d)




_Figure3_. Schematic of training intervention mechanisms. **(a):** Using weights to perform gradient editing on the encoders of different modalities independently changes the optimization paths of various components. **(b):** Using auxiliary heads to derive losses for each modality and integrating these losses through weighted fusion enables a global adjustment of the model's training trajectory.
![图片1](https://github.com/user-attachments/assets/e3f20af7-811d-431c-84cf-6841d387c5e7)


