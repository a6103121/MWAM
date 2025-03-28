
_Figure2_. Schematic diagrams of the modal weight allocation module. **(a):** Main structure of MWAM. **(b):** FRM bank for handling exception inputs and its update rule is shown in Eq. 1. **(c):** Simple example of embedding MWAM in an underlying model. The calculation rules of FRM follow Eq. 3, which requires flipping and aligning the high-frequency components.
![f2](https://github.com/user-attachments/assets/10074e07-f262-4610-b5f6-8bc1d696b514)



_Figure3_. Schematic of training intervention mechanisms. **(a):** Using weights to perform gradient editing on the encoders of different modalities independently changes the optimization paths of various components. **(b):** Using auxiliary heads to derive losses for each modality and integrating these losses through weighted fusion enables a global adjustment of the model's training trajectory.
![f3](https://github.com/user-attachments/assets/8c9121de-85d2-4359-b0dc-e48e03db37bf)

