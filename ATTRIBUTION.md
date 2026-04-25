# ATTRIBUTION

## Dataset

bigP3BCI Dataset
- Used for all model training and evaluation in this project.
- Mainsah, Boyla, et al. "bigP3BCI: An Open, Diverse and Machine Learning Ready P300-based Brain-Computer Interface Dataset" (version 1.0.0). PhysioNet (2025). RRID:SCR_007345. https://doi.org/10.13026/0byy-ry86
- License: Publicly available for research use.

## Reference Papers

Farwell LA, Donchin E. Talking off the top of your head: toward a mental prosthesis utilizing event-related brain potentials. *Electroencephalogr Clin Neurophysiol.* 1988 Dec;70(6):510-23. doi: 10.1016/0013-4694(88)90149-6. PMID: 2461285.

This foundational paper introduced the P300-based BCI speller paradigm that this project is built upon. It informed our understanding of the P300 component, the row/column flashing stimulus design, and the approach to decoding target characters from EEG signals. It also gave the idea to implement a Linear Discriminant Analysis model.

Lawhern, V. J.; Solon, A. J.; Waytowich, N. R.; Gordon, S. M.; Hung, C. P.; and Lance, B. J. 2018. EEGNet: A compact convolutional neural network for EEG-based brain-computer interfaces. *Journal of Neural Engineering*, 15(5): 056013.

This paper informed the design of our custom CNN architecture (P300NN) for detecting P300 signals in the EEG data. 

## AI Assistance

AI tools were used in the following limited capacities:

- Line autocompletion — used during code writing to suggest completions for standard code patterns and library calls.
- Visualization generation — used to assist in generating and formatting ERP plots and analysis visualizations in the error analysis and evaluation notebooks.

All core model architectures, preprocessing logic, experimental design, and analysis were designed and implemented by the project authors.