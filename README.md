# Stage_M2 : Emulation of subglacial lake/ocean dynamics with machine learning

## Short-term objectives :

- Becoming familiar with the physical mechanisms controlling the time evolution of the flow fields in the subglacial lake dataset (read: couston-etal-2022 [[1]](#1)) [[2]](#2)
- Creating a PSMN account (go to Contacts->Formulaire du PSMN)and accessing PSMN (setting up ssh keys)
- Becoming familiar with the DMD and SINDy algorithms (read: schmid-2010 & brunton-etal-2016) 
- Applying POD and DMD algorithms to the dataset to identify dominant modes-Comparing POD and DMD modes with modes obtained by thresholding the mean horizontal velocity

## ToDOlist
[] Understand Sindy \
[] Run Sindy algo on data\
[] Write a resume of Couston's paper\
[] Read article symetries\
[] clean code
[] check reconstruction via POD and DMD\
[] write autoencoder
[] check dynamic within the modes
[] run decomposition on HC


# References
#### Physical system : 
<a id="1">[1]</a> 
Couston, Louis-Alexandre, Joseph Nandaha, and Benjamin Favier. “Competition between Rayleigh–Bénard and Horizontal Convection.” Journal of Fluid Mechanics 947 (September 2022): A13. https://doi.org/10.1017/jfm.2022.613.

<a id="2">[2]</a> 
EDP Sciences. “Hydrodynamique Physique - 3e Édition - Étienne Guyon, Jean-Pierre Hulin, Luc Petit (EAN13 : 9782759808939) | EDP Sciences La Boutique : E-Bookstore, Online Sale of Scientific Books and Ebooks.” Accessed February 2, 2024. https://laboutique.edpsciences.fr/produit/595/9782759808939/Hydrodynamique%20physique.

#### POD : 
<a id="3">[3]</a> 
Weiss, Julien. “A Tutorial on the Proper Orthogonal Decomposition.” In AIAA Aviation 2019 Forum. American Institute of Aeronautics and Astronautics. Accessed January 31, 2024. https://doi.org/10.2514/6.2019-3333.

#### DMD : 
<a id="4">[4]</a> 
Schmid, Peter J. “Dynamic Mode Decomposition of Numerical and Experimental Data.” Journal of Fluid Mechanics 656 (August 2010): 5–28. https://doi.org/10.1017/S0022112010001217.


#### Autoencoder :
<a id="5">[5]</a> 
Bank, Dor, Noam Koenigstein, and Raja Giryes. “Autoencoders.” In Machine Learning for Data Science Handbook: Data Mining and Knowledge Discovery Handbook, edited by Lior Rokach, Oded Maimon, and Erez Shmueli, 353–74. Cham: Springer International Publishing, 2023. https://doi.org/10.1007/978-3-031-24628-9_16.
