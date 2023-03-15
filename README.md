# Cascaded Channel Estimation for XL-RIS aided Hybrid-field communication systems

This work has been submitted for possible publication. We highly respect reproducible research, so we try to provide the simulation codes for our submitted papers.

How to use this simulation code package?

1.Data Generation and Download

We have provided the paired samples in following link, where the LS-based pre-estimation processing and data normalization have been completed.

DOI Link: https://dx.doi.org/10.21227/3c2t-dz81

You can download the dataset and put it in the desired folder. The “LS_64_256R_6users_32pilot.mat” file includes the training and validation dataset, while the “LS_64_256R_test_6users_32pilot” file is used in the test phase.

Remark: In the hybird-field channel modeling for XL-RIS systems, we refer to the channel modeling scheme in [1] for RIS-aided mmWave Massive MIMO systems (e.g., the path loss model and clustered scatters distribution), in which the far-field communication scenarios is extend to the hybrid-field communication by supplementing the near-field array response and VR cover vector. We are very grateful for the author of following reference paper and the open-source SimRIS Channel Simulator MATLAB package [2].

[1] E. Basar, I. Yildirim, and F. Kilinc, “Indoor and outdoor physical channel modeling and efficient positioning for reconfigurable intelligent surfaces in mmWave bands,” IEEE Trans. Commun., vol. 69, no. 12, pp. 8600-8611, Dec. 2021.

[2] E. Basar, I. Yildirim, “Reconfigurable Intelligent Surfaces for Future Wireless Networks: A Channel Modeling Perspective“, IEEE Wireless Commun., vol. 28, no. 3, pp. 108–114, June 2021.

2.The Training and Testing of U-MLP model
1) Network architecture

We have integrated the model training and test code, and you can run the “main.py” file to obtain the channel estimation result of the LPAN or LPAN-L model. The detailed network model is given in the “LPAN.py” and “LPAN-L.py”.

Notes:

(1) Please confirm the required library files have been installed.

(2) Please switch the desired data loading path and network models.

(3) In this work, our goal is to propose a general multi-scale channel estimation network backbone for RIS-aided communication systems. In the model training phase, we did not carefully find the optimal hyper-parameters. Intuitively, hyper-parameters can be further optimized to obtain better channel estimation performance gain, e.g., the training batchisze, epochs, and the depth and width of neural network.

The author in charge of this simulation code pacakge is: Jian Xiao (email: jianx@mails.ccnu.edu.cn). If you have any queries, please don’t hesitate to contact me.

Copyright reserved by the WiCi Lab, Department of Electronics and Information Engineering, Central China Normal University, Wuhan 430079, China.
