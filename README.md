# U-MLP Based Hybrid-Field Channel Estimation for XL-RIS Assisted Millimeter-Wave MIMO Systems

This work has been submitted for possible publication. We highly respect reproducible research, so we try to provide the simulation codes for our submitted papers.

How to use this simulation code package?

1.Data Generation and Download

In the "data.txt" file, we have provided the download links and descriptions of paired samples for cascaded channel estimation in XL-RIS systems, where the data pre-estimation processing and normalization operations have been completed. 

You can download the dataset and put it in the desired folder. The “inHmix_73_32_512__4users_64pilot.mat” file and “inHmixLS_73_32_512_4users_32pilot.mat” file include the training and validation dataset, in which the LS pre-estimation is used in "inHmixLS_73_32_512_4users_32pilot.mat" file. The “inHmix_73_32_512_test4users_32pilot.mat” file, “inHmix_73_32_512_test4users_64pilot.mat”, and “inHmix_73_32_512_4users_128pilot.mat” file are the testing dataset, whose pilot overhead are set to 32, 64 and 128, respectively.

Remark: In the hybrid-field channel modeling for XL-RIS systems, we refer to the channel modeling scheme in [1] for RIS-aided mmWave Massive MIMO systems (e.g., the path loss model and clustered scatters distribution), in which the far-field communication scenarios is extend to the hybrid-field communication by supplementing the near-field array response and VR cover vector. We are very grateful for the author of following reference paper and the open-source SimRIS Channel Simulator MATLAB package [2].

[1] E. Basar, I. Yildirim, and F. Kilinc, “Indoor and outdoor physical channel modeling and efficient positioning for reconfigurable intelligent surfaces in mmWave bands,” IEEE Trans. Commun., vol. 69, no. 12, pp. 8600-8611, Dec. 2021.

[2] E. Basar, I. Yildirim, “Reconfigurable Intelligent Surfaces for Future Wireless Networks: A Channel Modeling Perspective“, IEEE Wireless Commun., vol. 28, no. 3, pp. 108–114, June 2021.

2.The Training and Testing of U-MLP model

We have provided the model training and test code to reproduce the corresponding results. Specifically, you can run the “main_UMLP.py” file to train the channel estimation network, and then run the “test_UMLP.py” to realize the cascaded channel estimation under different SNR conditions. The detailed network architecture is given in the “model_UMLP.py”.

Notes:
1. In the training stage, the different hyper-parameters setup will result in slight difference for final channel estimation perfromance. According to our training experiences and some carried attempts, the hyper-parameters and network architecture can be further optimized to obtain better channel estimation performance gain, e.g., the dividing ratio between training samples and vadilation samples, the number of kernel, and the training learning rate, batchsize and epochs.
2. Since the limitation of sample space (e.g., the fixed number of channel samples is collected for each user), the inevitable overfitting phenomenon may occur in the network training stage with the increase of epochs

If you have any queries, please don’t hesitate to contact the email jianx@mails.ccnu.edu.cn.

Copyright reserved by the WiCi Lab, Department of Electronics and Information Engineering, Central China Normal University, Wuhan, China.
