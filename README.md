# Expression-Clasification
Making Accurate Object Detection at the Edge: Review and New Progress

Datasets and AntCNN-G's trained models can be obtained in the following two ways：

1.Google Cloud Disk：https://drive.google.com/drive/folders/1WHebQdZrACVms0_3RDGxTTTSyuzfR48C?usp=sharing
2.链接：https://pan.baidu.com/s/1gai-6yjIL-QmLdNTDXUUnQ 
提取码：dug8

Please put fer2013_data.h5, RAF_data.h5, CK_data.h5 in the data folder downloaded from the above link into the ./data path.Then, put the RAF_AntCNN and FER2013_AntCNN folders under the models folder downloaded from the above link into ./models.

That is, the data folder contains the following files: 
        
        ./data:         
                --RAF_data.h5          
                --RAF.py             
                --fer2013_data.h5             
                --fer.py
                --CK_data.h5         
                --CK.py
                
                
The models folder contains the following files: 

        ./models: 
                --RAF_AntCNN              folder  
                  --PrivateTest_model.t7  
                --FER2013_AntCNN          folder  
                  --PrivateTest_model.t7
                --AntCNN.py
                --AntGCNN.py
                --__init__.py
        
Experimental environment：

        python 3.6
        pytorch 0.4.0
       
Running on the Raspberry Pi 3B+：
        
        pytorch2onn.py： Pytorch cannot be directly converted to an IR file. 
                Therefore, you need to convert the pytorch model to an onxx file using the pytorch2onn.py file.
                Finally, you can convert the onnx file to an IR file on ubuntu.
        
        pi_demo.py： Run on the Raspberry Pi 3B+ using an IR file.
In addition, the modified path of the file is required.

First, we use the dlib library to capture faces. The  implementation of this part can be viewed in our other project  implementation：https://github.com/tobysunx/face_recognition

