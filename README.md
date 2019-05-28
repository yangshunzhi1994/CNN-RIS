# Expression-Clasification
EdgeNet:Improved CNN for Edge Computing  Devices and Its Applications on Facial Expression  Clasification

Datasets and trained models can be obtained in the following two ways：

1.Google Cloud Disk：https://drive.google.com/drive/folders/1WHebQdZrACVms0_3RDGxTTTSyuzfR48C?usp=sharing
2.Baidu cloud disk：

Please put fer2013_data.h5, RAF_data.h5, CK_data.h5 in the data folder downloaded from the above link into the ./data path.Then, put the RAF_EdgeNet and FER2013_EdgeNet folders under the models folder downloaded from the above link into ./models.

The data folder contains the following files: 
        
        ./data: 
        
                --RAF_data.h5
                
                --RAF.py
                
                --fer2013_data.h5
                
                --fer.py
                
                --CK_data.h5
                
                --CK.py
                
                
The models folder contains the following files: 

        ./models: 
                --RAF_EdgeNet folder  
                
                  --PrivateTest_model.t7
                  
                  
                --FER2013_EdgeNet
                
                  --PrivateTest_model.t7
                  
                --EdgeNet.py
                
                --vgg.py
                
                --resnet.py
                
                --__init__.py
        
