# Face_Attributes
![image](https://user-images.githubusercontent.com/92204945/137964074-b972240c-a533-41e9-8e29-7571b875796c.png)

### This project has the purpose of resolving common biometric problems: recognition, face attributes estimation, light fast inference.
### The project implies generaly 3 different models:
+ Face Reconizer which is also able to predict face attributes by auxiliary heads
+ Lightweight network only yeilding face attributes
+ Outstanding model able to relatively accurate predict human facial expressions

### The structure of the repository is as follows:
+ Age - folder containing training experiments, hypothesis validation, approaches approbation conserning human age
+ Expressions - folder containing approaches comporison, training experiments, hypothesis validation  conserning human facial expressions
+ Gender - folder containing mofdel training for human gender
+ Datasets - folder containing various dataset links and dataloaders for training supply 
+ utils - folder containing tools such as model modifying classes for target attribute heads 

### You also can find 2 .py scripts running webcam inference
