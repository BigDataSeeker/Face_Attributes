import os
importos.path.join as join

#Creating annotation for gender val-phase dataset
os.chdir('/storage_labs/3030/BelyakovM/Face_attributes/ds/db/gender/All-Age-Faces Dataset/aglined faces/val ')
for currdir,dirnames,files in os.walk('.'):
    for file in files:
        if (file.split('.')[-1]!='txt'):
            gender = file.split('_')[1]
            os.chdir('..')
            text_file = open('annotation_val.txt','a')
            text_file.write(file + ',' + gender + '\n')
            os.chdir('/storage_labs/3030/BelyakovM/Face_attributes/ds/db/gender/All-Age-Faces Dataset/aglined faces/val ')
            #os.replace(file,os,path.join(gender,file))
            print(file," ",gender)
os.chdir('..')