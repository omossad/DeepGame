import time
import os
import qp_264 as q_script
#import qp as q_script
#base_path='C:\\Users\\omossad\\Desktop\\dataset\\encoding\\'
base_path='D:\\Encoding\\gamingAnywhere\\fifa\\'

qp=[22,27,32,37]
K=[[7,6,5,4]]
#length=304
length=197
size=0
width=[1920]
height=[1080]
cnt=0
for x in range(1):
    p=base_path;
    #q_script.encode(p,width[cnt],height[cnt])
    cnt_in=0
    for q in qp:
        size=int(os.path.getsize(p+'QP\\enc_'+str(q)+'.mkv')*8/length)
        print(size)
	#   os.system("START Encoder.exe "+p+" "+str(size)+" 0 1 "+str(K[cnt][cnt_in])+" 0");
	#   os.system("START Encoder.exe "+p+" "+str(size)+" 4 1 0 0");	   	  
	#   os.system("START Encoder.exe "+p+" "+str(size)+" 2 1 0 0");
	#   time.sleep(160)
	#   cnt_in=cnt_in+1
	#time.sleep(160)
	#cnt=cnt+1
