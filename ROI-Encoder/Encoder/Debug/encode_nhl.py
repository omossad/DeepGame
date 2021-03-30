import time
import os
import qp as q_script
#base_path='C:\\Users\\omossad\\Desktop\\dataset\\encoding\\'
base_path='D:\\Encoding\\encoding_files\\nhl\\'

qp=[22,27,32,37]
K=[[7,6,5,4]]
#length=304
length=99
size=0
width=[1280]
height=[720]
cnt=0
for x in range(1):
	p=base_path+'ga'+str(x)+'\\';
	#q_script.encode(p,width[cnt],height[cnt])
	cnt_in=0
	for q in qp:
	   size=int(os.path.getsize(p+'QP\\enc_'+str(q)+'.mp4')*8/length);
	   os.system("START Encoder.exe "+p+" "+str(size)+" 0 1 "+str(K[cnt][cnt_in])+" 0");
	#   os.system("START Encoder.exe "+p+" "+str(size)+" 4 1 0 0");	   	  
	#   os.system("START Encoder.exe "+p+" "+str(size)+" 2 1 0 0");
	#   time.sleep(160)
	#   cnt_in=cnt_in+1
	#time.sleep(160)
	cnt=cnt+1
