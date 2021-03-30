import time
import os
import qp as q_script
base_path='C:\\Users\\omossad\\Desktop\\dataset\\encoding\\'
qp=[22,27,32,37]
K=[[7,6,5,4]]
length=67
size=0
i=0
count=5
width=[1920]
height=[1080]
cnt=0
for x in range(15,16):
	p=base_path+'ga'+str(x)+'\\';
	#q_script.encode(p,width[cnt],height[cnt])
	cnt_in=0
	#for q in qp:
	#size=int(os.path.getsize(p+'QP\\enc_'+str(q)+'.mp4')*8/length);
	size=1957544
	os.system("START Encoder.exe "+p+" "+str(size)+" 0 1 "+str(K[cnt][cnt_in])+" 0");
	time.sleep(160)
	#cnt_in=cnt_in+1
	#time.sleep(160)
	cnt=cnt+1
