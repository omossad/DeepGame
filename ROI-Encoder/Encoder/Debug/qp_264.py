import os
def encode(p,width,height):
    qp=[22,27,32,37]
    fps = 29.97
    for q in qp:		
        os.system("x264.exe --input-res "+str(width)+"x"+str(height)+ " --preset ultrafast --tune zerolatency --qp "+ str(q)+" --me dia --merange 16 --fps "+ str(fps)+" --keyint 90 --ref 1 --bframes 0 --ipratio 1.059 --intra-refresh -o " +p+"QP\enc_"+str(q)+".mkv " +p+"raw_"+str(width)+"_"+str(height)+".yuv");

