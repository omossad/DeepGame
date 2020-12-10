import csv
import numpy as np
import configparser
import math
from shapely.geometry import Polygon


config = configparser.ConfigParser()
config.read(['C:\\Users\\omossad\\Desktop\\codes\\ROI-PyTorch\\DeepGame\\config.ini'])

def get_visual_pixels():
	return int(config.get("preprocessing", "radius"))

def get_num_tiles():
	return int(config.get("preprocessing", "num_tiles"))

def get_intersection_threshold():
	return float(config.get("preprocessing", "intersection_threshold"))

def get_img_dim():
	W = float(config.get("data", "W"))
	H = float(config.get("data", "H"))
	return [W,H]

def get_fps():
	return float(config.get("data", "fps"))



def get_bitrate():
	return float(config.get("data", "bitrate"))

def get_encoder_mb():
	mb_w = int(config.get("encoder", "mb_width"))
	mb_h = int(config.get("encoder", "mb_height"))
	return [mb_w, mb_h]

def get_encoder_params():
	K = int(config.get("encoder", "K"))
	CU_SIZE = int(config.get("encoder", "CU_size"))
	return [K, CU_SIZE]

def get_encoder_consts():
	ALPHA=float(config.get("encoder", "ALPHA"))
	BETA=float(config.get("encoder", "BETA"))
	MIN_LAMBDA =float(config.get("encoder", "MIN_LAMBDA"))
	MAX_LAMBDA =float(config.get("encoder", "MAX_LAMBDA"))
	C1=float(config.get("encoder", "C1"))
	C2=float(config.get("encoder", "C2"))
	QP_BASE=float(config.get("encoder", "QP_BASE"))
	return [ALPHA, BETA, MIN_LAMBDA, MAX_LAMBDA, C1, C2, QP_BASE]

def get_encoder_block_size():
	gt_w = float(config.get("encoder", "gt_w"))
	gt_h = float(config.get("encoder", "gt_w"))
	return [gt_w, gt_h]

def get_model_conf():
	ts = int(config.get("model", "input_frames"))
	t_overlap = int(config.get("model", "sample_overlap"))
	fut = int(config.get("model", "pred_frames"))
	return [ts, t_overlap, fut]



def get_no_files(game):
	num_files = 0
	with open('..\\frames_info_' + game + '.csv', 'r') as f:
		for line in f:
			num_files += 1
		# number of files is the number of files to be processed #
		num_files = num_files - 1
		print("Total number of files is:", num_files)
	return num_files

def get_files_list(num_files, game):
	file_names = []
	with open('..\\frames_info_' + game + '.csv') as csv_file:
			csv_reader = csv.reader(csv_file, delimiter=',')
			line_count = 0
			for row in csv_reader:
				if line_count == 0:
					line_count += 1
				elif line_count < num_files+1:
					file_names.append(row[0])
					line_count += 1
				else:
					break
			print('Files read in order are')
			print(file_names)
	return file_names

def circleRectangleIntersectionArea(r, xcenter, ycenter, xleft, xright, ybottom, ytop):
#find the signed (negative out) normalized distance from the circle center to each of the infinitely extended rectangle edge lines,
	d = [0, 0, 0, 0]
	d[0]=(xcenter-xleft)/r
	d[1]=(ycenter-ybottom)/r
	d[2]=(xright-xcenter)/r
	d[3]=(ytop-ycenter)/r
	#for convenience order 0,1,2,3 around the edge.

	# To begin, area is full circle
	area = math.pi*r*r

	# Check if circle is completely outside rectangle, or a full circle
	full = True
	for d_i in d:
		if d_i <= -1:   #Corresponds to a circle completely out of bounds
			return 0
		if d_i < 1:	 #Corresponds to the circular segment out of bounds
			full = False

	if full:
		return area

	# this leave only one remaining fully outside case: circle center in an external quadrant, and distance to corner greater than circle radius:
	#for each adjacent i,j
	adj_quads = [1,2,3,0]
	for i in [0,1,2,3]:
		j=adj_quads[i]
		if d[i] <= 0 and d[j] <= 0 and d[i]*d[i]+d[j]*d[j] > 1:
			return 0

	# now begin with full circle area  and subtract any areas in the four external half planes
	a = [0, 0, 0, 0]
	for d_i in d:
		if d_i > -1 and d_i < 1:
			a[i] = math.asin( d_i )  #save a_i for next step
			area -= 0.5*r*r*(math.pi - 2*a[i] - math.sin(2*a[i]))

	# At this point note we have double counted areas in the four external quadrants, so add back in:
	#for each adjacent i,j

	for i in [0,1,2,3]:
		j=adj_quads[i]
		if  d[i] < 1 and d[j] < 1 and d[i]*d[i]+d[j]*d[j] < 1 :
			# The formula for the area of a circle contained in a plane quadrant is readily derived as the sum of a circular segment, two right triangles and a rectangle.
			area += 0.25*r*r*(math.pi - 2*a[i] - 2*a[j] - math.sin(2*a[i]) - math.sin(2*a[j]) + 4*math.sin(a[i])*math.sin(a[j]))

	return area

def fixation_to_tile(x,y):
	n_tiles = get_num_tiles()
	#X = x*W
	#Y = y*H
	#tile_width  = W/num_tiles
	#tile_height = H/num_tiles
	X = min(n_tiles - 1, x * n_tiles)
	Y = min(n_tiles - 1, y * n_tiles)
	return [int(X), int(Y)]


def object_to_tile_intersection(x1,y1,x2,y2):
	#print('Object coor : ' + str(x1) + ' ' + str(x2) + ' ' + str(y1) + ' ' + str(y2))
	#print(x1)
	#print(y1)
	#print(x2)
	#print(y2)
	[W,H] = get_img_dim()
	n_tiles = get_num_tiles()
	arr_x = np.zeros((n_tiles))
	arr_y = np.zeros((n_tiles))
	tile_w = W/n_tiles
	tile_h = H/n_tiles
	object_poly = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
	for i in range(n_tiles):
		#print(i*tile_w)
		#print(i*tile_h)
		#print((i+1)*tile_w)
		#print((i+1)*tile_h)
		#tile_poly = Polygon([(i*tile_w, i*tile_h), ((i+1)*tile_w, i*tile_h), ((i+1)*tile_w, (i+1)*tile_h), (i*tile_w, (i+1)*tile_h)])
		tile_poly_x = Polygon([(i*tile_w, y1), ((i+1)*tile_w, y1), ((i+1)*tile_w, y2), (i*tile_w, y2)])
		tile_poly_y = Polygon([(x1, i*tile_h), (x2, i*tile_h), (x2, (i+1)*tile_h), (x1, (i+1)*tile_h)])
		#print('poly coor : ' + str(i*tile_w) + ' ' + str((i+1)*tile_w) + ' ' + str(y1) + ' ' + str(y2))
		intersection = object_poly.intersection(tile_poly_x)
		arr_x[i] = intersection.area/(tile_w*(y2-y1))
		intersection = object_poly.intersection(tile_poly_y)
		arr_y[i] = intersection.area/(tile_h*(x2-x1))
		#print('Object intersection with tile ' + str(i) + ' : ' + str(arr_x[i]) + ' ' + str(arr_y[i]))
		#print(intersection.area/(W))
	return [arr_x, arr_y]
	#for i in range(n_tiles):


def fixation_to_tile_intersection(x,y):
	n_tiles = get_num_tiles()
	radius = get_visual_pixels()
	#X = x*W
	#Y = y*H
	#tile_width  = W/num_tiles
	#tile_height = H/num_tiles
	X = min(n_tiles - 1, x * n_tiles)
	Y = min(n_tiles - 1, y * n_tiles)
	return [int(X), int(Y)]

def create_ROI_arr(rois, flag):
	[W,H] = get_img_dim()
	[mb_w, mb_h] = get_encoder_mb()
	[K, CU_size] = get_encoder_params()
	arr = np.zeros((mb_h, mb_w))
	if flag == 'gt':
		[w,h] = get_encoder_block_size()
	else:
		n_tiles = get_num_tiles()
		w = 1/n_tiles
		h = 1/n_tiles
	for roi in rois:
		start_x = math.floor(roi[0]*W/CU_size)
		start_y = math.floor(roi[1]*H/CU_size)
		end_x = math.ceil(start_x + w*W/CU_size)
		end_y = math.ceil(start_y + h*H/CU_size)
		arr[start_y:end_y,start_x:end_x] = 1
	return arr

def create_weights_arr(ROI_arr, rois):
	[K, CU_size] = get_encoder_params()
	[W,H] = get_img_dim()
	[w,h] = get_encoder_block_size()
	diagonal = math.sqrt(W**2 + H**2)
	hd = ROI_arr.shape[0]
	wd = ROI_arr.shape[1]
	arr = np.zeros((hd,wd))
	for i in range(hd):
		for j in range(wd):
			if ROI_arr[i][j] > 0:
				e = 1/(K*ROI_arr[i][j])
				arr[i][j] =  math.exp(-1.0*e)
			else:
				y_pix = i*CU_size + CU_size/2
				x_pix = j*CU_size + CU_size/2
				sumDist = 0
				for roi in rois:
					mid_x_pix = (roi[0] + w/2)*W
					mid_y_pix = (roi[1] + h/2)*H
					dist = max(1.0, math.sqrt((mid_y_pix-y_pix)**2 + (mid_x_pix-x_pix)**2))
					e = (K*dist/ diagonal)
					sumDist = sumDist +  math.exp(-1.0*e)
				arr[i][j] = sumDist / max(len(rois),1)

	arr = arr / arr.sum()
	return arr

def CLIP(min_, max_, value):
	return max(min_, min(max_, value))

def clipLambda(lambda_, MIN_LAMBDA, MAX_LAMBDA):
	if math.isnan(lambda_):
		lambda_ = MAX_LAMBDA
	else:
		lambda_ = CLIP(MIN_LAMBDA, MAX_LAMBDA, lambda_)

def roundRC(d):
	if math.isnan(d):
		d = 34.0
	return int(d + 0.5)

def LAMBDA_TO_QP(lambda_, C1, C2):
	return int(CLIP(0.0, 51.0, roundRC(C1 * math.log(lambda_) + C2)))

def QPToBits(QP, ALPHA, BETA, C1, C2):
	lambda_ = math.exp((QP - C2) / C1)
	bpp = ((lambda_/ALPHA)*1.0) ** (1.0 / BETA)
	return bpp * (64.0**2.0)

def calcQPLambda(weights, roi_array):
	bitrate = get_bitrate()
	fps = get_fps()
	[K, CU_size] = get_encoder_params()
	[W,H] = get_img_dim()
	hd = roi_array.shape[0]
	wd = roi_array.shape[1]
	[ALPHA, BETA, MIN_LAMBDA, MAX_LAMBDA, C1, C2, QP_BASE] = get_encoder_consts()
	pixelsPerBlock = CU_size*CU_size

	roi_array = roi_array.flatten()
	weights = weights.flatten()
	bitsPerBlock = np.zeros((hd*wd))
	QP = np.zeros((hd*wd))
	avg_bits_per_pic = bitrate/fps
	targetbppFrame = avg_bits_per_pic / (W * H)
	FRAME_LAMBDA = ALPHA * (targetbppFrame**BETA)
	FRAME_LAMBDA = clipLambda(FRAME_LAMBDA, MIN_LAMBDA, MAX_LAMBDA)
	bitsAlloc = avg_bits_per_pic
	qp_frame = LAMBDA_TO_QP(ALPHA*(bitsAlloc / (W*H))**BETA, C1, C2)
	qp_delta = 0.0
	extra = 0.0
	sumSaliencyImportantAreas = 0.0
	sumSaliency = 0.0
	reassigned = 0
	avgLambda = 0.0
	block_ind=0
	avg_qp = 0
	sum_non_roi = 0
	for i in range(block_ind, hd*wd):
		sumSaliency = sumSaliency + weights[block_ind]
		assigned = max(1.0, 1.0 / (1.0*wd*hd) * bitsAlloc)
		assignedW = weights[block_ind] * bitsAlloc
		targetbpp=0.0
		targetbpp = assignedW / pixelsPerBlock
		if roi_array[block_ind] == 0:
			sum_non_roi += 1
		lambdaConst = ALPHA * (targetbpp**BETA)
		avgLambda = avgLambda + math.log(lambdaConst)
		temp = LAMBDA_TO_QP(lambdaConst, C1, C2)
		qp_delta = qp_delta + temp - qp_frame
		QP[block_ind] = temp -QP_BASE
		avg_qp = avg_qp + QP[block_ind]+QP_BASE
		bitsPerBlock[block_ind] = min(QPToBits(int(QP[block_ind]), ALPHA, BETA, C1, C2),assigned)
		block_ind += 1

	if qp_delta != 0:
		while qp_delta < 0:
			for block_ind in range(0, hd*wd and qp_delta<0):
				if roi_array[block_ind] == 0:
					QP[block_ind] = min(51-QP_BASE, QP[block_ind] + 1)
					qp_delta = qp_delta + 1
	avgLambda = math.exp(avgLambda / (hd*wd))
	old_avg_bits_per_pic = avg_bits_per_pic

	return QP/QP.sum()

def tile_to_raw(x,y):
	#[W,H] = get_img_dim()
	n_tiles = get_num_tiles()
	tile_w = 1/n_tiles
	tile_h = 1/n_tiles
	raw_top_x = x * tile_w
	raw_top_y = y * tile_h
	return [raw_top_x, raw_top_y]
