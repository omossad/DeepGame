import os
import utils
from DeepLearning.preprocessing import preprocess
from Analysis.analysis import analyze
data_path = utils.get_path()

if __name__ == "__main__":
	#os.makedirs(os.path.join(data_path, 'model_data'), exist_ok=True)
	#preprocess()
	os.makedirs(os.path.join(data_path, 'analysis'), exist_ok=True)
	analyze()
