import pandas as pd
import pickle 

from pgmpy.metrics import SHD
from pgmpy.estimators import BICGauss
from models.ground_truth_model import create_ground_truth

gt_model = create_ground_truth()

with open("models/Simulation/model1_1st_order_trained.pkl","rb") as f:
    model1_DAG = pickle.load(f)

test_data = pd.read_pickle("data/Simulation/GT_test.pkl")

print("GT nodes:")
print(list(gt_model.nodes()))

print("\nLearned nodes:")
print(list(model1_DAG.nodes()))

shd = SHD()
shd_score = shd(gt_model, model1_DAG)

print("SHD: ", shd_score)

bic_ground_truth = BICGauss(test_data).score(gt_model)

bic_learned = BICGauss(test_data).score(model1_DAG)


print("BIC ground truth: ", bic_ground_truth)
print("BIC learned: ", bic_learned)