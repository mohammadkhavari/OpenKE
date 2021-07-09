import openke
from openke.config import Trainer, Tester
from openke.module.model import DistMult
from openke.module.loss import SoftplusLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader

# NOTE: we use pythorch dataloader that is written in python instead of TrainDataLoader written in cpp
from openke.data.PyTorchTrainDataLoader import PyTorchTrainDataLoader


# dataloader for training
train_dataloader = PyTorchTrainDataLoader(
	in_path = "./benchmarks/WN18RR/", 
	nbatches = 100,
	threads = 8, 
	sampling_mode = "normal", 
	bern_flag = 1, 
	filter_flag = 1, 
	neg_ent = 25,
	neg_rel = 0
)

# dataloader for test
test_dataloader = TestDataLoader("./benchmarks/WN18RR/", "link")

# define the model
distmult = DistMult(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = 200
)

# define the loss function
model = NegativeSampling(
	model = distmult, 
	loss = SoftplusLoss(),
	batch_size = train_dataloader.get_batch_size(), 
	regul_rate = 1.0
)


# train the model
# NOTE: epochs are too short in order to have faster develope and test they should refactor to normal values
trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 20, alpha = 0.5, use_gpu = True, opt_method = "adagrad")
trainer.run()
distmult.save_checkpoint('./checkpoint/distmult.ckpt')

# test the model
distmult.load_checkpoint('./checkpoint/distmult.ckpt')
tester = Tester(model = distmult, data_loader = test_dataloader, use_gpu = True)
tester.run_link_prediction(type_constrain = False)
