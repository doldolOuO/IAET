class params:
    def __init__(self): 
        #General Parameters
        self.cat = "cabinet"
        # plane cabinet car chair lamp couch table watercraft all
        
        #Training parameters
        self.batch_size = 64
        self.nThreads = 8
        self.lr = 0.001
        self.milestones = [20, 40, 60, 80, 100, 120, 140, 160, 180]
        self.data_root = "/data/FCH/data/ShapeNetViPC_2048"
        self.n_epochs = 200
        self.eval_epoch = 1
        self.resume = False
        self.ckpt = "./ckpt_vipc/all.pt"