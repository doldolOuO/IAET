class params:
    def __init__(self):
        #Training parameters
        self.batch_size = 64
        self.nThreads = 8
        self.lr = 0.001
        self.milestones = [20, 40, 60, 80]
        self.data_root = "/data/FCH/data/3D-EPN"
        self.n_epochs = 100