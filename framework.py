import torch
import torch.utils.tensorboard as tensorboard
import random, tqdm, time, os, math

torch.set_default_tensor_type('torch.cuda.DoubleTensor')

class option:
    class data:
        def __init__(self):
            self.data_dir_list:list = None
            self.read_func = None
            self.read_func_arg:list = None
            self.transform_func = None
            self.transform_func_arg:list = None
    class train:
        def __init__(self):
            self.model:torch.Module = None
            self.optimizer_func:torch.optim = None
            self.optimizer_arg:list = None
            self.lossfunc = None
            self.lossfunc_arg:list = None
            self.metrics:dict = None
            self.validate:bool = False
            self.validate_rate:float = 0.0
            self.epoch:int = 0
            self.batch_size:int = 1
            self.model_dir:str = './models'
            self.model_name:str = 'model'
            self.log_dir:str = './logs'
    class predict:
        def __init__(self):
            self.model_path:str = None
            self.predict_dir:str = './predict'
            self.predict_name:str = 'answer'

class data:
    def __init__(self, data_opt:option.data):
        self.data_opt = data_opt
        self.Data_Load()

    def Data_Load(self)->tuple:
        self.data = []
        self.label = []
        for data_dir in self.data_opt.data_dir_list:
            data, label = self.data_opt.read_func(data_dir, *self.data_opt.read_func_arg)
            self.data += data
            self.label += label
        if self.data_opt.transform_func != None:
            self.data = self.data_opt.transform_func(self.data, *self.data_opt.transform_func_arg)
        return self.data, self.label

    def Data_Shuffle(self)->None:
        data = list(zip(self.data, self.label))
        random.shuffle(data)
        self.data[:], self.label[:] = zip(*data)

    def Batch(self, batch_size)->None:
        batch_data = []
        batch_label = []
        for i in range(0, len(self.data), batch_size):
            batch_data.append(self.data[i:i+batch_size])
            batch_label.append(self.label[i:i+batch_size])
        self.data = batch_data
        self.label = batch_label

    def Validation_Create(self, rate)->tuple:
        v_data = []
        v_label = []
        data = list(zip(self.data, self.label))
        num = math.ceil(len(data)*rate)
        if num != len(data):
            validation = data[:num]
            data = data[num:]
            self.data[:], self.label[:] = zip(*data)
            v_data[:], v_label[:] = zip(*validation)
        return v_data, v_label

    def Data_Clean(self)->None:
        self.data = []
        self.label = []

class train(data):
    def __init__(self, data_opt:option.data, train_opt:option.train):
        self.data_opt = data_opt
        self.train_opt = train_opt
        super(train, self).__init__(self.data_opt)
        self.Data_Shuffle()
        self.Batch(self.train_opt.batch_size)
        if self.train_opt.validate:
            self.v_data, self.v_label = self.Validation_Create(self.train_opt.validate_rate)
        self.model = self.train_opt.model.cuda()
        self.optimizer = self.train_opt.optimizer_func(self.model.parameters(), *self.train_opt.optimizer_arg)
        self.lossfunc = self.train_opt.lossfunc(*self.train_opt.lossfunc_arg).cuda()

    def __print_loss(self, loss:float, metrics:dict, start_time:float)->None:
        Info = '  Elapse: {elapse:3.3f}s - Loss: {loss:8.5f}'.format(elapse=(time.time()-start_time), loss=loss)
        for metric_name in metrics.keys():
            Info += ', {name:>8}: {value:<3.3f}'.format(name=metric_name, value=metrics[metric_name])
        print(Info)
        print()

    def __run_epoch(self, flag:str)->tuple:
        if flag == 'Train': 
            self.model.train()
            data = self.data
            label = self.label
        elif flag == 'Validation': 
            self.model.eval()
            data = self.v_data
            label = self.v_label
        y_list = []
        pred_y_list = []
        loss_list = []
        for i in tqdm.tqdm(range(len(data)), '  {flag:10}'.format(flag=flag), ncols=120, unit='Batch'):
            x = torch.tensor(data[i], dtype=torch.double)
            y = torch.tensor(label[i], dtype=torch.int64).reshape((-1, 1))
            self.optimizer.zero_grad()
            pred_y, pred_p = self.model(x)
            pred_y = pred_y.type(torch.int64)
            loss = self.lossfunc(pred_p, y.reshape((-1,)))
            loss.backward()
            self.optimizer.step()
            y_list += y.tolist()
            pred_y_list += pred_y.tolist()
            loss_list.append(loss.item())
        y = torch.tensor(y_list, dtype=torch.int64)
        pred_y = torch.tensor(pred_y_list, dtype=torch.int64)
        metrics_dict = {}
        for metric_name in self.train_opt.metrics.keys():
            metric_func, metric_arg = self.train_opt.metrics[metric_name]
            metric = metric_func(y, pred_y, *metric_arg)
            if isinstance(metric, tuple):
                metric_name_list = metric_name.split(',')
                for i in range(len(metric_name_list)):
                    metrics_dict[metric_name_list[i]] = metric[i]
            else: metrics_dict[metric_name] = metric
        loss = sum(l for l in loss_list)/len(loss_list)
        return loss, metrics_dict

    def __run(self, epoch:int, tb_writer:tensorboard.SummaryWriter, flag:str)->None:
        start = time.time()
        loss, metrics = self.__run_epoch(flag)
        self.__print_loss(loss, metrics, start)
        tb_writer.add_scalar('Loss', loss, epoch)
        for tag in metrics.keys():
            tb_writer.add_scalar(tag, metrics[tag], epoch)

    def Run(self)->None:
        print('--Training Process Running:')
        tb_writer1 = tensorboard.SummaryWriter(log_dir=self.train_opt.log_dir+'/train')
        if self.train_opt.validate: tb_writer2 = tensorboard.SummaryWriter(log_dir=self.train_opt.log_dir+'/validation')
        for epoch in range(self.train_opt.epoch):
            print('[ Epoch:', epoch, ']')
            flag = 'Train'
            self.__run(epoch, tb_writer1, flag)
            if self.train_opt.validate:
                flag = 'Validation'
                self.__run(epoch, tb_writer2, flag)
        self.model.eval()
        example_input = [torch.tensor(self.data[0])]
        #tb_writer1.add_graph(self.model, example_input)
        if not os.path.exists(self.train_opt.model_dir): os.mkdir(self.train_opt.model_dir)
        torch.save(self.model, self.train_opt.model_dir+'/'+self.train_opt.model_name+'.pkl')

class predict(data):
    def __init__(self, data_opt:option.data, predict_opt:option.predict):
        self.data_opt = data_opt
        self.predict_opt = predict_opt
        super(predict, self).__init__(data_opt)
        self.model = torch.load(self.predict_opt.model_path).cuda().eval()

    def Run(self)->None:
        print('--Prediction Process Running ... ...')
        x = torch.split(torch.tensor(self.data, dtype=torch.double), (1000, 9000), 0)[0]
        pred_y, pred_p = self.model(x)
        pred_label = pred_y.reshape((-1,)).type(torch.int64).tolist()
        if not os.path.exists(self.predict_opt.predict_dir): os.mkdir(self.predict_opt.predict_dir)
        with open(self.predict_opt.predict_dir +'/'+self.predict_opt.predict_name+'.txt', 'w') as file:
            for i in range(len(pred_label)):
                file.write("{:<60}{:d}".format(i, pred_label[i])+'\n')
            file.close()