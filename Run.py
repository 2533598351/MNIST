import framework, model, utils

def Run(module:model.torch.nn.Module, module_arg:list, model_name:str, train:bool=True, predict:bool=True)->None:
    opt_train_data = framework.option.data()
    opt_train_data.data_dir_list = ['./data'
                                   ]
    opt_train_data.read_func = utils.utils.Data_Loader
    opt_train_data.read_func_arg = ['Train']

    opt_train = framework.option.train()
    opt_train.validate = False
    opt_train.validate_rate = 0
    opt_train.epoch = 40
    opt_train.batch_size = 256
    opt_train.model = module(*module_arg)
    opt_train.optimizer_func = model.torch.optim.Adadelta
    opt_train.optimizer_arg = []
    opt_train.lossfunc = model.torch.nn.NLLLoss
    opt_train.lossfunc_arg = []
    opt_train.metrics = {
                        }
    opt_train.model_dir = './models'
    opt_train.log_dir = './logs/'+model_name+'_log'
    opt_train.model_name = model_name

    opt_predict_data = framework.option.data()
    opt_predict_data.data_dir_list = ['./data'
                                     ]
    opt_predict_data.read_func = utils.utils.Data_Loader
    opt_predict_data.read_func_arg = ['Test']

    opt_predict = framework.option.predict()
    opt_predict.model_path = './models/'+model_name+'.pkl'
    opt_predict.predict_dir = './predict'
    opt_predict.predict_name = model_name+'_answer'

    if train: framework.train(opt_train_data, opt_train).Run()
    if predict: framework.predict(opt_predict_data, opt_predict).Run()

Run(model.models.CNN, [(256, 28, 28), 64], 'MNIST', False, True)
