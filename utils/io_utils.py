import os
import json
import torch
import importlib.util


def load_config(path):
    with open(path,'r') as file:
        config=json.load(file)
        #config['SYSTEM']=system
        config['device']=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config['base_path']=get_base_path(config['SYSTEM'])
        #config['max_filters']=-1
        return fix_bools(config)

def fix_bools(config):
    for key,value in config.items():
        if value == "False" or value == "True":
            config[key]=str_to_bool(value)
    return config
def str_to_bool(s):
    if s == 'False':
        return False
    if s == 'True':
        return True
    else:
        print('warning: this is not a true/false string')
        return s

def get_base_path(system):
    if system == 'cluster':
        base_path = '/cluster/projects/uludag/uzair/'
    elif system == 'local':
        base_path = '/home/uzair/nvme/'
    elif system == 'graham':
        base_path = '/home/u2hussai/projects/def-uludagk/u2hussai/'
    elif system == 'cedar':
        base_path = '/home/u2hussai/projects_u/data/'
    return base_path

def model_save_path(config,roi=None):
    out=[]
    keys_exclude=['SYSTEM','device','base_path','models_path']
    for key in config.keys():
        if key not in keys_exclude:
            out.append(str(key)+'-'+str(config[key]))
    if roi is not None:
        out.append('ROI-'+str(roi))
    out_path=config['base_path'] + '/trainedEncoders/'
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    return out_path + '_'.join(out) + '_trained.pt'

def sysargs(argg,backbone,UPTO,epochs,max_filters,train_backbone):
    buff = argg[0]
    argg = [buff]

    argg.append(backbone)
    argg.append(UPTO)
    argg.append(epochs)
    argg.append(max_filters)
    argg.append(train_backbone)
    return argg

def change_freesurfer_subjects(path):
    var_name="SUBJECTS_DIR"
    curr_val=os.environ.get(var_name)
    new_val=path
    os.environ[var_name]=new_val


def get_function_names(file_path):
    # Load the module from the file
    spec = importlib.util.spec_from_file_location("module_name", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Get all names in the module and filter out functions
    function_names = [name for name in dir(module) if callable(getattr(module, name))]

    return function_names