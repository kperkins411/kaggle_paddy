'''
use this function to setup a Paperspace Instance
'''

def setup_env():
    # How to download competition data to temp folder(data) 
    # unzip it there, then symlink it like its a subdir
    # NOTE: make sure kaggle.json is in /root/.kaggle/
    import os
    from os.path import exists
    import subprocess
    
    #put all data here, it is removed whenever instance terminated
    data_loc='/root/data/'
    
    #if /root/data exists then this script has been run already so bail
    if exists(data_loc):
        print('Script already ran...bailing')
        return
    
    #remove original symlink from this directory
    if exists('./data'):os.remove('./data')

    #remove old setup file
    if exists('./setup.sh'):os.remove('./setup.sh')

    #create temp holder
    subprocess.Popen(['mkdir',data_loc]).wait()

    #symlink it
    subprocess.Popen('ln -s /root/data ./data',shell=True).wait()

    #download competition data to temp data folder
    subprocess.Popen('cd ./data;kaggle competitions download -c paddy-disease-classification',shell=True).wait()

    #unzip it, -q is silent
    subprocess.Popen('cd ./data;unzip -q paddy-disease-classification.zip',shell=True).wait()

    #setup dotfiles   
    print('Setup dotfiles...')
    subprocess.Popen('wget "https://raw.githubusercontent.com/kperkins411/dotfiles/master/setup.sh";',shell=True).wait()

    #make executable, then run
    print('Running setup.sh...')
    subprocess.Popen('chmod 700 setup.sh; ./setup.sh',shell=True).wait()
   
    #just in case its wide open
    print('Changing key permissions...')
    os.system('chmod 600 /root/.kaggle/kaggle.json')

    __setup_packages()  
        
def __setup_packages():
    '''
    install needed libraries for this project
    '''
    print('Setup python packages...')
    import os
    try: from path import Path
    except ModuleNotFoundError:
        os.system('pip install path --quiet')
        from path import Path
    try: import timm
    except ModuleNotFoundError:
        os.system('pip install timm --quiet')
        import timm
    try: import optuna
    except ModuleNotFoundError:
        os.system('pip install optuna --quiet')
        import optuna    