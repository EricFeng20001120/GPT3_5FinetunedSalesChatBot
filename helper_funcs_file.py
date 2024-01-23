import os, shutil

import status

# look up the target directory, and return all folders in a list
def get_folders(target_directory):
    if not os.path.exists(target_directory):
        raise FileNotFoundError("Directory {} does not exist!".format(target_directory))
    
    folders = [name for name in os.listdir(target_directory) if os.path.isdir(os.path.join(target_directory, name))]
    folders.sort()
    return folders

def get_files(target_directory, database):
    target_directory = os.path.join(target_directory, database) 
    
    if not os.path.exists(target_directory):
        raise FileNotFoundError("Directory {} does not exist!".format(target_directory))
    
    files = [name for name in os.listdir(target_directory) if not os.path.isdir(os.path.join(target_directory, name))]
    files.sort()
    return files

def create_folder(target_directory, folder_name):
    if not os.path.exists(target_directory):
        raise FileNotFoundError("Directory {} does not exist!".format(target_directory))
    
    folders = get_folders(target_directory)
    if folder_name in folders:
        return "Database exist!"
    
    path = os.path.join(target_directory, folder_name) 
    os.mkdir(path)
    return 

def remove_folder(target_directory, folder_name):
    if not os.path.exists(target_directory):
        raise FileNotFoundError("Directory {} does not exist!".format(target_directory))
    
    folders = get_folders(target_directory)
    if folder_name not in folders:
        return "Database doesnot exist!"
    
    path = os.path.join(target_directory, folder_name) 
    shutil.rmtree(path)
    return

def remove_file(target_directory, database, file_list):
    target_directory = os.path.join(target_directory, database)
    if not os.path.exists(target_directory):
        raise FileNotFoundError("Directory {} does not exist!".format(target_directory))
    
    for file in file_list:
        os.remove(os.path.join(target_directory, file))
        
    return 

def save_file(target_directory, database, files):
    target_directory = os.path.join(target_directory, database)
    if not os.path.exists(target_directory):
        raise FileNotFoundError("Directory {} does not exist!".format(target_directory))
    
    ignored_list = []
    for file in files:
        postfix = '.' + os.path.basename(file.name).split('.')[-1]
        if postfix in status.supported_file_types:
            target_file_path = os.path.join(target_directory, os.path.basename(file.name))
            shutil.copyfile(file.name, target_file_path)
        else:
            ignored_list.append(os.path.basename(file.name))
    if len(ignored_list) == 0:
        return ""
    else:
        return "Files {} are ignored due to file type not supported!".format(ignored_list)