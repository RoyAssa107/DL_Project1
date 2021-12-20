import configparser

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config_file_path = 'config.txt'
    config.read(filenames=config_file_path)
    print(config['GENERAL']['device'])
    print(config['DATASET']['relative_path'])
    print(config['DATASET']['full_path'])


