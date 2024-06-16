import logging
import logging.config
from pathlib import Path
from utils.utility import read_json

def setup_logging(
     log_config='/data/nelkazwi/code/UniTVelo-ATAC/UniTVelo-ATAC/unitvelo/config/config_main_bonemarrow.json', ##change the path here
    default_level=logging.INFO
):
    log_config = Path(log_config)

    if log_config.is_file():
        config = read_json(log_config)

       

        logging.config.dictConfig(config)
        
        return config
    else:
        print ("Warning: logging configuration file is not found in {}.".format(log_config))
        logging.basicConfig(level=default_level)
    
