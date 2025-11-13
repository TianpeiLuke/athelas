import sys

config_path = "/opt/ml/processing/input/config"
sys.path.append(config_path + "/scripts/")

from preprocess_functions import chunk_processing

if __name__ == "__main__":
    ### Input ########################################
    training_data_path = "/opt/ml/processing/input/training"
    validation_data_path = "/opt/ml/processing/input/validation"
    calibration_data_path = "/opt/ml/processing/input/calibration"

    """
    ##############################################
    ############# Training data ###########################
    ##############################################
    
    chunk_processing(training_data_path, 'train', 20)
    
    """

    """
    ##############################################
    ############# Calibration data ###########################
    ##############################################
    
    chunk_processing(calibration_data_path, 'cali', 5)
    
    """

    #     """
    ##############################################
    ############# Validation data ###########################
    ##############################################
    chunk_processing(validation_data_path, "vali", 20)

    #     """

    print("Finished preprocessing")
