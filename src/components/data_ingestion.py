import os
import sys
from src.exception import CustomException

from src.logger import logging

import pandas as pd
from src.components.data_transformation import Preprocessor
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import model_trainer
from src.components.model_trainer import model_trainer_config
from src.utils import save_object, evalute_model
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    SourceDataPath: str = ('notebook\data\stud.csv')
    RawDataPath : str = os.path.join("artifacts",'raw.csv')
    TrainDataPath : str = os.path.join("artifacts",'train.csv')
    TestDataPath : str = os.path.join("artifacts",'test.csv')   
    

class DataIngestion:
    def __init__ (self):
     self.IngestionConfig = DataIngestionConfig() ## DataIngestionConfig() calls the automatically generated __init__  by @dataclass method and creates an instance of the class.

    def InitiateDataIngestion(self):

        logging.info("data ingestion started")

        try:
            df = pd.read_csv(self.IngestionConfig.SourceDataPath)
            logging.info('succsefully read the data from its source into pandas')
        
            TrainSet,TestSet = train_test_split( df , test_size = 0.2 , random_state=42)
            logging.info('succsefully preformed train test split')

            df.to_csv(self.IngestionConfig.RawDataPath, index= False , header = True)
            logging.info('succsefully exported raw data to artifacts folder')

            TrainSet.to_csv(self.IngestionConfig.TrainDataPath , index=False , header = True)
            logging.info('succsefully exported train set to artifacts folder')

            TestSet.to_csv(self.IngestionConfig.TestDataPath, index=False , header = True)
            logging.info('succsefully exported test set to artifacts folder')

        
            return (
            self.IngestionConfig.TrainDataPath,
            self.IngestionConfig.TestDataPath,
            
            )
            
            logging.info('ingestion complete !')
        except Exception as e:
            raise CustomException (e,sys)


if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.InitiateDataIngestion()

    data_transformation=Preprocessor()
    train_arr,test_arr,_=data_transformation.ApplyPreprocessing(train_data,test_data)

    modeltrainer=model_trainer()
    print(modeltrainer.intiate_model_trainer(train_arr,test_arr))
