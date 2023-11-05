from src.components.data_transformation import DataTransformation
from src.components.data_ingestion import DataIngestion
from src.components.mode_trainer import ModelTrainer

# Run the code

if __name__ == "__main__":
    obj = DataIngestion()
    train_data_path, test_Data_path = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_Data_path)
    model_trainer = ModelTrainer()
    model_trainer.initiate_model_training(train_array=train_arr, test_array=test_arr)