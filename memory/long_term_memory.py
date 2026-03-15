```json
{
    "memory/long_term_memory.py": {
        "content": "
import logging
from typing import List, Dict
from pydantic import BaseModel
from autogluon import TabularPredictor
from segmentation_models_pytorch import Unet
import torch
import torch.nn as nn
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LongTermMemory(BaseModel):
    """
    Long term memory model for tobacco stemming and redrying optimization.
    """
    non_stationary_drift_index: float
    stochastic_regime_switch: bool

    def __init__(self, non_stationary_drift_index: float, stochastic_regime_switch: bool):
        """
        Initialize long term memory model.

        Args:
        - non_stationary_drift_index (float): Index of non-stationary drift.
        - stochastic_regime_switch (bool): Flag for stochastic regime switch.
        """
        self.non_stationary_drift_index = non_stationary_drift_index
        self.stochastic_regime_switch = stochastic_regime_switch

    def update_memory(self, new_data: List[Dict]):
        """
        Update long term memory with new data.

        Args:
        - new_data (List[Dict]): New data to update memory.

        Returns:
        - None
        """
        try:
            # Update memory using autogluon predictor
            predictor = TabularPredictor()
            predictor.fit(new_data)
            self.non_stationary_drift_index = predictor.get_non_stationary_drift_index()
            logger.info('Updated non-stationary drift index')
        except Exception as e:
            logger.error(f'Error updating memory: {e}')

    def predict(self, input_data: Dict):
        """
        Predict output using long term memory.

        Args:
        - input_data (Dict): Input data for prediction.

        Returns:
        - float: Predicted output.
        """
        try:
            # Use segmentation models pytorch for prediction
            model = Unet('resnet34', encoder_weights='imagenet')
            input_tensor = torch.tensor(input_data['input'])
            output = model(input_tensor)
            return output.item()
        except Exception as e:
            logger.error(f'Error predicting output: {e}')

    def stochastic_regime_switching(self):
        """
        Perform stochastic regime switching.

        Returns:
        - bool: Flag indicating regime switch.
        """
        try:
            # Use numpy for stochastic regime switching
            random_value = np.random.rand()
            if random_value < 0.5:
                self.stochastic_regime_switch = True
            else:
                self.stochastic_regime_switch = False
            logger.info('Performed stochastic regime switching')
            return self.stochastic_regime_switch
        except Exception as e:
            logger.error(f'Error performing stochastic regime switching: {e}')

if __name__ == '__main__':
    # Simulate 'Rocket Science' problem
    long_term_memory = LongTermMemory(non_stationary_drift_index=0.5, stochastic_regime_switch=False)
    new_data = [{'input': [1, 2, 3], 'output': 4}, {'input': [4, 5, 6], 'output': 7}]
    long_term_memory.update_memory(new_data)
    input_data = {'input': [7, 8, 9]}
    predicted_output = long_term_memory.predict(input_data)
    print(f'Predicted output: {predicted_output}')
    regime_switch = long_term_memory.stochastic_regime_switching()
    print(f'Regime switch: {regime_switch}')
",
        "commit_message": "feat: implement specialized long_term_memory logic"
    }
}
```