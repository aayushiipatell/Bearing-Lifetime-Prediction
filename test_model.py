from utils.load_model import load_trained_model
from utils.predictor import predict_rul
from utils.helpers import get_health_status

model = load_trained_model()

prediction = predict_rul(
    model=model,
    mean=-0.094593,
    std=0.081122,
    rms=0.124614,
    kurtosis=1.069163,
    skewness=-0.029993,
    peak_to_peak=1.108
)

print("Predicted RUL :", prediction)
print("Status        :", get_health_status(prediction))