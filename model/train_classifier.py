import tensorflow as tf
from datetime import datetime
from pathlib import Path
import mlflow
from data_prep import load_data, image_size
from model_definition import create_model

# --- Main Training Function ---
def run_training():
    mlflow.autolog()
    # 1. Load Data
    train_ds, val_ds, test_ds, class_names = load_data()
    num_classes = len(class_names)
    
    # 2. Define Model Input Shape
    input_shape = image_size + (3,) # e.g., (64, 64, 3) for RGB images
    
    # 3. Create Model
    model = create_model(input_shape, num_classes)
    model.summary() # Print the model structure

    # 4. Train the Model
    # epochs: number of passes over the entire training dataset
    log_dir = Path("logs") / "fit" / datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    tb_callback = tf.keras.callbacks.TensorBoard(
        log_dir=str(log_dir),
        histogram_freq=1,
    )

    mlflow.set_experiment("Dog_Cat_Classifier")

    with mlflow.start_run(run_name="CNN_Classifier"):
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=20, # Start with 20, adjust based on validation results
            callbacks=[tb_callback],
        )

        # 5. Evaluate the Model on the Test Set
        print("\nEvaluating model on the test set...")
        loss, accuracy = model.evaluate(test_ds, verbose=2)
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")

        # 6. Save the Model
        MODEL_SAVE_PATH = 'trained_cnn_model.keras'
        model.export(MODEL_SAVE_PATH)
        print(f"\nModel saved to: {MODEL_SAVE_PATH}")

# To run this script:
if __name__ == '__main__':
#     # NOTE: You must first create your image folders and place images inside them!
     run_training()