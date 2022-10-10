from seg_models import build_unet
from data_loader import demo_train_val_test, show_predictions
import tensorflow as tf

# Build and compile our U-Net model
model = build_unet((128, 128, 3))
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss="sparse_categorical_crossentropy",
              metrics="accuracy")

# Generate train, val, and test batches respectively
train_ds, val_ds, test_ds, info = demo_train_val_test()

# Data Parameters for training
BATCH_SIZE = 64
NUM_EPOCHS = 40
TRAIN_LENGTH = info.splits["train"].num_examples
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
VAL_SUBSPLITS = 5
TEST_LENTH = info.splits["test"].num_examples
VALIDATION_STEPS = TEST_LENTH // BATCH_SIZE // VAL_SUBSPLITS

# Train our model
model_history = model.fit(train_ds,
                          epochs=NUM_EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=val_ds)

loss, accuracy = model.evaluate(test_ds, verbose=2)

show_predictions(test_ds, model)
