import data_prep_mnist as dload
import model_prep_mnist as mload
import matplotlib.pyplot as plt

raw_images, labels = dload.load_data("mnist.onion")
image_list, label_list = dload.load_and_process_data("mnist.onion")
model = mload.prepare_model_for_training()

# _id = 23
# plt.imshow(raw_images[_id])
# print(labels[_id])
# plt.imshow(image_list[_id])
mload.train_model(model, image_list, label_list, n_splits=10, test_size=0.2, random_state=47, n_epochs=50)