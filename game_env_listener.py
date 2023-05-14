from listener_agent import ListenerAgent

class GameEnvironmentListener:
    def __init__(self, data_path, filenames, shapes):
        self.listener_agent = ListenerAgent(data_path=data_path, filenames=filenames, shapes=shapes)

    def play_game(self, test_img_dir, test_img, predicted_labels):
        matched_images = []
        predicted_labels_listener = []
        listener_reward = 0

        for image_path, predicted_label in zip(test_img, predicted_labels):
            test_image_path = test_img_dir + image_path
            matched_image, predicted_label_listener = self.listener_agent.predict_label(test_image_path, predicted_label)
            matched_images.append(matched_image)
            predicted_labels_listener.append(predicted_label_listener)

            if predicted_label_listener == predicted_label:
                listener_reward += 1

        return matched_images, predicted_labels_listener, listener_reward