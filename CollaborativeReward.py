import nltk
nltk.download('averaged_perceptron_tagger')

class CollaborativeReward:
    def __init__(self, pred_label, true_label, list_pred_label):
        self.pred_label = pred_label
        self.true_label = true_label
        self.list_pred_label = list_pred_label

    def calculate_reward(self):
        collab_reward = 0
        true_conditions = []  # initialize an empty list to store the index numbers of true conditions
        false_conditions = []
        for idx, (i, j, k) in enumerate(zip(self.pred_label, self.true_label, self.list_pred_label)):
            if i == j == k:
                tagged_word = nltk.pos_tag([i])
                tagged_word1 = nltk.pos_tag([j])
                if tagged_word[0][1] == 'NNS' and tagged_word1[0][1] == 'NNS':
                    collab_reward += 1
                elif tagged_word[0][1] == 'NN' and tagged_word1[0][1] == 'NN':
                    collab_reward += 1
                else:
                    collab_reward += 0
                true_conditions.append(idx)  # add the index number of the true condition to the list
            else:
                false_conditions.append(idx)

        return collab_reward, true_conditions, false_conditions  # return the collaborative reward and the list of index numbers of true conditions
