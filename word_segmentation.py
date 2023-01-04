import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix


def read_data():
    """
    Returns the decoded lines of text read from training.txt and test.txt.
        Returns:
            lines ([],[]) - - tuple containing lists of text from the training and test data, respectively.
    """
    encoding = 'big5hkscs'
    lines = ([], [])
    num_errors_train = 0
    num_errors_test = 0

    with open('training.txt', 'rb') as f:
        for i, line in enumerate(f):
            try:
                lines[0].append(line.decode(encoding))
                decoded = line.decode(encoding)
            except UnicodeDecodeError as e:
                num_errors_train += 1
    with open('test.txt', 'rb') as f:
        for i, line in enumerate(f):
            try:
                lines[1].append(line.decode(encoding))
                decoded = line.decode(encoding)
            except UnicodeDecodeError as e:
                num_errors_test += 1
    print('Encountered %d decoding errors in the training data.' % num_errors_train)
    print('Encountered %d decoding errors in the test data.' % num_errors_test)

    return lines


def encode_data(lines, isTest, train_dict=None):
    """
    Returns the sparse matrix containing the encoded data from input lines. Also returns the y-values indicating the
    presence of a word segmentation in the 4-gram.
        Parameters:
            lines ([],[]) - - tuple containing lists of text in training and test data, respectively in input text.
            isTest (boolean) - - boolean indicating whether we are encoding test data
            train_dict (dictionary) - - dictionary of word encodings created using the training data (used to encode
                                            both training and test data.
        Returns:
            M (sklearn csr matrix) - - sparse row matrix storing encoded lines of text
            y (boolean) - - boolean array indicating presence of a word segmentation in each 4-gram
            used_chars (dictionary) - - dictionary storing words seen in the lines
        """
    y = []
    features = ([], [])
    used_chars = {}
    if not isTest:
        train_dict = used_chars

    # Extract 4-grams and y-values, maintaining encoding in used_chars_train for every seen character
    for i, line in enumerate(lines[int(isTest)]):  # 0 for Train, 1 for Test
        # without wrapping around, a line is guaranteed to be unusable if it is shorter than 4 characters
        if len(line) < 4:
            continue

        for start in range(len(line) - 3):
            skip = False
            if line[start].isspace():  # make sure to skip spaces as starting point for N-gram to avoid duplicate grams
                continue

            gram, has_space = extract_gram(line[start: min(start + 12, len(line))], used_chars)
            if gram is None:  # line does not have 4 non-space characters
                break
            if isTest:  # check if gram contains characters the model was not trained on
                for char in gram:
                    if char not in train_dict:
                        skip = True  # character is not in the training data
            if not skip:
                y.append(int(has_space))  # append 0/1 determining if there is a word separation
                features[int(isTest)].append([gram[0] + gram[1], gram[1], gram[1] + gram[2], gram[2], gram[2] + gram[3]])

    encoded = []
    for feature in features[int(isTest)]:
        encoded.append(encode(feature, train_dict))

    K = len(train_dict)
    encoded = np.array(encoded)
    cols = encoded + np.array([0, K ** 2, K ** 2 + K, 2 * K ** 2 + K, 2 * K ** 2 + 2 * K])
    indices = np.reshape(cols, -1)
    indptr = np.arange(0, 5 * (encoded.shape[0] + 1), 5)
    data = np.ones(len(indices))
    M = csr_matrix((data, indices, indptr), shape=(encoded.shape[0], 3 * K ** 2 + 2 * K))

    return M, y, used_chars


def encode(feature, used_chars):
    """
    Encode a list of features using the encoding defined by the dictionary of seen characters (used_chars).
        Parameters:
            feature (list) - - list of format (AB,B,BC,C,CD) for extracted 4-gram ABCD
            used_chars (dictionary) - - dictionary storing encodings for each character seen thus far in the training
                                        data.
        Returns:
            encoded (list) - - list of format (##,#,##,#,##) consisting of the base-K representation encodings of the
                                input feature.
    """
    encoded = []
    K = len(used_chars)
    for i in range(5):
        if i % 2 == 0:
            encoded.append(K * (used_chars[feature[i][0]] + 1) + used_chars[feature[i][1]])
        else:
            encoded.append(used_chars[feature[i]])
    return encoded


def extract_gram(text, used_chars):
    """
    Return the first gram (list of 4 consecutive non-space characters) in the input text and a boolean indicating
    whether there is a space between the middle two gram characters. Returns None if there are not 4 non-space
    characters in input text.
        Parameters:
            text (string) - - string to extract the first 4 non-space characters from
            used_chars (dictionary) - - dictionary storing encodings for each character seen thus far in the training
                                            data.
        Returns:
            gram (list) - - first 4 non-space characters. Returns None if there are not 4 non-space characters
                            in input text.
            has_space (bool) - - True if there is a space between the middle two non-space characters in the gram,
                                    False otherwise.
    """
    gram = []
    has_space = False

    for i, char in enumerate(text):

        if len(gram) == 4:  # if 4 gram has been filled, add new characters to encoding in dictionary and break
            for char_ in gram:
                if char_ not in used_chars:
                    used_chars[char_] = len(used_chars)
            break

        if not char.isspace():  # add character to the 4 gram if it is not a space
            gram.append(char)
            if len(gram) == 2 and text[i + 1].isspace():  # find if there is a space between B and C
                has_space = True

    if len(gram) < 4:
        return None, False
    return gram, has_space


def main():
    """
    Trains a word segmentation classifier (logistic regression or naive bayes classifier) on the sparse row matrix
    representation of the encoded 4-grams extracted from the training data.
    Tests the classifier on the test data, printing classification metrics to evaluate performance.
    """
    # Read and encode data
    lines = read_data()
    print('Read data.')
    M_train, y_train, train_chars = encode_data(lines, isTest=False)
    print('Encoded train.')
    M_test, y_test, test_chars = encode_data(lines, isTest=True, train_dict=train_chars)
    print('Encoded test.')

    # Print message for unseen characters in test data
    unseen = []
    for char in test_chars:
        if char not in train_chars:
            unseen.append(char)
    if len(unseen) > 0:
        print('Test data contained previously unseen characters. Discarding 4-grams in text containing letters:', unseen)

    # Train model
    print('Training '+model+' model...')
    if model == 'Logistic Regression':
        clf = SGDClassifier(loss='log', max_iter=10000, tol=1e-5).fit(M_train, y_train)
    elif model == 'Naive Bayes':
        clf = MultinomialNB().fit(M_train, y_train)
    else:
        print('No such model - please select \'Logistic Regression\' or \'Naive Bayes\'')
        return
    print('Model trained.')

    # Print training accuracy
    y_hat_train = clf.predict(M_train)
    train_error = np.mean(np.abs(y_train - y_hat_train))
    print('Train Accuracy = ', round(1 - train_error, 3))

    # Print test metrics
    y_hat_test = clf.predict(M_test)
    test_error = np.mean(np.abs(y_test - y_hat_test))
    print('Test Accuracy = ', round(1 - test_error, 3))
    cm = confusion_matrix(y_test, y_hat_test)
    print('Confusion matrix on test data: \n', cm)
    tn, fp, fn, tp = cm.reshape(-1)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    F1 = 2 * (precision * recall)/(precision + recall)
    print('Precision = ', round(precision, 3))
    print('Recall = ', round(recall, 3))
    print('F1 score = ', round(F1, 3))


# Select model to use:
model = 'Naive Bayes'  # ['Logistic Regression' | 'Naive Bayes']
main()
