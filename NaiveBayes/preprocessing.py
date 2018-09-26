import re
import random


# Different types of built in filters should be declared here
# The structure should be: filter_name = {list of match_pattern : replacement}

# filter_ascii is a chain of three regexp
# The first one removes the new line, avoiding to put a space at the end of the phrase
# The first one matches any character which is not a letter and replaces it with a space
# The second one matches any sequence of more than one space introduced before with a single space
filter_ascii = {
    '[^a-zA-Z]*\n': '',
    '[^a-zA-Z ]': ' ',
    '[ ]+': ' '
}


# Class used to preprocess out dataset
# Please note that the class has been built to be functional
# This means that by default each method return self to use dot notation and allow operation chaining
# You can turn off this feature and use the methods as normal function by passing functional=false
class Preprocessor:

    # Initialize instance, requires dataset as input
    # To debug, pass verbose=True
    def __init__(self, data, verbose=False):
        self.data = data
        self.preprocessed = None
        self.tokenized = None
        self.vocabulary = None
        self.x_tr, self.y_tr, self.x_ts, self.y_ts = None, None, None, None

        self.verbose = verbose

    # Preprocess the input, removing by default all non alpha characters
    # Can receive a different preprocess filter defined by the user
    # Creates a list of preprocessed sentences
    def preprocess(self, filter=filter_ascii, functional=True):
        self.preprocessed = []
        for line in self.data:
            filtered = line
            for exp, replacement in filter.items():
                filtered = re.sub(exp, replacement, filtered)

            if self.verbose:
                print(line + " -> " + filtered)

            self.preprocessed.append(filtered)
        return self if functional else self.preprocessed

    # Tokenize each entry: the first element is the class, the rest are the various tokens
    # Each token is set to lowercase
    # Creates a list of list of token, where the first element is the class label
    # The tokens are separated with the given separator, by default it is a space
    def tokenize(self, separator=' ', functional=True):

        if self.preprocessed is None:
            self.preprocess()

        self.tokenized = []
        for line in self.preprocessed:

            # For each line in the preprocessed data split it into tokens according to the given separator
            tokens_ = [tk.lower() for tk in line.split(separator)]
            self.tokenized.append(tokens_)

            if self.verbose:
                print(str.upper(tokens_[0]) + " -> " + str(tokens_[1:]))

        return self if functional else self.tokenized

    # Return a vocabulary containing all the different words present in the dataset
    def words(self, functional=True):

        if self.tokenized is None:
            self.tokenize()

        self.vocabulary = []

        for record in self.tokenized:
            for token in record[1:]:
                if token not in self.vocabulary:
                    self.vocabulary.append(token)

        return self if functional else self.vocabulary

    # Splits data in train and test
    # Can define custom train size, default is 80%
    # Can define a customized seed, to have always the same split. Useful to debug. Default is None
    # If shuffle is true, the data will be shuffled
    def split(self, percentage_train=0.8, seed=None, shuffle=True, functional=True):

        # Tokenize data if not already done it
        if self.tokenized is None:
            self.tokenize()

        # If a random seed is provided the split will always be the same
        if seed is not None:
            random.seed(seed)

        if shuffle:
            random.shuffle(self.tokenized)

        # Prepare containers for train/test X and Y
        self.x_tr, self.y_tr, self.x_ts, self.y_ts = [], [], [], []

        # The number of messages in our tokenized dataset
        data_size = len(self.tokenized)

        # Compute train and test size as an integer numbers
        train_size = round(percentage_train * data_size)
        test_size = data_size - train_size

        # Compute the start index for the training data
        tr_idx = random.randint(0, data_size - 1)

        # Compute the start index for the testing data.
        # Say we have 5000 samples. start_idx for train is 4000 and train is 4000 samples long.
        # We need a way to init the proper test_idx to 3000
        # These few lines of code achieve such task.
        # The idea is like having a circular array, whenever we reach the top the index is floored to 0 and
        # incremented again from there
        ts_idx = tr_idx + train_size
        if ts_idx >= data_size:
            available = data_size - tr_idx
            ts_idx = train_size - available

        print("data_size = " + str(data_size))
        print("train_size = " + str(train_size) + " :: test_size = " + str(test_size))
        print("train_begin_index = " + str(tr_idx) + " :: test_begin_index = " + str(ts_idx))

        # Build training set
        self.x_tr, self.y_tr = self._subsplit(data=self.tokenized, start=tr_idx, size=train_size)
        # Build testing set
        self.x_ts, self.y_ts = self._subsplit(data=self.tokenized, start=ts_idx, size=test_size)

        return self if functional else self.x_tr, self.y_tr, self.x_ts, self.y_ts

    # This method creates two vectors X and Y.
    # X and Y could be the training or testing batches
    # This methods should never be invoked directly by the user
    @staticmethod
    def _subsplit(data, start, size):

        # Containers for x and y (either train, validation or test)
        x, y = [], []

        # Starting index
        idx = start
        data_size = len(data)

        # Counter to keep track of how many samples have been added to the list
        added = 0
        while added < size:
            if idx == data_size:
                idx = 0

            # This is the current sample we are analyzing
            sample = data[idx]

            # [1:] Means every token except the first one which is the class label
            # basically we are adding to x all the features and to y the class label
            x.append(sample[1:])
            y.append(sample[0])

            # Increment both counter and index
            added += 1
            idx += 1

        return x, y



