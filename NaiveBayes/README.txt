Dependencies:

You should have numpy installed, if not type in the terminal "pip install numpy"
If you are using an IDE such as PyCharm, you can have it automatically install numpy for you.

All other components are already builtin into the latest releases of python

Execution:

If you are using an IDE such as PyCharm, right click on the file "main.py" and select RUN
Otherwise open a terminal and cd into the root folder "NaiveBayes".
Typing ls (on unix-like systems) or dir (Windows) you should see all the project files (main, model, preprocessing..)

At this point enter : "python main.py"

Depending on your hardware specs, the program should take around 5-7 seconds to exit.

It will print the request of point a) on STDOUT and will plot the requested graphs in point b)

Additional Information:

During the split phase is possible to pass a seed. This is a way to obtain always the same split.
With a seed = 5555 (The value is totally arbitrary, is just a way to initialize the random function always to the same value),
alpha = 0.1,
N = 20.000
I obtained the following results:

Description of Training scores
Confusion matrix:
TP = 607
TN = 3836
FP = 11
FN = 5

Measures:
ACCURACY = 0.9964117515137924
PRECISION = 0.982200647249191
RECALL = 0.9918300653594772
F-SCORE = 0.9869918699186992


Description of Training scores
Confusion matrix:
TP = 124
TN = 967
FP = 13
FN = 11

Measures:
ACCURACY = 0.97847533632287
PRECISION = 0.9051094890510949
RECALL = 0.9185185185185185
F-SCORE = 0.9117647058823529

