# k-Nearest Neighbours

Implemented from scratch with NumPy, tested on iris.

## What it is

A supervised classifier built on one idea: things close together are probably
the same kind of thing. To classify a new flower it finds the k most similar
training flowers and lets them vote.

It does not really learn anything. My `fit` just stores the data, so the
training set *is* the model. All the work happens at prediction time, measuring
distance from the new point to every training point.

## How it works

For each test point:

1. Euclidean distance to every training point, across all four features.
2. `np.argsort(distances)[:k]` keeps the k closest.
3. `Counter(...).most_common(1)` takes the majority vote.

The only real decision is k.

## What data it suits

Numeric features on comparable scales, not too many of them, small datasets.
It struggles with high dimensional data like text or images, with large
datasets since every prediction searches the whole training set, and with
unscaled features in mixed units.

Iris fits well. Four features, all centimetres, 150 samples.

## Setup and output

`train_test_split(test_size=0.2, random_state=1234)` gives 120 training and 30
test samples. Training splits as 41 setosa, 37 versicolor, 42 virginica. Those
counts matter later.

```
Predictions: [1 1 2 0 1 0 0 0 1 2 1 0 2 1 0 1 2 0 2 1 1 1 1 1 2 0 2 1 2 0]
```

Thirty class labels in the same order as `X_test`, where 0 is setosa, 1 is
versicolor and 2 is virginica. Accuracy is how often this matches `y_test`.

## Varying k

| k  | Accuracy| Wrong| Predicted counts [0,1,2] |
|----|---------|------|--------------------------|
| 1  | 1.0000  | 0    | [9, 13, 8]               |
| 5  | 0.9667  | 1    | [9, 12, 9]               |
| 10 | 1.0000  | 0    | [9, 13, 8]               |
| 25 | 1.0000  | 0    | [9, 13, 8]               |
| 50 | 1.0000  | 0    | [9, 13, 8]               |
| 75 | 0.7667  | 7    | [9, 6, 15]               |
| 100| 0.6333  | 11   | [9, 2, 19]               |

True counts are [9, 13, 8].

**Small k.** k=1 copies the single nearest neighbour. It worked here but is
fragile, since one odd training point creates a wrong region around itself with
no second opinion. The k=5 error was a versicolor called virginica, which are
the two overlapping species.

**Large k.** Accuracy holds to k=50 then collapses, and the class counts show
why. Versicolor predictions fall from 13 to 6 to 2. The model is not making
random errors, it is abandoning one class.

With only 120 training points, k=100 consults 83% of them. The neighbourhood
stops being local, so the vote reflects which class is biggest rather than where
the test flower sits. Versicolor is the smallest training class at 37, so it
gets outvoted, and virginica at 42 climbs from 8 to 19.

I checked this by pushing to k=120. Every test point came back class 2, accuracy
0.2667, which is exactly the training majority class and nothing else.

This is the bias-variance trade-off in one integer. Small k follows the data
including its noise. Large k is stable but smooths the boundaries away.

## Limitations of this test

Four values of k scored 1.0, which says more about iris being easy than about my
model. With 30 test points one flower is worth 3.3 percentage points, so I cannot
meaningfully separate k=10 from k=50. Cross validation across several splits
would be needed for that.

I also did not scale the features. Defensible on iris since everything is
centimetres, but distance sums across features, so on mixed units one feature
would dominate and the rest would be ignored.

Accuracy alone was the weakest part. The class counts diagnosed the large-k
failure, accuracy only showed that something was wrong.

## Reflection

k is not a knob to tune for fractions of a percent. It controls how local the
model is, and at the extremes it changes what the model does at all. At k=1 it
memorises. At k=120 it ignores the input and counts classes.

Implementing it rather than calling scikit-learn is what made that clear.
