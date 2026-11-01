# Decision Tree

Implemented from scratch with NumPy, tested on the breast cancer dataset.

## What it is

A decision tree classifies by asking a series of yes or no questions. You start
at the top, apply the question stored at that node, go left or right depending on
the answer, and keep going until you reach a leaf. The leaf stores a class label,
and that label is the prediction.

Each question has the form "is feature X less than or equal to some threshold".
Nothing more complicated than that. The whole skill of the algorithm is choosing
which feature and which threshold at each node.

## Entropy

The tree picks splits by measuring uncertainty, and the measure it uses is
entropy.

```
E = -sum( p(X) * log2(p(X)) )     where p(X) = count of X / n
```

For a two class problem:

- **Entropy 1** is the worst case. A 50/50 mix, completely uncertain, no basis
  for a prediction.
- **Entropy 0** is the best case. Every sample in the node is the same class, so
  you can predict with certainty.

Worked example from a node holding `[0,0,0,0,0,1,1,1,1,1]`:

```
E = -(5/10)log2(5/10) - (5/10)log2(5/10) = -0.5(-1) - 0.5(-1) = 1
```

Maximum uncertainty, as expected from an even split.

One note on my code. My `_entropy` uses `np.log`, which is natural log, not
`np.log2`. That scales every entropy value by a constant factor, so the split the
tree chooses is unchanged. But it means a printed entropy would read 0.693 at the
50/50 point rather than 1, so the values would not match the formula above.

## Information gain

Entropy tells you how uncertain one node is. Information gain tells you how much
a split reduces that uncertainty.

```
IG = E(parent) - weighted average of E(children)
```

The weighting is by how many samples land on each side, so a split that isolates
three samples counts for less than one that splits the node in half.

Worked example. Parent `S = [0,0,0,0,0,1,1,1,1,1]` with entropy 1, split into
`S1 = [0,0,1,1,1,1,1]` and `S2 = [0,0,0]`:

```
IG = 1 - [ (7/10)(0.863) + (3/10)(0) ] = 0.395
```

S2 is pure so contributes zero entropy. The split has bought 0.395 bits.

## How the tree is built

At every node the algorithm does a **greedy search**. It loops over every feature
and every possible threshold in that feature, calculates the information gain for
each, and keeps the best one. Then it splits the data and repeats on both
children.

Greedy means it takes the best split available right now without checking whether
a worse split now would allow a better one later. It never backtracks. This is
fast but it does not guarantee the best overall tree.

Growing stops when any stopping criterion is met: maximum depth reached, the node
is pure, or too few samples remain to split.

## What data it suits

Decision trees work well with:

- Mixed feature types, since each split only compares one feature to a threshold.
- Features on different scales. Unlike kNN there is no distance calculation, so a
  feature measured in thousands does not drown out one measured in decimals. No
  scaling needed.
- Problems where you need to explain the decision, since the path from root to
  leaf is a readable list of conditions.
- Irrelevant features, which the tree simply never splits on.

They work badly on:

- Problems with smooth diagonal boundaries, since splits are always axis aligned
  and have to approximate a diagonal with a staircase.
- Small datasets, where they overfit readily.
- Data where small changes matter, since a slightly different training set can
  produce a very different tree.

## The dataset

`load_breast_cancer()` from scikit-learn. 569 samples, 30 features, binary target.

The features come from digitised images of fine needle aspirates of breast
masses. Ten base measurements are taken of each cell nucleus, things like radius,
texture, perimeter, area and smoothness. Each of those appears three times, as
the mean, the standard error and the worst value, which gives 30 columns.

The target is 212 malignant and 357 benign, so about 63% benign.

My split with `test_size=0.2, random_state=1234` gives:

- **455 training samples** (167 malignant, 288 benign)
- **114 test samples** (45 malignant, 69 benign)

That test size matters when reading the results. One test sample is worth 0.88
percentage points, so a difference of 0.9386 against 0.9211 is two samples.

The data suits a tree well. Every feature is numeric, they sit on very different
scales which the tree does not care about, and several features are strongly
correlated with the diagnosis so good splits exist near the root.

## What we changed

My class has three parameters. I varied only `max_depth`, for simplicity.

**`max_depth`** caps how many questions deep the tree can go. A shallow tree asks
few questions so it is simple and may be too crude. A deep tree asks many so it
can carve out very specific regions, including regions that only exist because of
noise in the training data.

**`min_samples_split`** is the minimum number of samples a node needs before it is
allowed to split at all. Raising it stops the tree splitting on tiny groups.
Default 2, meaning split whenever possible.

**`n_features`** is how many features each split considers. Mine defaults to all
30. Lowering it forces each split to choose from a random subset, which usually
hurts a single tree but is the mechanism behind Random Forest, where many
deliberately varied trees are averaged.

## Results

| max_depth | Accuracy |
|-----------|----------|
| 1 | 0.8772 |
| 2 | 0.8947 |
| 3 | 0.9386 |
| 5 | 0.9211 |
| 10 | 0.9298 |
| 20 | 0.9123 |
| 50 | 0.9298 |
| 100 | 0.9211 |

Depths 1 and 2 are clearly worse. The tree is only allowed one or two questions,
which is not enough to separate the classes. This is underfitting.

Depth 3 was the best result and everything from 5 onwards sits around 0.92 with
no clear trend.

### Why the deeper runs look random

The wobble from depth 5 upwards is not telling me anything about depth. Two
things are going on.

First, **the tree stops growing on its own long before the cap**. Running the
code repeatedly, the actual tree reaches depth 5 to 7 with 15 leaves regardless
of whether `max_depth` is set to 10, 20, 50 or 100. Once every leaf is pure or
too small to split, growth stops. So all four of those settings build
essentially the same tree, and `max_depth` above about 7 does nothing at all.

Second, **the code is not deterministic**. The line

```python
feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)
```

shuffles the feature order at every node. When two features give equal
information gain, `if gain > best_gain` keeps whichever was seen first, so ties
break randomly and repeated runs on identical data build different trees. Running
`max_depth=10` six times gave four different accuracies spanning 0.9035 to
0.9298.

So the differences between my depth 10, 20, 50 and 100 results are random
variation, not an effect of depth.

### The overfitting signature

Running the same sweep with training accuracy printed alongside test accuracy,
averaged over 5 runs to smooth out the randomness:

| depth | train | test |
|-------|-------|------|
| 1 | 0.9253 | 0.8772 |
| 2 | 0.9297 | 0.8982 |
| 3 | 0.9780 | 0.9526 |
| 5 | 0.9991 | 0.9193 |
| 10 | 1.0000 | 0.9211 |
| 20 | 1.0000 | 0.9211 |

This is the picture the test column alone could not show. By depth 5 the tree
classifies the training data essentially perfectly, and by depth 10 it is at
exactly 1.0. Test accuracy does not follow. It peaks around depth 3 and then
drops and flattens.

The gap between the two columns is overfitting. The tree has enough depth to
carve out a region around every awkward training point, which fits the training
set perfectly and does not generalise.

## Reflection

The thing I would not have predicted is that the parameter I was varying stopped
doing anything after about depth 7, and I could not see that from the accuracy
numbers alone. Four settings that looked like four different experiments were the
same experiment repeated with a random seed. Checking the actual depth the tree
reached is what revealed it.

Comparing with kNN, the two models fail in opposite directions as their parameter
grows. Large k made kNN simpler until it just predicted the majority class. Large
max_depth makes a tree more complex until it memorises the training set. Both
curves have a middle region that works, and both ends are useless for different
reasons.

The other difference worth noting is that I did not scale anything here. Breast
cancer has features ranging from fractal dimensions around 0.003 to areas in the
thousands, which would have wrecked a distance based model like kNN. A tree
compares one feature to one threshold at a time, so the units never interact.
