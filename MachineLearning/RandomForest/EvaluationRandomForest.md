# Random Forest

Implemented from scratch on top of my decision tree, tested on the breast cancer
dataset.

## What it is

A random forest is a collection of decision trees that vote. Each tree is grown
on a slightly different version of the training data and is allowed to see only a
random subset of the features at each split. To classify a sample, every tree
votes and the majority wins.

The point is not that any individual tree is good. Each one is deliberately made
worse than it could be. The point is that they are wrong in *different* ways, so
their errors cancel out when averaged.

## How it works

Three mechanisms, and only two of them involve randomness.

**1. Bootstrap sampling.** Each tree gets its own training set, drawn from the
original with replacement:

```python
idxs = np.random.choice(n_samples, n_samples, replace=True)
```

Same size as the original, but some rows appear two or three times and others not
at all. On average about 37% of rows are left out of any given tree.

**2. Feature subsampling.** Each split considers only `n_features` of the 30,
chosen at random, rather than all of them. This is passed straight down to the
DecisionTree.

**3. Majority vote.** `predict` collects every tree's prediction, transposes so
each row is one sample's set of votes, and takes the most common label:

```python
predictions = np.array([tree.predict(X) for tree in self.trees])
tree_preds = np.swapaxes(predictions, 0, 1)
```

## Why the randomness matters

This was the part I understood least at the start. I assumed restricting features
could only make things worse, since each tree is being denied information.

Checking which feature each tree splits on at its root made it concrete. With all
30 features available, 7 of 12 trees split on `worst perimeter` and only 3
distinct features appeared at the root at all. Every tree can see that one
feature is strongest, so they nearly all take it, and the trees end up
near-identical. Averaging twenty copies of the same tree gives you one tree.

With `n_features=5`, nine distinct features appeared at the root across 12 trees.
Any given tree often cannot see the best feature, so it has to use its second or
third choice.

That is the trade. Each tree gets worse, the ensemble gets better, and the
diversity is what makes averaging worth doing. It is the reason the method is
called a *random* forest rather than just a bagged one.

## What data it suits

It inherits the decision tree's strengths. No scaling needed, handles features on
wildly different scales, copes with irrelevant features, no distance calculation
anywhere.

What it adds is robustness. A single tree can be thrown off by a few awkward
training points. A forest averages that away. It also handles larger and noisier
datasets than a single tree, and it is harder to overfit badly.

What it loses is interpretability. A single tree can be read as a flowchart of
conditions. Twenty trees voting cannot. You trade a readable model for a more
reliable one.

## Dataset

Same as the decision tree write-up. `load_breast_cancer()`, 569 samples, 30
features from digitised images of cell nuclei, 212 malignant and 357 benign.
`test_size=0.2, random_state=1234` gives 455 training and **114 test samples**.

That test set size turned out to be the central problem with this whole
evaluation, for reasons in section 7.

## What each test checked

| Test | Question it answers |
|------|--------------------|
| Stability, 5 identical runs | How much does the answer move when nothing changes |
| `n_features` | Does restricting features per split actually help |
| `n_trees` | How many trees before the benefit stops |
| `max_depth` | Does averaging prevent the overfitting a single tree showed |

I ran 3 repeats per setting throughout, after the stability test showed single
runs were not trustworthy.

## Results

### Stability (n_trees=20, n_features=5, five identical runs)

```
0.9298, 0.9474, 0.9211, 0.9211, 0.9298
mean 0.9298, range 0.9211-0.9474, spread 0.0263
```

Three distinct answers from five runs of identical code on identical data.

This is inherited from my decision tree, which calls `np.random.choice` on the
features at every node with no seed, and compounded by the bootstrap sampling,
which is also unseeded. So every number in this report is a sample from a
distribution rather than a measurement.

**My prediction here was wrong.** I expected the forest to be noticeably more
stable than one tree, since averaging should smooth out the randomness. Repeated
runs of the single tree spanned 0.9035 to 0.9298, a spread of 0.0263. The forest
spans 0.9211 to 0.9474, which is the same 0.0263. The averaging shifted accuracy
up by roughly one test sample and did not visibly reduce variance at 20 trees.

### n_features (n_trees=20, 3 repeats)

| n_features | Mean | Range |
|-----------|------|-------|
| 1 | 0.9240 | 0.9211-0.9298 |
| 2 | 0.9327 | 0.9298-0.9386 |
| 5 | 0.9211 | 0.8947-0.9386 |
| 10 | 0.9386 | 0.9211-0.9474 |
| 20 | 0.9269 | 0.9035-0.9474 |
| 30 | 0.9181 | 0.9123-0.9298 |

**This sweep did not resolve anything.** Every range overlaps every other range.
I checked all 15 pairs and not one of them is separable. The gap from the best
mean (10, at 0.9386) to the worst (30, at 0.9181) is 0.0205, which is two test
samples.

The one hint pointing the right way is that `n_features=30`, using every feature,
had the lowest mean of the six. That is the direction theory predicts. Two samples
is a hint and not evidence.

### n_trees (n_features=5, 3 repeats)

| n_trees | Mean | Range | Spread in samples |
|---------|------|-------|------------------|
| 1 | 0.9123 | 0.9035-0.9211 | 2 |
| 5 | 0.9152 | 0.8947-0.9386 | 5 |
| 10 | 0.9181 | 0.9035-0.9474 | 5 |
| 20 | 0.9386 | 0.9298-0.9474 | 2 |
| 50 | 0.9269 | 0.9211-0.9386 | 2 |
| 100 | 0.9357 | 0.9298-0.9386 | 1 |

**This is the only clean result in the entire evaluation.** `n_trees=1` ranges
0.9035 to 0.9211. `n_trees=20` ranges 0.9298 to 0.9474. Those do not overlap.
Neither do 1 and 100. Every other pair in every other sweep does.

So one tree is genuinely worse than twenty, and that is the ensemble benefit
demonstrated rather than assumed. It is worth about three test samples.

Beyond 20 nothing separates. The dip at 50 (0.9269) below both 20 and 100 is
noise, since all three ranges overlap heavily. The broad shape matches the
expected climb-then-plateau, but I cannot locate where the plateau starts.

There is one weak signal in the spread column. The run-to-run spread is 5 samples
at 10 trees and 1 sample at 100. That is the variance reduction I expected to find
in the stability test, showing up only at tree counts well above 20. With 3
repeats a spread estimate is very unreliable, so this is suggestive at best.

### max_depth (n_trees=20, n_features=5, 3 repeats)

| max_depth | Train | Test | Test range |
|-----------|-------|------|-----------|
| 1 | 0.9502 | 0.8977 | 0.8772-0.9123 |
| 2 | 0.9663 | 0.8977 | 0.8860-0.9123 |
| 3 | 0.9853 | 0.9094 | 0.8947-0.9298 |
| 5 | 0.9971 | 0.9298 | 0.9211-0.9386 |
| 10 | 0.9993 | 0.9298 | 0.9298-0.9298 |
| 20 | 0.9985 | 0.9269 | 0.9123-0.9386 |
| 100 | 1.0000 | 0.9269 | 0.9211-0.9298 |

Training accuracy climbs steadily to exactly 1.0. Test accuracy climbs to 0.9298
by depth 5 and then flattens.

**The forest did not prevent memorisation.** I expected averaging to stop the
training accuracy reaching 1.0 the way the single tree's did. It did not. At
depth 100 the forest classifies the training data perfectly, exactly as one deep
tree does.

What it did change is the consequence. The single tree's test accuracy fell away
from its peak as depth grew. The forest's test accuracy plateaus instead of
falling. So the forest still overfits in the sense of memorising, but it is less
punished for it. That is a narrower claim than the one I started with and it is
the one the data supports.

One oddity: depth 10 returned 0.9298 three times exactly. With 3 repeats that is
coincidence rather than a sign of stability, and depth 5 and 20 on either side of
it both moved around.

## Roadblocks

**Nothing was reproducible.** The underlying tree seeds nothing, so identical
inputs gave different answers. Everything downstream inherits this. I should have
added a `random_state` before running a single experiment rather than working
around the noise afterwards.

**Three repeats was not enough.** It was enough to show that single runs lie. It
was not enough to separate settings that differ by two or three test samples,
which turned out to be most of them. Fifteen out of fifteen `n_features` pairs
were inseparable.

**114 test samples is too coarse.** One sample is 0.88 percentage points, so
accuracy can only land on multiples of 0.0088. Most of the differences I was
trying to read were one or two of those steps. No number of repeats fixes this,
because the limit is the test set, not the sampling.

**I picked n_features=5 before the sweep finished.** I used the sqrt(30)
convention to fix a value for the `n_trees` and `max_depth` sweeps, and my own
sweep then weakly favoured 10 over 5. The ranges overlap so this does not
invalidate the later results, but choosing a parameter by convention and then
discovering the data mildly disagreed is worth recording.

**Two predictions were falsified.** I expected the forest to be more stable than
one tree, and to avoid reaching 1.0 training accuracy. Neither happened. Both are
reported above as they came out.

**Smaller things.** The constructor took `n_feature` while storing
`self.n_features`, so `clf.n_feature` raised an `AttributeError` even though
passing `n_feature=5` worked. Import errors from having `decision_tree.py` in one
folder and `random_forest.py` in another, since Python looks for modules beside
the script being run.

## What I could have tested

**Out-of-bag error.** The most obvious omission, and specific to forests.
Bootstrap sampling leaves roughly 37% of rows out of each tree, and those rows are
unseen validation data for that tree, available for free. Scoring each sample
using only the trees that did not train on it gives a validation estimate across
all 455 training rows rather than 114 test rows, which would have solved the
resolution problem that undermined most of this evaluation. My implementation
does not record which rows were left out, so this was not available.

**Cross-validation.** Reusing the data as several different splits instead of
trusting one. More total effort than repeats but it attacks the actual problem,
which was the size of the test set rather than the randomness of the model.

**Feature importance.** A forest can report which features contributed most
information gain across all trees. This is the main interpretability output the
method offers and I did not implement it. Given my project is on explainable
machine learning this is the gap I should close first.

**A confusion matrix.** My test set holds 45 malignant and 69 benign cases and I
never looked at which direction the errors fell. For a cancer dataset the
distinction between a missed malignant case and a false alarm matters more than
the overall accuracy.

**Correlation between trees, measured directly.** The root-split check was a
rough proxy. Measuring how often pairs of trees agree on the test set would
quantify the decorrelation that `n_features` is supposed to produce, without
going through accuracy at all.

**scikit-learn's RandomForestClassifier** as a reference point, to see whether my
implementation reaches comparable accuracy and what it costs in speed.

## Reflection

The main lesson is about measurement rather than about forests. I ran four sweeps,
24 configurations and over 70 model fits, and exactly one comparison in the whole
set is separable from noise. Everything else overlaps. Had I not printed the
range alongside the mean I would have written up a confident story about
`n_features=10` being optimal and a dip at 50 trees, and all of it would have been
invented.

What that points to is that the bottleneck was never the number of repeats. It was
the single 114-sample test set. Out-of-bag scoring would have used all 455
training rows and was sitting there unused inside the bootstrap I had already
implemented.

Comparing the three models: kNN had one parameter and a clear optimum. The
decision tree had a parameter that stopped doing anything past depth 7. The forest
has four parameters and I could only resolve one of them. Model complexity went up
and my ability to measure the effect of any individual knob went down, which is
not something I expected.

The one thing I am confident of is that twenty trees beat one, by about three test
samples, and the ranges do not overlap.

