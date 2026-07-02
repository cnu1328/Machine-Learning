# Module 01 Quiz

**Passing score:** 12/15 (80%)

Answer all questions. Check solutions only after completing the quiz.

---

## Section A: NumPy (5 questions)

**Q1.** What is the shape of `np.array([[1,2,3],[4,5,6]])`?

- (a) (3, 2)
- (b) (2, 3)
- (c) (6,)
- (d) (3,)

**Q2.** What does `X.mean(axis=0)` compute for a matrix X of shape (100, 5)?

- (a) 100 row means
- (b) 5 column means
- (c) Single overall mean
- (d) 5 row means

**Q3.** What is the result shape of `A @ B` if A is (50, 10) and B is (10, 3)?

- (a) (50, 10)
- (b) (10, 3)
- (c) (50, 3)
- (d) Error

**Q4.** Which operation uses broadcasting?

- (a) `(3, 4) + (4,)`
- (b) `(3, 4) + (3,)`
- (c) `(3, 4) @ (3, 4)`
- (d) Both a and c

**Q5.** Why does ML prefer NumPy arrays over Python lists?

- (a) Homogeneous dtype and contiguous memory
- (b) Built-in linear algebra
- (c) 10-1000× faster numerical operations
- (d) All of the above

---

## Section B: Pandas (5 questions)

**Q6.** What does `df.loc[0, 'price']` do?

- (a) Selects row 0, column 'price' by label
- (b) Selects row 0, column 'price' by integer position
- (c) Filters rows where price equals 0
- (d) Sets price to 0

**Q7.** Why should you split train/test BEFORE imputing missing values?

- (a) Imputation is slow
- (b) To prevent data leakage from test set statistics
- (c) Pandas requires it
- (d) Missing values only exist in test set

**Q8.** What does `pd.get_dummies(df['color'])` produce?

- (a) Label encoding
- (b) One-hot encoding
- (c) Normalization
- (d) Binning

**Q9.** What does `df.groupby('class')['score'].mean()` return?

- (a) Mean score per class
- (b) Total score per class
- (c) Count per class
- (d) Standard deviation per class

**Q10.** What causes SettingWithCopyWarning?

- (a) Modifying a slice/view of a DataFrame
- (b) Using wrong dtype
- (c) Missing values
- (d) Duplicate rows

---

## Section C: Visualization & Vectorization (5 questions)

**Q11.** When should you use a boxplot instead of a histogram?

- (a) To compare distributions across groups and see outliers
- (b) To show correlation
- (c) To plot time series
- (d) To show categorical counts

**Q12.** What is the speedup of vectorized NumPy vs Python loops typically?

- (a) 2-5×
- (b) 10-1000×
- (c) No difference
- (d) Loops are faster

**Q13.** Why subtract the max before computing softmax?

- (a) Numerical stability (prevent overflow in exp)
- (b) Normalize to unit length
- (c) Required by definition
- (d) Makes output sparse

**Q14.** What does a confusion matrix show?

- (a) Feature correlations
- (b) True vs predicted class counts
- (c) Loss over epochs
- (d) Feature importance

**Q15.** Why set `np.random.default_rng(42)`?

- (a) Reproducible random numbers for experiments
- (b) Faster computation
- (c) Required by NumPy
- (d) Better random quality

---

## Answer Key

See [solutions/quiz_answers.md](solutions/quiz_answers.md) after submitting your answers.
