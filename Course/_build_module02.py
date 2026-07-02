#!/usr/bin/env python3
"""Generate Module 02 Mathematics notebooks."""
import json
from pathlib import Path

M02 = Path(__file__).resolve().parent / "02_Mathematics"


def nb(cells):
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "cells": cells,
    }


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": [s] if isinstance(s, str) else s}


def code(s):
    return {"cell_type": "code", "metadata": {}, "source": [s], "outputs": [], "execution_count": None}


def save(name, cells):
    p = M02 / name
    p.write_text(json.dumps(nb(cells), indent=1))
    print(f"  {name}")


HEADER = "**Module:** 02 Mathematics  \n**Part:** {part}  \n**Duration:** ~{dur}\n\n---\n\n## Learning Objectives\n\n{objs}\n"


def std_footer(n, title, nxt=None):
    nxt_line = f"\n\n**Next:** [{nxt}]({nxt})" if nxt else ""
    return md(f"---\n\n## Interview Questions\n\nSee module quiz and exercises.\n\n## Summary\n\n{title}\n\n**Notebook {n} complete.**{nxt_line}")


IMPORTS = code(
    "import numpy as np\nimport matplotlib.pyplot as plt\nfrom matplotlib import cm\n\nplt.rcParams['figure.figsize'] = (8, 5)\nplt.rcParams['font.size'] = 11\nrng = np.random.default_rng(42)"
)


def build_part_a():
    """Linear Algebra notebooks 01-07."""
    save("01_Scalars_and_Vectors.ipynb", [
        md(HEADER.format(part="A — Linear Algebra", dur="2 hours",
            objs="1. Define scalars and vectors mathematically\n2. Compute magnitude (L2 norm) and unit vectors\n3. Perform vector addition, scalar multiplication, dot product\n4. Understand geometric interpretation of vectors in ML")),
        md("## 1. Intuition\n\nA **vector** is an ordered list of numbers that represents:\n- A **point** in space (e.g., house features: [area, bedrooms, age])\n- A **direction and magnitude** (e.g., gradient direction during training)\n\nA **scalar** is a single number (e.g., learning rate η = 0.01, loss L = 0.45)."),
        md("## 2. Mathematical Definition\n\n**Scalar:** $c \\in \\mathbb{R}$ — one real number.\n\n**Vector:** $\\mathbf{v} = \\begin{bmatrix} v_1 \\\\ v_2 \\\\ \\vdots \\\\ v_n \\end{bmatrix} \\in \\mathbb{R}^n$ — n real numbers.\n\n**Magnitude (L2 norm):**\n$$\\|\\mathbf{v}\\| = \\sqrt{v_1^2 + v_2^2 + \\cdots + v_n^2} = \\sqrt{\\mathbf{v}^T \\mathbf{v}}$$\n\n**Unit vector:** $\\hat{\\mathbf{v}} = \\frac{\\mathbf{v}}{\\|\\mathbf{v}\\|}$"),
        IMPORTS,
        code("v = np.array([3.0, 4.0])\nmagnitude = np.linalg.norm(v)\nunit = v / magnitude\nprint(f'v = {v}')\nprint(f'||v|| = {magnitude}')  # 5.0\nprint(f'unit vector = {unit}')\nprint(f'||unit|| = {np.linalg.norm(unit):.6f}')  # 1.0"),
        md("## 3. Vector Operations\n\n**Addition:** $\\mathbf{a} + \\mathbf{b} = [a_1+b_1, \\ldots, a_n+b_n]$\n\n**Scalar multiplication:** $c\\mathbf{v} = [cv_1, \\ldots, cv_n]$\n\n**Dot product:** $\\mathbf{a} \\cdot \\mathbf{b} = \\sum_{i=1}^{n} a_i b_i = \\mathbf{a}^T \\mathbf{b}$\n\nGeometric meaning of dot product:\n$$\\mathbf{a} \\cdot \\mathbf{b} = \\|\\mathbf{a}\\| \\|\\mathbf{b}\\| \\cos\\theta$$\n\nIf dot product = 0, vectors are **orthogonal** (perpendicular)."),
        code("a = np.array([1, 2, 3])\nb = np.array([4, 5, 6])\nprint('a + b =', a + b)\nprint('2 * a =', 2 * a)\nprint('a · b =', np.dot(a, b))  # 32\n\n# Orthogonal vectors\nx = np.array([1, 0])\ny = np.array([0, 1])\nprint('x · y =', np.dot(x, y))  # 0"),
        md("## 4. Visual Explanation"),
        code("fig, axes = plt.subplots(1, 2, figsize=(12, 5))\n\n# Vector as arrow\nax = axes[0]\nax.quiver(0, 0, 3, 4, angles='xy', scale_units='xy', scale=1, color='blue', label='v=(3,4)')\nax.set_xlim(-1, 5); ax.set_ylim(-1, 5)\nax.set_aspect('equal'); ax.grid(True, alpha=0.3)\nax.set_title('Vector as Arrow'); ax.legend()\n\n# Dot product and angle\nax = axes[1]\nfor vec, lbl, c in [([3,1],'a','blue'), ([1,2],'b','red')]: ax.quiver(0,0,*vec,color=c,angles='xy',scale_units='xy',scale=1,label=lbl)\ncos_theta = np.dot([3,1],[1,2]) / (np.linalg.norm([3,1])*np.linalg.norm([1,2]))\nax.set_xlim(-1,4); ax.set_ylim(-1,3); ax.set_aspect('equal'); ax.grid(True,alpha=0.3)\nax.set_title(f'Dot product: a·b = {np.dot([3,1],[1,2])}, cos θ = {cos_theta:.3f}'); ax.legend()\nplt.tight_layout(); plt.show()"),
        md("## 5. ML Connection\n\n- **Feature vector:** Each data sample is $\\mathbf{x}_i \\in \\mathbb{R}^d$\n- **Weight vector:** Model parameters $\\mathbf{w} \\in \\mathbb{R}^d$\n- **Prediction:** $\\hat{y} = \\mathbf{w}^T \\mathbf{x} + b$ (dot product!)\n- **Similarity:** Cosine similarity uses dot product of unit vectors (Module 01)"),
        md("## Exercise 1\n\nCompute the angle (in degrees) between vectors $\\mathbf{a} = [1, 2, 2]$ and $\\mathbf{b} = [2, 1, -2]$ using:\n$$\\cos\\theta = \\frac{\\mathbf{a} \\cdot \\mathbf{b}}{\\|\\mathbf{a}\\| \\|\\mathbf{b}\\|}$$"),
        code("# YOUR CODE HERE\n"),
        std_footer("01", "Vectors are the fundamental data structure of ML. Dot product connects geometry to prediction.", "02_Matrices.ipynb"),
    ])

    save("02_Matrices.ipynb", [
        md(HEADER.format(part="A — Linear Algebra", dur="2 hours",
            objs="1. Define matrices and index notation\n2. Transpose, symmetric, identity, diagonal matrices\n3. Matrix-vector multiplication\n4. Connect matrices to ML datasets")),
        md("## 1. Definition\n\n**Matrix:** $A \\in \\mathbb{R}^{m \\times n}$ has $m$ rows and $n$ columns.\n\n$$A = \\begin{bmatrix} a_{11} & a_{12} & \\cdots & a_{1n} \\\\ a_{21} & a_{22} & \\cdots & a_{2n} \\\\ \\vdots & \\vdots & \\ddots & \\vdots \\\\ a_{m1} & a_{m2} & \\cdots & a_{mn} \\end{bmatrix}$$\n\nElement $a_{ij}$ = row $i$, column $j$.\n\n**ML convention:** $X \\in \\mathbb{R}^{n \\times d}$ = $n$ samples, $d$ features (each row is a sample)."),
        IMPORTS,
        code("A = np.array([[1, 2, 3], [4, 5, 6]])\nprint('Shape (m,n):', A.shape)\nprint('Element a_12 (row 0, col 1):', A[0, 1])\nprint('Row 1:', A[1])\nprint('Column 0:', A[:, 0])"),
        md("## 2. Special Matrices\n\n**Transpose:** $(A^T)_{ij} = a_{ji}$\n\n**Symmetric:** $A = A^T$ (covariance matrices are symmetric)\n\n**Identity:** $I_n$ — ones on diagonal, zeros elsewhere. $AI = IA = A$\n\n**Diagonal:** Non-zero only on diagonal"),
        code("A = np.array([[1, 2], [3, 4]])\nprint('A:\\n', A)\nprint('A^T:\\n', A.T)\n\nI = np.eye(3)\nprint('Identity:\\n', I)\n\n# Symmetric matrix (covariance example)\nC = np.array([[2.0, 0.5], [0.5, 1.5]])\nprint('Symmetric?', np.allclose(C, C.T))"),
        md("## 3. Matrix-Vector Multiplication\n\nFor $A \\in \\mathbb{R}^{m \\times n}$ and $\\mathbf{x} \\in \\mathbb{R}^n$:\n\n$$(A\\mathbf{x})_i = \\sum_{j=1}^{n} a_{ij} x_j$$\n\nResult is a vector in $\\mathbb{R}^m$. Each output element is a **dot product** of a row of $A$ with $\\mathbf{x}$."),
        code("# Dataset: 3 houses, 2 features (area, bedrooms)\nX = np.array([[1500, 3], [2000, 4], [1200, 2]])  # (3, 2)\nw = np.array([100, 50000])  # weight per sqft, base per bedroom\nb = 10000\n\npredictions = X @ w + b  # matrix-vector product + scalar\nprint('Predictions:', predictions)"),
        md("## 4. Visual: Matrix as Linear Transformation\n\nMultiplying by a matrix rotates/scales space."),
        code("fig, ax = plt.subplots(figsize=(6, 6))\nA = np.array([[2, 0.5], [0.5, 1]])\nvectors = np.array([[1,0],[0,1],[1,1]])\nfor v in vectors:\n    t = A @ v\n    ax.quiver(0,0,*v, color='blue', alpha=0.5, scale=1, scale_units='xy', angles='xy')\n    ax.quiver(0,0,*t, color='red', scale=1, scale_units='xy', angles='xy')\nax.set_xlim(-1,3); ax.set_ylim(-1,3); ax.set_aspect('equal'); ax.grid(True,alpha=0.3)\nax.set_title('Blue: original, Red: after A transform'); plt.show()"),
        md("## Exercise 2\n\nGiven $X = \\begin{bmatrix} 1 & 2 \\\\ 3 & 4 \\\\ 5 & 6 \\end{bmatrix}$ and $\\mathbf{w} = \\begin{bmatrix} 1 \\\\ 1 \\end{bmatrix}$, compute $X\\mathbf{w}$ by hand and verify with NumPy."),
        code("# YOUR CODE HERE\n"),
        std_footer("02", "Matrices store datasets and transformations. ML data is always a matrix.", "03_Matrix_Multiplication.ipynb"),
    ])

    save("03_Matrix_Multiplication.ipynb", [
        md(HEADER.format(part="A — Linear Algebra", dur="2.5 hours",
            objs="1. Derive matrix multiplication formula\n2. Understand shape compatibility rules\n3. Prove associativity and non-commutativity\n4. Derive the normal equation for linear regression")),
        md("## 1. Matrix Multiplication Definition\n\nFor $A \\in \\mathbb{R}^{m \\times n}$ and $B \\in \\mathbb{R}^{n \\times p}$:\n\n$$(AB)_{ij} = \\sum_{k=1}^{n} a_{ik} b_{kj}$$\n\n**Shape rule:** $(m \\times n)(n \\times p) = (m \\times p)$\n\nElement $(i,j)$ of product = dot product of row $i$ of $A$ with column $j$ of $B$."),
        IMPORTS,
        code("A = np.array([[1, 2], [3, 4]])  # (2,2)\nB = np.array([[5, 6], [7, 8]])  # (2,2)\nC = A @ B\nprint('A @ B =\\n', C)\n# c_11 = 1*5 + 2*7 = 19\nprint('Manual c_11:', 1*5 + 2*7)"),
        md("## 2. Properties\n\n1. **Associative:** $(AB)C = A(BC)$\n2. **NOT commutative:** $AB \\neq BA$ in general\n3. **Identity:** $AI = IA = A$\n4. **Transpose:** $(AB)^T = B^T A^T$"),
        code("A = np.array([[1, 2], [3, 4]])\nB = np.array([[0, 1], [1, 0]])\nprint('AB =\\n', A @ B)\nprint('BA =\\n', B @ A)\nprint('Equal?', np.allclose(A @ B, B @ A))"),
        md("## 3. Derivation: Normal Equation for Linear Regression\n\n**Problem:** Given $X \\in \\mathbb{R}^{n \\times d}$ and $\\mathbf{y} \\in \\mathbb{R}^n$, find $\\mathbf{w}$ minimizing:\n$$L(\\mathbf{w}) = \\|X\\mathbf{w} - \\mathbf{y}\\|^2 = (X\\mathbf{w} - \\mathbf{y})^T(X\\mathbf{w} - \\mathbf{y})$$\n\n**Step 1:** Expand:\n$$L = \\mathbf{w}^T X^T X \\mathbf{w} - 2\\mathbf{y}^T X \\mathbf{w} + \\mathbf{y}^T \\mathbf{y}$$\n\n**Step 2:** Take gradient (Module 02 Notebook 09):\n$$\\nabla_{\\mathbf{w}} L = 2X^T X \\mathbf{w} - 2X^T \\mathbf{y}$$\n\n**Step 3:** Set to zero:\n$$X^T X \\mathbf{w} = X^T \\mathbf{y}$$\n\n**Step 4:** Normal equation:\n$$\\boxed{\\hat{\\mathbf{w}} = (X^T X)^{-1} X^T \\mathbf{y}}$$"),
        code("# Verify normal equation on house price data\nfrom pathlib import Path\nimport pandas as pd\n\nhouse = pd.read_csv(Path('../../houseprice.csv'))\nX = house[['area']].values\ny = house['price'].values\n\n# Add bias column of ones\nX_b = np.column_stack([np.ones(len(X)), X])\n\n# Normal equation\nw_hat = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y\nprint(f'Intercept: {w_hat[0]:,.2f}')\nprint(f'Slope (area): {w_hat[1]:.2f}')\n\n# Compare with sklearn\nfrom sklearn.linear_model import LinearRegression\nm = LinearRegression().fit(X, y)\nprint(f'sklearn intercept: {m.intercept_:,.2f}, slope: {m.coef_[0]:.2f}')"),
        md("## Exercise 3\n\nProve that $(AB)^T = B^T A^T$ by computing both sides for random 2×3 and 3×2 matrices."),
        code("# YOUR CODE HERE\n"),
        std_footer("03", "Matrix multiplication powers linear regression, neural networks, and attention.", "04_Matrix_Inverse_and_Determinant.ipynb"),
    ])

    save("04_Matrix_Inverse_and_Determinant.ipynb", [
        md(HEADER.format(part="A — Linear Algebra", dur="2 hours",
            objs="1. Define determinant for 2×2 and general n×n\n2. Understand when a matrix is invertible\n3. Compute inverse and solve linear systems\n4. Connect to normal equation stability")),
        md("## 1. Determinant (2×2)\n\nFor $A = \\begin{bmatrix} a & b \\\\ c & d \\end{bmatrix}$:\n$$\\det(A) = ad - bc$$\n\nGeometric meaning: absolute value = area scaling factor of the linear transformation.\n\n$\\det(A) = 0$ → matrix is **singular** (not invertible)."),
        md("## 2. Matrix Inverse\n\n$A^{-1}$ satisfies: $AA^{-1} = A^{-1}A = I$\n\nExists **if and only if** $\\det(A) \\neq 0$.\n\n**2×2 formula:**\n$$A^{-1} = \\frac{1}{ad-bc} \\begin{bmatrix} d & -b \\\\ -c & a \\end{bmatrix}$$"),
        IMPORTS,
        code("A = np.array([[4.0, 7.0], [2.0, 6.0]])\ndet = np.linalg.det(A)\nA_inv = np.linalg.inv(A)\nprint(f'det(A) = {det}')\nprint(f'A @ A_inv =\\n{A @ A_inv}')\n\n# Solve Ax = b\nb = np.array([1, 2])\nx = A_inv @ b\nprint(f'Solution x = {x}')\nprint(f'Verify Ax = {A @ x}')"),
        md("## 3. Why Normal Equation Can Fail\n\nIf $X^T X$ is singular (columns of $X$ are linearly dependent), $(X^T X)^{-1}$ does not exist.\n\n**Solution in practice:** Use gradient descent or add regularization (Ridge): $(X^T X + \\lambda I)^{-1} X^T y$"),
        md("## Exercise 4\n\nCompute the inverse of $\\begin{bmatrix} 2 & 1 \\\\ 5 & 3 \\end{bmatrix}$ by hand using the 2×2 formula, then verify with NumPy."),
        code("# YOUR CODE HERE\n"),
        std_footer("04", "Inverse matrices solve linear systems. Singularity causes ML training failures.", "05_Rank_and_Linear_Independence.ipynb"),
    ])

    save("05_Rank_and_Linear_Independence.ipynb", [
        md(HEADER.format(part="A — Linear Algebra", dur="2 hours",
            objs="1. Define linear independence\n2. Compute rank of a matrix\n3. Understand null space and column space\n4. Connect rank to feature redundancy in ML")),
        md("## 1. Linear Independence\n\nVectors $\\mathbf{v}_1, \\ldots, \\mathbf{v}_k$ are **linearly independent** if:\n$$c_1\\mathbf{v}_1 + c_2\\mathbf{v}_2 + \\cdots + c_k\\mathbf{v}_k = \\mathbf{0} \\implies c_1 = c_2 = \\cdots = c_k = 0$$\n\nIf one vector can be written as a combination of others → **linearly dependent** (redundant feature)."),
        md("## 2. Rank\n\n**Rank** = number of linearly independent columns (or rows) = dimension of column space.\n\n$\\text{rank}(A) \\leq \\min(m, n)$ for $A \\in \\mathbb{R}^{m \\times n}$.\n\nFull rank: $\\text{rank}(A) = \\min(m, n)$."),
        IMPORTS,
        code("A_full = np.array([[1, 0], [0, 1], [1, 1]])\nA_dep = np.array([[1, 2], [2, 4], [3, 6]])  # col2 = 2*col1\n\nprint('Full rank matrix rank:', np.linalg.matrix_rank(A_full))\nprint('Dependent matrix rank:', np.linalg.matrix_rank(A_dep))"),
        md("## 3. ML Connection\n\n- **Redundant features** → rank-deficient $X^T X$ → normal equation fails\n- **PCA (Module 03):** Projects to lower rank while preserving variance\n- **Attention rank:** Low-rank approximations in transformers"),
        md("## Exercise 5\n\nFind which column of $\\begin{bmatrix} 1 & 2 & 3 \\\\ 2 & 4 & 6 \\\\ 1 & 1 & 1 \\end{bmatrix}$ is redundant using `np.linalg.matrix_rank`."),
        code("# YOUR CODE HERE\n"),
        std_footer("05", "Rank reveals redundancy. Multicollinearity breaks closed-form regression.", "06_Eigenvalues_and_Eigenvectors.ipynb"),
    ])

    save("06_Eigenvalues_and_Eigenvectors.ipynb", [
        md(HEADER.format(part="A — Linear Algebra", dur="2.5 hours",
            objs="1. Define eigenvalues and eigenvectors\n2. Compute them numerically\n3. Understand diagonalization\n4. Derive PCA from eigendecomposition")),
        md("## 1. Definition\n\nFor square matrix $A \\in \\mathbb{R}^{n \\times n}$, if:\n$$A\\mathbf{v} = \\lambda \\mathbf{v}$$\n\nfor non-zero vector $\\mathbf{v}$ and scalar $\\lambda$, then:\n- $\\lambda$ = **eigenvalue**\n- $\\mathbf{v}$ = **eigenvector**\n\nIntuition: eigenvectors are directions unchanged by $A$ (only scaled by $\\lambda$)."),
        md("## 2. Characteristic Equation\n\n$\\det(A - \\lambda I) = 0$ → polynomial whose roots are eigenvalues."),
        IMPORTS,
        code("A = np.array([[4, 1], [2, 3]])\neigenvalues, eigenvectors = np.linalg.eig(A)\nprint('Eigenvalues:', eigenvalues)\nprint('Eigenvectors (columns):\\n', eigenvectors)\n\n# Verify Av = λv\nfor i in range(len(eigenvalues)):\n    lam, v = eigenvalues[i], eigenvectors[:, i]\n    Av = A @ v\n    lv = lam * v\n    print(f'λ={lam:.2f}: ||Av - λv|| = {np.linalg.norm(Av - lv):.2e}')"),
        md("## 3. Derivation: PCA from Eigendecomposition\n\nGiven centered data matrix $X_c$ (n × d):\n\n1. Compute **covariance matrix:** $C = \\frac{1}{n} X_c^T X_c$ (symmetric d × d)\n2. Eigendecompose: $C = V \\Lambda V^T$\n3. Eigenvectors $V$ = principal directions\n4. Eigenvalues $\\Lambda$ = variance along each direction\n5. Project: $X_{\\text{PCA}} = X_c V_k$ (keep top k eigenvectors)\n\nThis is exactly your `Day - 21 Plant Iris Clustering Using PCA`."),
        code("from sklearn.datasets import load_iris\n\niris = load_iris()\nX = iris.data\nX_c = X - X.mean(axis=0)\n\n# Covariance and eigendecomposition\nC = (X_c.T @ X_c) / len(X_c)\neigenvalues, eigenvectors = np.linalg.eigh(C)\n\n# Sort descending\nidx = np.argsort(eigenvalues)[::-1]\neigenvalues = eigenvalues[idx]\neigenvectors = eigenvectors[:, idx]\n\n# Project to 2D\nX_pca = X_c @ eigenvectors[:, :2]\n\nfig, ax = plt.subplots()\nfor label, color in zip([0,1,2], ['r','g','b']):\n    mask = iris.target == label\n    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, label=iris.target_names[label], alpha=0.7)\nax.set_xlabel(f'PC1 ({eigenvalues[0]/eigenvalues.sum():.1%} var)')\nax.set_ylabel(f'PC2 ({eigenvalues[1]/eigenvalues.sum():.1%} var)')\nax.legend(); ax.set_title('PCA via Eigendecomposition (Iris)'); plt.show()"),
        md("## Exercise 6\n\nCompute eigenvalues of $\\begin{bmatrix} 3 & 1 \\\\ 1 & 3 \\end{bmatrix}$ by hand using $\\det(A - \\lambda I) = 0$, then verify with NumPy."),
        code("# YOUR CODE HERE\n"),
        std_footer("06", "Eigenvalues power PCA, spectral clustering, and graph neural networks.", "07_Orthogonality_and_Projection.ipynb"),
    ])

    save("07_Orthogonality_and_Projection.ipynb", [
        md(HEADER.format(part="A — Linear Algebra", dur="2 hours",
            objs="1. Define orthogonal vectors and matrices\n2. Project vectors onto subspaces\n3. Understand Gram-Schmidt process\n4. Connect projection to linear regression geometry")),
        md("## 1. Orthogonality\n\nVectors $\\mathbf{u}, \\mathbf{v}$ are **orthogonal** if $\\mathbf{u}^T \\mathbf{v} = 0$.\n\n**Orthogonal matrix:** $Q^T Q = I$ (columns are orthonormal).\n\nProperty: $\\|Q\\mathbf{x}\\| = \\|\\mathbf{x}\\|$ (preserves length — used in QR decomposition)."),
        md("## 2. Projection\n\nProjection of $\\mathbf{b}$ onto vector $\\mathbf{a}$:\n$$\\text{proj}_{\\mathbf{a}} \\mathbf{b} = \\frac{\\mathbf{a}^T \\mathbf{b}}{\\mathbf{a}^T \\mathbf{a}} \\mathbf{a}$$\n\n**ML:** Linear regression prediction = projection of $\\mathbf{y}$ onto column space of $X$."),
        IMPORTS,
        code("a = np.array([1.0, 1.0])\nb = np.array([2.0, 0.0])\nproj = (a @ b) / (a @ a) * a\nprint(f'Projection of b onto a: {proj}')\n\n# Residual is orthogonal to a\nresidual = b - proj\nprint(f'a · residual = {a @ residual:.2e} (should be ~0)')"),
        md("## 3. Gram-Schmidt (Conceptual)\n\nOrthogonalizes a set of vectors:\n1. $\\mathbf{u}_1 = \\mathbf{v}_1$\n2. $\\mathbf{u}_2 = \\mathbf{v}_2 - \\text{proj}_{\\mathbf{u}_1}\\mathbf{v}_2$\n3. Continue for remaining vectors\n\nUsed in QR decomposition — numerically stable alternative to normal equation."),
        code("fig, ax = plt.subplots(figsize=(6, 6))\nax.quiver(0,0,*a, color='blue', scale=1, scale_units='xy', angles='xy', label='a')\nax.quiver(0,0,*b, color='green', scale=1, scale_units='xy', angles='xy', label='b')\nax.quiver(0,0,*proj, color='red', scale=1, scale_units='xy', angles='xy', label='projection')\nax.plot([proj[0], b[0]], [proj[1], b[1]], 'k--', alpha=0.5, label='residual')\nax.set_xlim(-0.5, 2.5); ax.set_ylim(-0.5, 1.5); ax.set_aspect('equal'); ax.legend(); ax.grid(True,alpha=0.3)\nax.set_title('Vector Projection'); plt.show()"),
        md("## Part A Assignment Preview\n\nDerive the normal equation by setting the gradient of $\\|X\\mathbf{w} - \\mathbf{y}\\|^2$ to zero. Full assignment in `exercises/README.md`."),
        md("## Exercise 7\n\nProject vector $\\mathbf{b} = [3, 4]$ onto $\\mathbf{a} = [1, 0]$. What is the residual?"),
        code("# YOUR CODE HERE\n"),
        std_footer("07", "Orthogonality and projection underpin regression geometry and QR decomposition.", "08_Functions_and_Derivatives.ipynb"),
    ])


def build_part_b():
    save("08_Functions_and_Derivatives.ipynb", [
        md(HEADER.format(part="B — Calculus", dur="2 hours",
            objs="1. Understand limits and continuity\n2. Apply power rule, product rule, chain rule\n3. Compute derivatives numerically and symbolically\n4. Connect derivatives to loss minimization")),
        md("## 1. Derivative Definition\n\n$$f'(x) = \\lim_{h \\to 0} \\frac{f(x+h) - f(x)}{h}$$\n\nGeometric meaning: **slope of tangent line** at point $x$.\n\nIn ML: derivative of loss w.r.t. parameter = direction to increase loss. We move **opposite** to minimize."),
        md("## 2. Rules\n\n| Rule | Formula |\n|------|--------|\n| Power | $\\frac{d}{dx}x^n = nx^{n-1}$ |\n| Constant | $\\frac{d}{dx}c = 0$ |\n| Sum | $\\frac{d}{dx}[f+g] = f' + g'$ |\n| Product | $\\frac{d}{dx}[fg] = f'g + fg'$ |\n| **Chain** | $\\frac{d}{dx}f(g(x)) = f'(g(x)) \\cdot g'(x)$ |\n\n**Chain rule is the foundation of backpropagation (Module 05).**"),
        IMPORTS,
        code("def numerical_derivative(f, x, h=1e-7):\n    return (f(x + h) - f(x - h)) / (2 * h)\n\nf = lambda x: x**3 - 2*x**2 + x\nx0 = 2.0\nexact = 3*x0**2 - 4*x0 + 1  # power rule\napprox = numerical_derivative(f, x0)\nprint(f'f(x) = x³ - 2x² + x')\nprint(f\"f'({x0}) exact: {exact}\")\nprint(f\"f'({x0}) numerical: {approx:.6f}\")"),
        code("# Visualize function and tangent\nx = np.linspace(-1, 4, 200)\ny = x**3 - 2*x**2 + x\nx0 = 2.0; slope = 3*x0**2 - 4*x0 + 1\ntangent = slope * (x - x0) + (x0**3 - 2*x0**2 + x0)\n\nplt.plot(x, y, label='f(x)')\nplt.plot(x, tangent, '--', label=f'Tangent at x={x0}, slope={slope}')\nplt.axhline(0, color='k', linewidth=0.5); plt.axvline(0, color='k', linewidth=0.5)\nplt.legend(); plt.title('Function and Tangent Line'); plt.show()"),
        md("## 3. Chain Rule Example (Backprop Preview)\n\nLet $L = (y - \\hat{y})^2$ where $\\hat{y} = wx + b$.\n\n$\\frac{dL}{dw} = \\frac{dL}{d\\hat{y}} \\cdot \\frac{d\\hat{y}}{dw} = 2(\\hat{y} - y) \\cdot x$\n\nThis is the gradient for linear regression weight update."),
        md("## Exercise 8\n\nCompute $\\frac{d}{dx}(x^2 + 3x)(\\sin x)$ using the product rule. Verify numerically at $x = 1$."),
        code("# YOUR CODE HERE\n"),
        std_footer("08", "Derivatives tell us how to adjust parameters. Chain rule enables deep learning.", "09_Partial_Derivatives_and_Gradient.ipynb"),
    ])

    save("09_Partial_Derivatives_and_Gradient.ipynb", [
        md(HEADER.format(part="B — Calculus", dur="2.5 hours",
            objs="1. Compute partial derivatives\n2. Build the gradient vector\n3. Derive gradient of MSE loss\n4. Visualize gradient as direction of steepest ascent")),
        md("## 1. Partial Derivatives\n\nFor $f(x, y)$:\n$$\\frac{\\partial f}{\\partial x} = \\lim_{h \\to 0} \\frac{f(x+h, y) - f(x, y)}{h}$$\n\nTreat other variables as constants."),
        md("## 2. Gradient\n\n$$\\nabla f = \\begin{bmatrix} \\frac{\\partial f}{\\partial x_1} \\\\ \\vdots \\\\ \\frac{\\partial f}{\\partial x_n} \\end{bmatrix}$$\n\nPoints in direction of **steepest ascent**. Negative gradient = steepest descent (optimization)."),
        IMPORTS,
        code("def f(x, y): return x**2 + 3*y**2\n\ndef grad_f(x, y):\n    return np.array([2*x, 6*y])\n\nprint('∇f(1, 2) =', grad_f(1, 2))  # [2, 12]"),
        md("## 3. Derivation: Gradient of MSE\n\n$$L(\\mathbf{w}) = \\frac{1}{n}\\|X\\mathbf{w} - \\mathbf{y}\\|^2 = \\frac{1}{n}(X\\mathbf{w} - \\mathbf{y})^T(X\\mathbf{w} - \\mathbf{y})$$\n\n$$\\nabla_{\\mathbf{w}} L = \\frac{2}{n} X^T(X\\mathbf{w} - \\mathbf{y})$$\n\nAt optimum: $X^T(X\\mathbf{w} - \\mathbf{y}) = 0$ → same as normal equation."),
        code("# Verify gradient numerically\nn, d = 50, 3\nX = rng.normal(0, 1, (n, d))\ny = rng.normal(0, 1, n)\nw = rng.normal(0, 1, d)\n\n# Analytical gradient\nresidual = X @ w - y\ngrad_analytical = (2 / n) * X.T @ residual\n\n# Numerical gradient (per coordinate)\ngrad_numerical = np.zeros(d)\neps = 1e-7\nfor i in range(d):\n    w_plus, w_minus = w.copy(), w.copy()\n    w_plus[i] += eps; w_minus[i] -= eps\n    L_plus = np.mean((X @ w_plus - y)**2)\n    L_minus = np.mean((X @ w_minus - y)**2)\n    grad_numerical[i] = (L_plus - L_minus) / (2 * eps)\n\nprint('Analytical:', grad_analytical)\nprint('Numerical: ', grad_numerical)\nprint('Max error:', np.max(np.abs(grad_analytical - grad_numerical)))"),
        code("# Visualize gradient field\nx = np.linspace(-2, 2, 20)\ny = np.linspace(-2, 2, 20)\nXg, Yg = np.meshgrid(x, y)\nZ = Xg**2 + 3*Yg**2\n\nplt.contour(Xg, Yg, Z, levels=20)\nplt.quiver(Xg, Yg, 2*Xg, 6*Yg, alpha=0.6)\nplt.xlabel('x'); plt.ylabel('y'); plt.title('Gradient Field of f(x,y) = x² + 3y²'); plt.show()"),
        md("## Exercise 9\n\nDerive $\\nabla_{\\mathbf{w}} \\|X\\mathbf{w} - \\mathbf{y}\\|^2$ step by step on paper. Then verify with random X, y, w."),
        code("# YOUR CODE HERE\n"),
        std_footer("09", "The gradient is the engine of all ML optimization.", "10_Jacobian_and_Hessian.ipynb"),
    ])

    save("10_Jacobian_and_Hessian.ipynb", [
        md(HEADER.format(part="B — Calculus", dur="2 hours",
            objs="1. Define Jacobian matrix for vector-valued functions\n2. Define Hessian matrix for second derivatives\n3. Interpret Hessian for curvature and optimization\n4. Preview Newton's method")),
        md("## 1. Jacobian\n\nFor $\\mathbf{f}: \\mathbb{R}^n \\to \\mathbb{R}^m$:\n$$J_{ij} = \\frac{\\partial f_i}{\\partial x_j}$$\n\nShape: $(m \\times n)$. Generalizes gradient to vector outputs.\n\n**Backpropagation** computes Jacobians layer by layer (Module 05)."),
        md("## 2. Hessian\n\nFor scalar $f: \\mathbb{R}^n \\to \\mathbb{R}$:\n$$H_{ij} = \\frac{\\partial^2 f}{\\partial x_i \\partial x_j}$$\n\nSymmetric matrix. Describes **curvature**:\n- Positive definite → convex (bowl shape, one minimum)\n- Indefinite → saddle point\n- Negative definite → concave"),
        IMPORTS,
        code("# Hessian of f(x,y) = x² + 3y²\n# H = [[2, 0], [0, 6]] — positive definite (convex)\nH = np.array([[2, 0], [0, 6]])\neigenvalues = np.linalg.eigvalsh(H)\nprint('Hessian eigenvalues:', eigenvalues)\nprint('Convex?', np.all(eigenvalues > 0))"),
        code("# Saddle point: f(x,y) = x² - y²\nx = np.linspace(-2, 2, 100)\ny = np.linspace(-2, 2, 100)\nX, Y = np.meshgrid(x, y)\nZ = X**2 - Y**2\n\nfig = plt.figure(figsize=(8, 5))\nax = fig.add_subplot(111, projection='3d')\nax.plot_surface(X, Y, Z, cmap=cm.coolwarm, alpha=0.8)\nax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('f(x,y)')\nax.set_title('Saddle Point: f(x,y) = x² - y²'); plt.show()"),
        md("## Exercise 10\n\nCompute the Hessian of $f(x, y) = x^3 + xy^2$ at point $(1, 2)$."),
        code("# YOUR CODE HERE\n"),
        std_footer("10", "Jacobian generalizes gradients; Hessian reveals optimization landscape curvature.", "11_Optimization_Landscapes.ipynb"),
    ])

    save("11_Optimization_Landscapes.ipynb", [
        md(HEADER.format(part="B — Calculus", dur="2 hours",
            objs="1. Define convex and non-convex functions\n2. Identify local vs global minima\n3. Understand saddle points in high dimensions\n4. Visualize loss landscapes of ML models")),
        md("## 1. Convexity\n\n$f$ is **convex** if for all $x, y$ and $\\lambda \\in [0,1]$:\n$$f(\\lambda x + (1-\\lambda)y) \\leq \\lambda f(x) + (1-\\lambda)f(y)$$\n\nConvex → any local minimum is global minimum. Linear regression MSE is convex."),
        md("## 2. Local vs Global Minima\n\nNeural networks have **non-convex** loss landscapes with many local minima and saddle points. In practice, SGD finds good enough solutions."),
        IMPORTS,
        code("# Non-convex function\nx = np.linspace(-3, 3, 500)\nf = lambda x: x**4 - 4*x**2 + x\nplt.plot(x, f(x)); plt.axhline(0, color='k', linewidth=0.5)\nplt.xlabel('x'); plt.ylabel('f(x)'); plt.title('Non-convex: multiple local minima'); plt.show()"),
        code("# 2D loss landscape\nw1 = np.linspace(-2, 2, 100)\nw2 = np.linspace(-2, 2, 100)\nW1, W2 = np.meshgrid(w1, w2)\nL = W1**2 + 5*W2**2  # convex bowl\n\nplt.contour(W1, W2, L, levels=30)\nplt.xlabel('w1'); plt.ylabel('w2'); plt.title('Convex Loss Landscape (MSE-like)'); plt.show()"),
        md("## ML Insight\n\n- **Linear/logistic regression:** Convex → guaranteed global optimum\n- **Neural networks:** Non-convex → need good initialization, learning rate, and luck\n- **Saddle points** are more common than local minima in high dimensions"),
        std_footer("11", "Understanding loss landscape geometry explains why training succeeds or fails.", "12_Probability_Foundations.ipynb"),
    ])


def build_part_c():
    save("12_Probability_Foundations.ipynb", [
        md(HEADER.format(part="C — Probability & Statistics", dur="2 hours",
            objs="1. Define sample spaces and events\n2. Apply conditional probability\n3. Derive and apply Bayes theorem\n4. Connect Bayes theorem to Naive Bayes classifier")),
        md("## 1. Probability Axioms\n\n1. $P(A) \\geq 0$\n2. $P(\\Omega) = 1$ (sample space)\n3. $P(A \\cup B) = P(A) + P(B)$ if mutually exclusive"),
        md("## 2. Conditional Probability\n\n$$P(A|B) = \\frac{P(A \\cap B)}{P(B)} \\quad \\text{if } P(B) > 0$$\n\n\"Probability of A given B has occurred.\""),
        md("## 3. Bayes Theorem\n\n$$\\boxed{P(A|B) = \\frac{P(B|A) \\cdot P(A)}{P(B)}}$$\n\n- $P(A)$ = **prior**\n- $P(B|A)$ = **likelihood**\n- $P(A|B)$ = **posterior**\n- $P(B)$ = **evidence**"),
        IMPORTS,
        code("# Medical test example\n# Disease prevalence 1%, test sensitivity 99%, specificity 95%\nP_D = 0.01  # prior\nP_pos_given_D = 0.99  # sensitivity\nP_pos_given_noD = 0.05  # false positive rate\n\nP_pos = P_pos_given_D * P_D + P_pos_given_noD * (1 - P_D)\nP_D_given_pos = P_pos_given_D * P_D / P_pos\n\nprint(f'P(disease | positive test) = {P_D_given_pos:.1%}')\nprint('Despite 99% sensitivity, only ~16% chance of disease!')"),
        md("## 4. Naive Bayes Preview (Module 03)\n\n$$P(y | \\mathbf{x}) \\propto P(y) \\prod_{j=1}^{d} P(x_j | y)$$\n\n\"Naive\" = features conditionally independent given class. Your `Day - 6 Titanic` uses GaussianNB."),
        md("## Exercise 11\n\nA spam filter: 30% of emails are spam. 90% of spam contains \"free\", 10% of ham contains \"free\". If an email contains \"free\", what is P(spam | \"free\")?"),
        code("# YOUR CODE HERE\n"),
        std_footer("12", "Bayes theorem is the foundation of probabilistic ML.", "13_Random_Variables.ipynb"),
    ])

    save("13_Random_Variables.ipynb", [
        md(HEADER.format(part="C — Probability & Statistics", dur="2 hours",
            objs="1. Distinguish discrete and continuous random variables\n2. Define PMF, PDF, and CDF\n3. Compute expected value and variance\n4. Understand joint and marginal distributions")),
        md("## 1. Random Variables\n\n**Discrete:** Takes countable values (e.g., coin flip: 0 or 1)\n\n**Continuous:** Takes real values (e.g., house price)\n\n**PMF (discrete):** $P(X = x)$\n\n**PDF (continuous):** $f(x)$ where $P(a \\leq X \\leq b) = \\int_a^b f(x) dx$ and $\\int f(x)dx = 1$\n\n**CDF:** $F(x) = P(X \\leq x)$"),
        md("## 2. Expected Value and Variance\n\n$$E[X] = \\sum_x x \\cdot P(X=x) \\quad \\text{(discrete)}$$\n$$E[X] = \\int x f(x) dx \\quad \\text{(continuous)}$$\n\n$$\\text{Var}(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2$$\n\n$$\\sigma = \\sqrt{\\text{Var}(X)}$$"),
        IMPORTS,
        code("# Simulate coin flips\nflips = rng.integers(0, 2, size=10000)\nprint(f'Mean (→ E[X]): {flips.mean():.3f} (expect 0.5)')\nprint(f'Var: {flips.var():.3f} (expect 0.25)')\n\n# Continuous: sample from Normal\nsamples = rng.normal(5, 2, 10000)\nprint(f'\\nNormal(5,4): mean={samples.mean():.2f}, std={samples.std():.2f}')"),
        code("fig, axes = plt.subplots(1, 2, figsize=(12, 4))\naxes[0].hist(flips, bins=[-0.5,0.5,1.5], density=True, edgecolor='black')\naxes[0].set_title('Discrete: Bernoulli(0.5)'); axes[0].set_xlabel('x')\n\naxes[1].hist(samples, bins=50, density=True, alpha=0.7, label='Samples')\nx = np.linspace(samples.min(), samples.max(), 200)\nfrom scipy.stats import norm\naxes[1].plot(x, norm.pdf(x, 5, 2), 'r-', linewidth=2, label='True PDF')\naxes[1].set_title('Continuous: Normal(5, 4)'); axes[1].legend()\nplt.tight_layout(); plt.show()"),
        std_footer("13", "Random variables model uncertainty. PMF/PDF/CDF are the language of probabilistic ML.", "14_Distributions.ipynb"),
    ])

    save("14_Distributions.ipynb", [
        md(HEADER.format(part="C — Probability & Statistics", dur="2.5 hours",
            objs="1. Define Gaussian, Bernoulli, Binomial, Poisson distributions\n2. Write PDF/PMF and parameters for each\n3. Sample from each distribution\n4. Know when each applies in ML")),
        md("## Key Distributions\n\n| Distribution | PMF/PDF | Parameters | ML Use |\n|-------------|---------|------------|--------|\n| Bernoulli | $P(X=1)=p$ | $p$ | Binary classification |\n| Binomial | $\\binom{n}{k}p^k(1-p)^{n-k}$ | $n, p$ | k successes in n trials |\n| Gaussian | $\\frac{1}{\\sqrt{2\\pi\\sigma^2}}e^{-\\frac{(x-\\mu)^2}{2\\sigma^2}}$ | $\\mu, \\sigma^2$ | Continuous features, noise |\n| Poisson | $\\frac{\\lambda^k e^{-\\lambda}}{k!}$ | $\\lambda$ | Count data |"),
        IMPORTS,
        code("from scipy.stats import norm, bernoulli, binom, poisson\n\nx = np.linspace(-4, 4, 200)\nfig, axes = plt.subplots(2, 2, figsize=(12, 8))\n\naxes[0,0].plot(x, norm.pdf(x, 0, 1)); axes[0,0].set_title('Gaussian N(0,1)')\naxes[0,1].bar([0,1], [0.3, 0.7]); axes[0,1].set_title('Bernoulli(p=0.7)')\naxes[1,0].bar(range(11), binom.pmf(range(11), 10, 0.3)); axes[1,0].set_title('Binomial(n=10, p=0.3)')\naxes[1,1].bar(range(15), poisson.pmf(range(15), 3)); axes[1,1].set_title('Poisson(λ=3)')\nplt.tight_layout(); plt.show()"),
        md("## ML Connections\n\n- **Gaussian:** Continuous features in Naive Bayes (your Day - 6), weight initialization\n- **Bernoulli:** Binary cross-entropy loss (Module 02 Notebook 22)\n- **Poisson:** Count targets (e.g., event counts)\n- **Gaussian noise:** Assumption in linear regression"),
        std_footer("14", "Knowing distributions lets you choose correct models and loss functions.", "15_Maximum_Likelihood_Estimation.ipynb"),
    ])

    save("15_Maximum_Likelihood_Estimation.ipynb", [
        md(HEADER.format(part="C — Probability & Statistics", dur="2.5 hours",
            objs="1. Define likelihood and log-likelihood\n2. Derive MLE for Gaussian mean and variance\n3. Derive MLE for Bernoulli (→ logistic regression)\n4. Connect MLE to loss functions in ML")),
        md("## 1. Likelihood\n\nGiven data $\\mathbf{x} = (x_1, \\ldots, x_n)$ and parameter $\\theta$:\n$$L(\\theta) = P(\\mathbf{x} | \\theta) = \\prod_{i=1}^{n} P(x_i | \\theta)$$\n\n**MLE:** $\\hat{\\theta} = \\arg\\max_\\theta L(\\theta)$\n\n**Log-likelihood:** $\\ell(\\theta) = \\log L(\\theta) = \\sum_i \\log P(x_i | \\theta)$ (easier to maximize)."),
        md("## 2. Derivation: MLE for Gaussian Mean\n\nFor $x_i \\sim \\mathcal{N}(\\mu, \\sigma^2)$ with known $\\sigma^2$:\n\n$$\\ell(\\mu) = -\\frac{1}{2\\sigma^2}\\sum_i (x_i - \\mu)^2 + \\text{const}$$\n\n$$\\frac{d\\ell}{d\\mu} = \\frac{1}{\\sigma^2}\\sum_i (x_i - \\mu) = 0$$\n\n$$\\boxed{\\hat{\\mu} = \\frac{1}{n}\\sum_{i=1}^{n} x_i}$$\n\n**The sample mean is the MLE for Gaussian mean!**"),
        md("## 3. MLE for Bernoulli → Cross-Entropy\n\nFor $y_i \\in \\{0, 1\\}$ with $P(y=1) = p$:\n\n$$\\ell(p) = \\sum_i [y_i \\log p + (1-y_i)\\log(1-p)]$$\n\nMaximizing log-likelihood = **minimizing cross-entropy loss** (Module 02 Notebook 22)."),
        IMPORTS,
        code("data = rng.normal(5, 2, 1000)\nmu_mle = data.mean()\nsigma2_mle = data.var()  # MLE uses n denominator\nprint(f'MLE μ = {mu_mle:.3f} (sample mean)')\nprint(f'MLE σ² = {sigma2_mle:.3f} (sample variance)')\n\n# Bernoulli MLE\ny = rng.binomial(1, 0.7, 1000)\np_mle = y.mean()\nprint(f'\\nMLE p (Bernoulli) = {p_mle:.3f}')"),
        md("## Exercise 12\n\nDerive MLE for Gaussian variance $\\sigma^2$. Result: $\\hat{\\sigma}^2 = \\frac{1}{n}\\sum(x_i - \\hat{\\mu})^2$."),
        code("# Verify with simulation\n"),
        std_footer("15", "MLE connects probability to optimization. Most ML losses are negative log-likelihoods.", "16_Descriptive_Statistics.ipynb"),
    ])

    save("16_Descriptive_Statistics.ipynb", [
        md(HEADER.format(part="C — Probability & Statistics", dur="2 hours",
            objs="1. Compute mean, median, variance, standard deviation\n2. Define covariance and correlation\n3. Interpret correlation in feature analysis\n4. Build covariance matrix for PCA")),
        md("## Key Formulas\n\n**Mean:** $\\bar{x} = \\frac{1}{n}\\sum x_i$\n\n**Variance:** $s^2 = \\frac{1}{n-1}\\sum(x_i - \\bar{x})^2$\n\n**Covariance:** $\\text{Cov}(X,Y) = E[(X-E[X])(Y-E[Y])] = \\frac{1}{n-1}\\sum(x_i - \\bar{x})(y_i - \\bar{y})$\n\n**Correlation:** $\\rho = \\frac{\\text{Cov}(X,Y)}{\\sigma_X \\sigma_Y}$ (between -1 and 1)"),
        IMPORTS,
        code("from pathlib import Path\nimport pandas as pd\n\nhouse = pd.read_csv(Path('../../houseprice.csv'))\nprint(house.describe())\nprint('\\nCorrelation matrix:')\nprint(house.corr())"),
        code("plt.imshow(house.corr(), cmap='RdBu_r', vmin=-1, vmax=1)\nplt.colorbar(label='Correlation')\nplt.xticks(range(len(house.columns)), house.columns)\nplt.yticks(range(len(house.columns)), house.columns)\nplt.title('Feature Correlation (House Prices)'); plt.show()"),
        md("## ML Connection\n\n- High correlation between features → multicollinearity (Module 03)\n- Covariance matrix → PCA (Notebook 06)\n- Normalization uses mean and std (Module 01, water-bodies percentile scaling)"),
        std_footer("16", "Descriptive statistics guide EDA and feature engineering.", "17_Hypothesis_Testing.ipynb"),
    ])

    save("17_Hypothesis_Testing.ipynb", [
        md(HEADER.format(part="C — Probability & Statistics", dur="2 hours",
            objs="1. Formulate null and alternative hypotheses\n2. Understand p-values and significance levels\n3. Compute confidence intervals\n4. Apply t-test for model comparison")),
        md("## 1. Hypothesis Testing Framework\n\n1. **Null hypothesis $H_0$:** No effect (e.g., model A = model B)\n2. **Alternative $H_1$:** There is an effect\n3. **p-value:** Probability of observing data at least as extreme, assuming $H_0$ is true\n4. If p-value < α (typically 0.05), **reject $H_0$**"),
        md("## 2. Confidence Interval\n\n95% CI for mean: $\\bar{x} \\pm 1.96 \\frac{s}{\\sqrt{n}}$\n\n\"We are 95% confident the true mean lies in this interval.\""),
        IMPORTS,
        code("from scipy.stats import ttest_ind, ttest_rel\n\n# Compare two models' accuracy on same test set\nacc_model_a = rng.normal(0.85, 0.02, 30)\nacc_model_b = rng.normal(0.87, 0.02, 30)\n\nt_stat, p_value = ttest_rel(acc_model_a, acc_model_b)\nprint(f'Mean acc A: {acc_model_a.mean():.3f}')\nprint(f'Mean acc B: {acc_model_b.mean():.3f}')\nprint(f'Paired t-test p-value: {p_value:.4f}')\nprint(f'Significant at α=0.05?', p_value < 0.05)"),
        md("## ML Use\n\n- Compare two models on same test folds (paired t-test)\n- A/B testing in production\n- Determine if accuracy improvement is statistically significant"),
        std_footer("17", "Hypothesis testing validates whether model improvements are real or noise.", "18_Gradient_Descent.ipynb"),
    ])


def build_part_d():
    save("18_Gradient_Descent.ipynb", [
        md(HEADER.format(part="D — Optimization", dur="2.5 hours",
            objs="1. Derive the gradient descent update rule\n2. Implement GD for linear regression from scratch\n3. Understand learning rate effects\n4. Visualize convergence")),
        md("## 1. Update Rule\n\nTo minimize $L(\\mathbf{w})$:\n$$\\mathbf{w}_{t+1} = \\mathbf{w}_t - \\eta \\nabla L(\\mathbf{w}_t)$$\n\n- $\\eta$ = **learning rate** (step size)\n- $\\nabla L$ = gradient at current point\n- Move opposite to gradient (steepest descent)"),
        md("## 2. GD for Linear Regression\n\n$L(w) = \\frac{1}{n}\\sum (wx_i + b - y_i)^2$\n\n$\\frac{\\partial L}{\\partial w} = \\frac{2}{n}\\sum (wx_i + b - y_i) x_i$\n\n$\\frac{\\partial L}{\\partial b} = \\frac{2}{n}\\sum (wx_i + b - y_i)$"),
        IMPORTS,
        code("from pathlib import Path\nimport pandas as pd\n\nhouse = pd.read_csv(Path('../../houseprice.csv'))\nX_raw = house['area'].values\ny = house['price'].values\nX = (X_raw - X_raw.mean()) / X_raw.std()  # normalize\n\nw, b = 0.0, 0.0\nlr = 0.1\nn = len(X)\nlosses = []\n\nfor epoch in range(100):\n    y_pred = w * X + b\n    dw = (2/n) * np.sum((y_pred - y) * X)\n    db = (2/n) * np.sum(y_pred - y)\n    w -= lr * dw\n    b -= lr * db\n    loss = np.mean((y_pred - y)**2)\n    losses.append(loss)\n\nprint(f'w = {w:.2f}, b = {b:.2f}')\nprint(f'Final MSE: {losses[-1]:,.0f}')"),
        code("fig, axes = plt.subplots(1, 2, figsize=(12, 4))\naxes[0].plot(losses); axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('MSE'); axes[0].set_title('Loss Convergence')\n\naxes[1].scatter(X_raw, y, alpha=0.5, label='Data')\nx_line = np.linspace(X_raw.min(), X_raw.max(), 100)\nx_norm = (x_line - X_raw.mean()) / X_raw.std()\naxes[1].plot(x_line, w * x_norm + b, 'r-', linewidth=2, label='GD fit')\naxes[1].set_xlabel('Area'); axes[1].set_ylabel('Price'); axes[1].legend()\nplt.tight_layout(); plt.show()"),
        md("## 3. Learning Rate Effects\n\n- Too large → diverges (loss increases)\n- Too small → slow convergence\n- Just right → smooth decrease"),
        code("def run_gd(lr, epochs=50):\n    w, b = 0.0, 0.0\n    losses = []\n    for _ in range(epochs):\n        y_pred = w * X + b\n        w -= lr * (2/n) * np.sum((y_pred - y) * X)\n        b -= lr * (2/n) * np.sum(y_pred - y)\n        losses.append(np.mean((y_pred - y)**2))\n    return losses\n\nfor lr in [0.001, 0.01, 0.1, 0.5]:\n    plt.plot(run_gd(lr), label=f'lr={lr}')\nplt.xlabel('Epoch'); plt.ylabel('MSE'); plt.legend(); plt.title('Learning Rate Comparison'); plt.yscale('log'); plt.show()"),
        md("## Exercise 13\n\nImplement gradient descent for $f(w) = w^2 + 4w + 6$ starting at $w=10$. Analytical minimum at $w=-2$. Verify convergence."),
        code("# YOUR CODE HERE\n"),
        std_footer("18", "Gradient descent is the core optimization algorithm of all deep learning.", "19_SGD_and_Mini_batch.ipynb"),
    ])

    save("19_SGD_and_Mini_batch.ipynb", [
        md(HEADER.format(part="D — Optimization", dur="2 hours",
            objs="1. Understand stochastic vs batch gradient descent\n2. Implement mini-batch SGD\n3. Analyze batch size tradeoffs\n4. Connect to PyTorch DataLoader (Module 05)")),
        md("## 1. Three Variants\n\n| Method | Gradient computed on | Noise | Speed |\n|--------|---------------------|-------|-------|\n| Batch GD | All n samples | None | Slow per epoch |\n| SGD | 1 sample | High | Fast but noisy |\n| Mini-batch | b samples | Moderate | Best tradeoff |\n\n**Mini-batch SGD** is the standard in deep learning (your water-bodies `batch_size=8`)."),
        md("## 2. Update Rule (Mini-batch)\n\n$$\\mathbf{w}_{t+1} = \\mathbf{w}_t - \\eta \\nabla L_{\\mathcal{B}_t}(\\mathbf{w}_t)$$\n\nwhere $\\mathcal{B}_t$ is a random batch of size $b$."),
        IMPORTS,
        code("def mini_batch_gd(X, y, lr=0.1, batch_size=32, epochs=50, seed=42):\n    rng = np.random.default_rng(seed)\n    n, d = X.shape\n    w = np.zeros(d)\n    losses = []\n    for epoch in range(epochs):\n        idx = rng.permutation(n)\n        for start in range(0, n, batch_size):\n            batch = idx[start:start+batch_size]\n            Xb, yb = X[batch], y[batch]\n            grad = (2/len(batch)) * Xb.T @ (Xb @ w - yb)\n            w -= lr * grad\n        losses.append(np.mean((X @ w - y)**2))\n    return w, losses\n\nX = rng.normal(0, 1, (200, 5))\ny = X @ rng.normal(0, 1, 5) + rng.normal(0, 0.1, 200)\n\nfor bs in [1, 16, 200]:\n    _, losses = mini_batch_gd(X, y, batch_size=bs, epochs=30)\n    plt.plot(losses, label=f'batch={bs}')\nplt.xlabel('Epoch'); plt.ylabel('MSE'); plt.legend(); plt.title('Batch Size Effect'); plt.show()"),
        std_footer("19", "Mini-batch SGD balances speed and stability — used in every deep learning trainer.", "20_Momentum_and_Adaptive_Methods.ipynb"),
    ])

    save("20_Momentum_and_Adaptive_Methods.ipynb", [
        md(HEADER.format(part="D — Optimization", dur="3 hours",
            objs="1. Derive momentum update rule\n2. Derive RMSProp and Adam update rules\n3. Implement Adam optimizer from scratch\n4. Compare optimizers on same problem")),
        md("## 1. Momentum\n\nAccumulates velocity to smooth updates:\n$$\\mathbf{v}_{t+1} = \\beta \\mathbf{v}_t + \\nabla L(\\mathbf{w}_t)$$\n$$\\mathbf{w}_{t+1} = \\mathbf{w}_t - \\eta \\mathbf{v}_{t+1}$$\n\nLike a ball rolling downhill — accelerates in consistent directions."),
        md("## 2. Adam (Adaptive Moment Estimation)\n\nCombines momentum (1st moment) and RMSProp (2nd moment):\n\n$$m_t = \\beta_1 m_{t-1} + (1-\\beta_1) g_t$$\n$$v_t = \\beta_2 v_{t-1} + (1-\\beta_2) g_t^2$$\n$$\\hat{m}_t = \\frac{m_t}{1-\\beta_1^t}, \\quad \\hat{v}_t = \\frac{v_t}{1-\\beta_2^t}$$\n$$\\mathbf{w}_{t+1} = \\mathbf{w}_t - \\eta \\frac{\\hat{m}_t}{\\sqrt{\\hat{v}_t} + \\epsilon}$$\n\nDefault: $\\beta_1=0.9, \\beta_2=0.999, \\epsilon=10^{-8}$\n\nYour **water-bodies-detection** uses AdamW (Adam + decoupled weight decay)."),
        IMPORTS,
        code("def adam_optimizer(grad_fn, w_init, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, steps=100):\n    w = w_init.copy()\n    m, v = np.zeros_like(w), np.zeros_like(w)\n    history = []\n    for t in range(1, steps + 1):\n        g = grad_fn(w)\n        m = beta1 * m + (1 - beta1) * g\n        v = beta2 * v + (1 - beta2) * g**2\n        m_hat = m / (1 - beta1**t)\n        v_hat = v / (1 - beta2**t)\n        w -= lr * m_hat / (np.sqrt(v_hat) + eps)\n        history.append(w.copy())\n    return w, history\n\n# Minimize f(w) = w[0]**2 + 10*w[1]**2 (ill-conditioned)\ngrad = lambda w: np.array([2*w[0], 20*w[1]])\nw0 = np.array([5.0, 5.0])\nw_final, hist = adam_optimizer(grad, w0, lr=0.5, steps=50)\nprint(f'Start: {w0}, End: {w_final}')"),
        code("# Compare SGD, Momentum, Adam\nw_sgd, w_mom, w_adam = [np.array([5., 5.]) for _ in range(3)]\nv = np.zeros(2)\nhist_sgd, hist_mom, hist_adam = [], [], []\n\nfor t in range(100):\n    g = np.array([2*w_sgd[0], 20*w_sgd[1]])\n    w_sgd -= 0.01 * g; hist_sgd.append(w_sgd.copy())\n    \n    g = np.array([2*w_mom[0], 20*w_mom[1]])\n    v = 0.9 * v + g\n    w_mom -= 0.01 * v; hist_mom.append(w_mom.copy())\n    \n    _, h = adam_optimizer(lambda w: np.array([2*w[0], 20*w[1]]), w_adam, lr=0.1, steps=1)\n    w_adam = h[0]; hist_adam.append(w_adam.copy())\n\nfor hist, lbl in [(hist_sgd,'SGD'),(hist_mom,'Momentum'),(hist_adam,'Adam')]:\n    h = np.array(hist)\n    plt.plot(h[:,0], h[:,1], 'o-', markersize=2, label=lbl, alpha=0.7)\nplt.plot(0, 0, 'k*', markersize=15, label='Optimum')\nplt.xlabel('w0'); plt.ylabel('w1'); plt.legend(); plt.title('Optimizer Trajectories'); plt.show()"),
        md("## Exercise 14\n\nImplement RMSProp from scratch. Compare convergence with SGD on $f(w_0, w_1) = w_0^2 + 10w_1^2$."),
        code("# YOUR CODE HERE\n"),
        std_footer("20", "Adam is the default optimizer in deep learning. Understanding its update rule is essential.", "21_Regression_Losses.ipynb"),
    ])


def build_part_e():
    save("21_Regression_Losses.ipynb", [
        md(HEADER.format(part="E — Loss Functions", dur="2 hours",
            objs="1. Derive MSE and MAE loss functions\n2. Compute gradients of regression losses\n3. Understand Huber loss robustness\n4. Visualize loss landscapes")),
        md("## 1. Mean Squared Error (MSE)\n\n$$L = \\frac{1}{n}\\sum_{i=1}^{n}(y_i - \\hat{y}_i)^2$$\n\n**Gradient:** $\\frac{\\partial L}{\\partial \\hat{y}_i} = \\frac{2}{n}(\\hat{y}_i - y_i)$\n\nPenalizes large errors heavily (squared). Sensitive to outliers."),
        md("## 2. Mean Absolute Error (MAE)\n\n$$L = \\frac{1}{n}\\sum_{i=1}^{n}|y_i - \\hat{y}_i|$$\n\nRobust to outliers. Not differentiable at zero (use subgradient)."),
        md("## 3. Huber Loss\n\nCombines MSE (small errors) and MAE (large errors):\n$$L_\\delta(e) = \\begin{cases} \\frac{1}{2}e^2 & |e| \\leq \\delta \\\\ \\delta(|e| - \\frac{\\delta}{2}) & |e| > \\delta \\end{cases}$$"),
        IMPORTS,
        code("errors = np.linspace(-5, 5, 200)\nmse = errors**2\nmae = np.abs(errors)\nhuber = np.where(np.abs(errors) <= 1, 0.5*errors**2, np.abs(errors) - 0.5)\n\nplt.plot(errors, mse, label='MSE')\nplt.plot(errors, mae, label='MAE')\nplt.plot(errors, huber, label='Huber (δ=1)')\nplt.xlabel('Error (y - ŷ)'); plt.ylabel('Loss'); plt.legend(); plt.title('Regression Loss Functions'); plt.show()"),
        md("## ML Connection\n\nYour `Day - 11` linear regression minimizes MSE. House price outliers would benefit from Huber or MAE."),
        std_footer("21", "Regression losses measure prediction error. MSE is the default for continuous targets.", "22_Classification_Losses.ipynb"),
    ])

    save("22_Classification_Losses.ipynb", [
        md(HEADER.format(part="E — Loss Functions", dur="2.5 hours",
            objs="1. Derive binary cross-entropy from MLE\n2. Derive multi-class cross-entropy\n3. Understand hinge loss for SVM\n4. Derive focal loss for class imbalance")),
        md("## 1. Binary Cross-Entropy (Derive from MLE)\n\nFor $y \\in \\{0,1\\}$, model outputs $\\hat{p} = \\sigma(z)$:\n\n$$L = -[y \\log \\hat{p} + (1-y)\\log(1-\\hat{p})]$$\n\n**Derivation from Bernoulli MLE:**\n$$\\ell(p) = y\\log p + (1-y)\\log(1-p)$$\n$$L = -\\ell(p) \\text{ (negate to minimize)}$$\n\n**Gradient w.r.t. logit $z$:**\n$$\\frac{\\partial L}{\\partial z} = \\hat{p} - y$$"),
        md("## 2. Multi-Class Cross-Entropy\n\nWith softmax $\\hat{p}_k = \\frac{e^{z_k}}{\\sum_j e^{z_j}}$:\n$$L = -\\sum_{k=1}^{K} y_k \\log \\hat{p}_k$$"),
        md("## 3. Focal Loss\n\n$$L = -(1 - \\hat{p})^\\gamma \\log(\\hat{p})$$\n\nDown-weights easy examples. Used in object detection (RetinaNet) and imbalanced segmentation."),
        IMPORTS,
        code("def binary_cross_entropy(y, p_hat, eps=1e-15):\n    p_hat = np.clip(p_hat, eps, 1 - eps)\n    return -(y * np.log(p_hat) + (1 - y) * np.log(1 - p_hat))\n\ndef sigmoid(z): return 1 / (1 + np.exp(-z))\n\n# Plot BCE vs predicted probability\np = np.linspace(0.001, 0.999, 100)\nplt.plot(p, binary_cross_entropy(1, p), label='y=1')\nplt.plot(p, binary_cross_entropy(0, p), label='y=0')\nplt.xlabel('Predicted probability'); plt.ylabel('Loss'); plt.legend(); plt.title('Binary Cross-Entropy'); plt.show()"),
        md("## 4. Module Assignment Preview\n\nDerive and implement the gradient of logistic regression loss:\n$$L = -[y \\log \\sigma(w^Tx) + (1-y)\\log(1-\\sigma(w^Tx))]$$\n$$\\frac{\\partial L}{\\partial w} = (\\sigma(w^Tx) - y) x$$\n\nFull assignment in `exercises/README.md`."),
        md("## Exercise 15\n\nImplement multi-class cross-entropy with softmax. Verify gradient numerically."),
        code("# YOUR CODE HERE\n"),
        std_footer("22", "Cross-entropy is the standard classification loss, derived from MLE.", "23_Segmentation_Losses.ipynb"),
    ])

    save("23_Segmentation_Losses.ipynb", [
        md(HEADER.format(part="E — Loss Functions", dur="2.5 hours",
            objs="1. Derive Dice loss\n2. Derive IoU (Jaccard) loss\n3. Understand Lovász loss extension\n4. Connect to water-bodies-detection AquaBoundaryLoss")),
        md("## 1. Dice Coefficient and Loss\n\n$$\\text{Dice} = \\frac{2|A \\cap B|}{|A| + |B|} = \\frac{2\\sum p_i g_i}{\\sum p_i + \\sum g_i}$$\n\n$$L_{\\text{Dice}} = 1 - \\text{Dice}$$\n\nGood for **class imbalance** (small water bodies in large tiles)."),
        md("## 2. IoU (Jaccard) Loss\n\n$$\\text{IoU} = \\frac{|A \\cap B|}{|A \\cup B|} = \\frac{\\sum p_i g_i}{\\sum p_i + \\sum g_i - \\sum p_i g_i}$$\n\n$$L_{\\text{IoU}} = 1 - \\text{IoU}$$"),
        md("## 3. Your water-bodies-detection Loss\n\nFrom `losses.py`:\n```python\nAquaBoundaryLoss = w_aqua * (BCE_aqua + Dice_aqua) + w_boundary * (BCE_boundary + Dice_boundary)\n```\n\nEach head gets BCE (pixel-wise classification) + Dice (region overlap). This handles both pixel accuracy and region-level segmentation quality."),
        IMPORTS,
        code("def dice_coefficient(pred, target, eps=1e-7):\n    pred = pred.flatten().astype(float)\n    target = target.flatten().astype(float)\n    intersection = (pred * target).sum()\n    return (2 * intersection + eps) / (pred.sum() + target.sum() + eps)\n\ndef iou_coefficient(pred, target, eps=1e-7):\n    pred = pred.flatten().astype(float)\n    target = target.flatten().astype(float)\n    intersection = (pred * target).sum()\n    union = pred.sum() + target.sum() - intersection\n    return (intersection + eps) / (union + eps)\n\n# Simulated prediction vs ground truth\npred = (rng.random((64, 64)) > 0.5).astype(float)\ntarget = (rng.random((64, 64)) > 0.7).astype(float)\n\nprint(f'Dice: {dice_coefficient(pred, target):.4f}')\nprint(f'IoU:  {iou_coefficient(pred, target):.4f}')\nprint(f'Dice Loss: {1 - dice_coefficient(pred, target):.4f}')"),
        code("# Visualize: imbalanced segmentation\nfig, axes = plt.subplots(1, 3, figsize=(12, 4))\n\n# Small object in large image\ntarget = np.zeros((128, 128))\ntarget[50:70, 50:70] = 1  # small square\n\n# Good vs bad prediction\npred_good = target.copy()\npred_good[55:65, 55:65] = 0.8  # slight error\npred_bad = np.zeros((128, 128))\npred_bad[40:80, 40:80] = 1  # too large\n\nfor ax, p, title in zip(axes, [target, pred_good, pred_bad], ['Ground Truth', 'Good Pred', 'Bad Pred']):\n    ax.imshow(p, cmap='Blues'); ax.set_title(title); ax.axis('off')\n\nprint(f'Good pred — Dice: {dice_coefficient(pred_good, target):.3f}, IoU: {iou_coefficient(pred_good, target):.3f}')\nprint(f'Bad pred  — Dice: {dice_coefficient(pred_bad, target):.3f}, IoU: {iou_coefficient(pred_bad, target):.3f}')\nplt.tight_layout(); plt.show()"),
        md("## Module 02 Complete\n\nYou now have the mathematical foundation for all ML algorithms.\n\n**Before Module 03:** Complete exercises, assignment (logistic regression gradient), quiz ≥80%, gate questions.\n\nSee: [CHEATSHEET.md](CHEATSHEET.md) | [quiz/module_02_quiz.md](quiz/module_02_quiz.md)"),
        std_footer("23", "Segmentation losses handle class imbalance — critical for GeoSpatial AI.", None),
    ])


if __name__ == "__main__":
    print("Building Module 02 notebooks...")
    build_part_a()
    build_part_b()
    build_part_c()
    build_part_d()
    build_part_e()
    print("Done: 23 notebooks created.")
