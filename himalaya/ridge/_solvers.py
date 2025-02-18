import numbers
import warnings

import numpy as np

from ..backend import get_backend
from ..utils import _batch_or_skip
from ..validation import check_cv
from ..progress_bar import bar
from ._random_search import solve_group_ridge_random_search, _decompose_ridge


def solve_ridge_svd(
    X,
    Y,
    alpha=1.0,
    method="svd",
    fit_intercept=False,
    negative_eigenvalues="zeros",
    n_targets_batch=None,
    warn=True,
):
    """Solve ridge regression using SVD decomposition.

    Solve the ridge regression::

        b* = argmin_B ||X @ b - Y||^2 + alpha ||b||^2

    Parameters
    ----------
    X : array of shape (n_samples, n_features)
        Input features.
    Y : array of shape (n_samples, n_targets)
        Target data.
    alpha : float, or array of shape (n_targets, )
        Regularization parameter.
    method : str in {"svd"}
        Method used to diagonalize the input feature matrix.
    fit_intercept : boolean
        Whether to fit an intercept.
        If False, X and Y must be zero-mean over samples.
    negative_eigenvalues : str in {"nan", "error", "zeros"}
        If the decomposition leads to negative eigenvalues (wrongly emerging
        from float32 errors):
        - "error" raises an error.
        - "zeros" replaces them with zeros.
        - "nan" returns nans if the regularization does not compensate
        twice the smallest negative value, else it ignores the problem.
    n_targets_batch : int or None
        Size of the batch for over targets during cross-validation.
        Used for memory reasons. If None, uses all n_targets at once.
    warn : bool
        If True, warn if the number of samples is smaller than the number of
        features.

    Returns
    -------
    weights : array of shape (n_features, n_targets)
        Ridge coefficients.
    intercept : array of shape (n_targets,)
        Intercept. Only returned when fit_intercept is True.
    """
    backend = get_backend()
    if isinstance(alpha, numbers.Number) or alpha.ndim == 0:
        alpha = backend.ones_like(Y, shape=(1,)) * alpha

    X, Y, alpha = backend.check_arrays(X, Y, alpha)

    n_samples, n_features = X.shape
    if n_samples < n_features and warn:
        warnings.warn(
            "Solving ridge is slower than solving kernel ridge when n_samples "
            f"< n_features (here {n_samples} < {n_features}). "
            "Using a linear kernel in himalaya.kernel_ridge.KernelRidge or "
            "himalaya.kernel_ridge.solve_kernel_ridge_eigenvalues would be "
            "faster. Use warn=False to silence this warning.",
            UserWarning,
        )
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of samples.")

    X_offset, Y_offset = None, None
    if fit_intercept:
        X_offset = X.mean(0)
        Y_offset = Y.mean(0)
        X = X - X_offset
        Y = Y - Y_offset

    if method == "svd":
        # SVD: X = U @ np.diag(eigenvalues) @ Vt
        U, eigenvalues, Vt = backend.svd(X, full_matrices=False)
    else:
        raise ValueError("Unknown method=%r." % (method,))

    inverse = eigenvalues[:, None] / (alpha[None] + eigenvalues[:, None] ** 2)

    # negative eigenvalues can emerge from incorrect kernels, or from float32
    if eigenvalues[0] < 0:
        if negative_eigenvalues == "nan":
            if alpha < -eigenvalues[0] * 2:
                return backend.ones_like(Y) * backend.asarray(
                    backend.nan, dtype=Y.dtype
                )
            else:
                pass

        elif negative_eigenvalues == "zeros":
            eigenvalues[eigenvalues < 0] = 0

        elif negative_eigenvalues == "error":
            raise RuntimeError(
                "Negative eigenvalues. Make sure the kernel is positive "
                "semi-definite, increase the regularization alpha, or use"
                "another solver."
            )
        else:
            raise ValueError(
                "Unknown negative_eigenvalues=%r." % (negative_eigenvalues,)
            )

    n_samples, n_features = X.shape
    n_samples, n_targets = Y.shape
    weights = backend.zeros_like(X, shape=(n_features, n_targets), device="cpu")
    if n_targets_batch is None:
        n_targets_batch = n_targets

    for start in range(0, n_targets, n_targets_batch):
        batch = slice(start, start + n_targets_batch)

        iUT = _batch_or_skip(inverse, batch, 1)[:, None, :] * U.T[:, :, None]
        iUT = backend.transpose(iUT, (2, 0, 1))
        # iUT.shape = (1 or n_targets_batch, n_samples, n_samples)

        if Y.shape[0] < Y.shape[1]:
            weights_batch = ((Vt.T @ iUT) @ Y.T[batch, :, None])[:, :, 0].T
        else:
            weights_batch = Vt.T @ (iUT @ Y.T[batch, :, None])[:, :, 0].T
        weights[:, batch] = backend.to_cpu(weights_batch)

    if fit_intercept:
        intercept = backend.to_cpu(Y_offset) - backend.to_cpu(X_offset) @ weights
        return weights, intercept
    else:
        return weights


#: Dictionary with all ridge solvers
RIDGE_SOLVERS = {"svd": solve_ridge_svd}


def solve_group_ridge_deterministic(
    Xs,
    Y,
    hparams,
    fit_intercept=False,
    cv=5,
    return_weights=False,
    n_targets_batch=None,
    n_targets_batch_refit=None,
    n_alphas_batch=None,
    progress_bar=True,
    Y_in_cpu=False,
    diagonalize_method="svd",
):
    """Solve group ridge regression using random search on the simplex.

    Solve the group-regularized ridge regression::

        b* = argmin_b ||Z @ b - Y||^2 + ||b||^2

    where the feature space X_i is scaled by a group scaling ::

        Z_i = exp(deltas[i] / 2) X_i

    Parameters
    ----------
    Xs : list of len (n_spaces), with arrays of shape (n_samples, n_features)
        Input features.
    Y : array of shape (n_samples, n_targets)
        Target data.
    hparams: tuple of length 2, with arrays (gammas, alphas)
        Hyperparameters that the deterministic search will use to solve the
        regression.
    fit_intercept : boolean
        Whether to fit an intercept.
        If False, Xs and Y must be zero-mean over samples.
    cv : int or scikit-learn splitter
        Cross-validation splitter. If an int, KFold is used.
    return_weights : bool
        Whether to refit on the entire dataset and return the weights.
    random_state : int, or None
        Random generator seed. Use an int for deterministic search.
    n_targets_batch : int or None
        Size of the batch for over targets during cross-validation.
        Used for memory reasons. If None, uses all n_targets at once.
    n_targets_batch_refit : int or None
        Size of the batch for over targets during refit.
        Used for memory reasons. If None, uses all n_targets at once.
    n_alphas_batch : int or None
        Size of the batch for over alphas. Used for memory reasons.
        If None, uses all n_alphas at once.
    progress_bar : bool
        If True, display a progress bar over gammas.
    Y_in_cpu : bool
        If True, keep the target values ``Y`` in CPU memory (slower).
    diagonalize_method : str in {"svd"}
        Method used to diagonalize the features.


    Returns
    -------
    deltas : array of shape (n_spaces, n_targets)
        Best log feature-space weights for each target.
    refit_weights : array of shape (n_features, n_targets), or None
        Refit regression weights on the entire dataset, using selected best
        hyperparameters. Refit weights are always stored on CPU memory.
    cv_scores : array of shape (n_iter, n_targets)
        Cross-validation scores per iteration, averaged over splits, for the
        best alpha. Cross-validation scores will always be on CPU memory.
    best_gammas: arrray of shape (n_spaces, n_targts)
        Best gamma (feature rescaling factor) for each target. For deteministic
        search, this is simply `hparams[0]`.
    intercept : array of shape (n_targets,)
        Intercept. Only returned when fit_intercept is True.
    """
    print("🐛 02/15/2025 Updates in _random_search.py > deterministic")

    backend = get_backend()

    if len(hparams) != 2:
        raise ValueError(
            "hparams should be length two and hold values [gammas, alphas]"
        )

    best_gammas, best_alphas = hparams

    n_spaces = len(Xs)

    dtype = Xs[0].dtype
    best_gammas = backend.asarray(best_gammas, dtype=dtype)
    best_alphas = backend.asarray(best_alphas, dtype=dtype)
    device = getattr(best_gammas, "device", None)
    Xs = [backend.asarray(X, dtype=dtype) for X in Xs]
    Y = backend.asarray(Y, dtype=dtype, device="cpu" if Y_in_cpu else device)

    # stack all features
    X_ = backend.concatenate(Xs, 1)
    n_features_list = [X.shape[1] for X in Xs]
    n_features = X_.shape[1]
    start_and_end = np.concatenate([[0], np.cumsum(n_features_list)])
    slices = [
        slice(start, end) for start, end in zip(start_and_end[:-1], start_and_end[1:])
    ]
    del Xs

    n_samples, n_features = X_.shape
    # if n_samples < n_features and warn:
    #     warnings.warn(
    #         "Solving banded ridge is slower than solving multiple-kernel ridge"
    #         f" when n_samples < n_features (here {n_samples} < {n_features}). "
    #         "Using linear kernels in "
    #         "himalaya.kernel_ridge.MultipleKernelRidgeCV or "
    #         "himalaya.kernel_ridge.solve_multiple_kernel_ridge_random_search "
    #         "would be faster. Use warn=False to silence this warning.",
    #         UserWarning,
    #     )
    if X_.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of samples.")

    X_offset, Y_offset = None, None
    if fit_intercept:
        X_offset = X_.mean(0)
        Y_offset = Y.mean(0)
        X_ = X_ - X_offset
        Y = Y - Y_offset

    n_samples, n_targets = Y.shape
    if n_targets_batch is None:
        n_targets_batch = n_targets
    if n_targets_batch_refit is None:
        n_targets_batch_refit = n_targets_batch
    if n_alphas_batch is None:
        n_alphas_batch = len(best_alphas)

    cv = check_cv(cv, Y)
    n_splits = cv.get_n_splits()
    for train, val in cv.split(Y):
        if len(val) == 0 or len(train) == 0:
            raise ValueError(
                "Empty train or validation set. "
                "Check that `cv` is correctly defined."
            )

    # initialize refit ridge weights
    refit_weights = None
    if return_weights:
        refit_weights = backend.zeros_like(
            best_gammas, shape=(n_features, n_targets), device="cpu"
        )
        print(
            "🐛 At _random_search.py deterministic: refit_weights initial shape",
            refit_weights.shape,
        )

    unique_gammas = backend.unique(best_gammas, axis=1).T
    print("best_gammas processed: ", unique_gammas)

    # Main loop
    for gamma in bar(
        unique_gammas,
        "Deterministic fitting with cv",
        use_it=progress_bar,
    ):
        for kk in range(n_spaces):
            X_[:, slices[kk]] *= backend.sqrt(gamma[kk])

        # Compute mask, which is the columns in which current set of gammas was used.
        cond = (
            best_gammas.T == gamma
        )  # We see which columns match the current set of gammas
        mask = cond[:, 0] * cond[:, 1]

        # 🟢 compute primal or dual weights on the entire dataset (nocv)
        if return_weights:
            update_indices = backend.flatnonzero(mask)
            if Y_in_cpu:
                update_indices = backend.to_cpu(update_indices)
            if len(update_indices) > 0:

                # *refit weights* only for alphas used by at least one target
                used_alphas = backend.unique(best_alphas[mask])
                primal_weights = backend.zeros_like(
                    X_, shape=(n_features, len(update_indices)), device="cpu"
                )
                # 🔴 TODO: Get rid of for loop and use the predetermined best alpha instead.
                for matrix, alpha_batch in _decompose_ridge(
                    Xtrain=X_,
                    alphas=used_alphas,
                    negative_eigenvalues="zeros",
                    n_alphas_batch=min(len(used_alphas), n_alphas_batch),
                    method=diagonalize_method,
                ):

                    for start in range(0, len(update_indices), n_targets_batch_refit):
                        batch = slice(start, start + n_targets_batch_refit)

                        weights = backend.matmul(
                            matrix,
                            backend.to_gpu(Y[:, update_indices[batch]], device=device),
                        )
                        # used_n_alphas_batch, n_features, n_targets_batch = \
                        # weights.shape

                        # select alphas corresponding to best cv_score
                        alphas_indices = backend.searchsorted(
                            used_alphas, best_alphas[mask][batch]
                        )
                        # mask targets whose selected alphas are outside the
                        # alpha batch
                        mask2 = backend.isin(
                            alphas_indices,
                            backend.arange(len(used_alphas))[alpha_batch],
                        )
                        # get indices in alpha_batch
                        alphas_indices = backend.searchsorted(
                            backend.arange(len(used_alphas))[alpha_batch],
                            alphas_indices[mask2],
                        )
                        # update corresponding weights
                        mask_target = backend.arange(weights.shape[2])
                        mask_target = backend.to_gpu(mask_target)[mask2]
                        tmp = weights[alphas_indices, :, mask_target]
                        primal_weights[:, batch][:, backend.to_cpu(mask2)] = (
                            backend.to_cpu(tmp).T
                        )
                        del weights, alphas_indices, mask2, mask_target
                    del matrix

                # multiply again by np.sqrt(g), as we then want to use
                # the primal weights on the unscaled features Xs, and not
                # on the scaled features (np.sqrt(g) * Xs)
                for kk in range(n_spaces):
                    primal_weights[slices[kk]] *= backend.to_cpu(
                        backend.sqrt(gamma[kk])
                    )

                refit_weights[:, backend.to_cpu(mask)] = primal_weights
                del primal_weights

            del update_indices

        # 🟢
        del mask

        for kk in range(n_spaces):
            X_[:, slices[kk]] /= backend.sqrt(gamma[kk])

    # end main loop

    deltas = backend.log(best_gammas / best_alphas[None, :])
    cv_scores = None

    if fit_intercept:
        intercept = (
            (backend.to_cpu(Y_offset) - backend.to_cpu(X_offset) @ refit_weights)
            if return_weights
            else None
        )
        return deltas, refit_weights, cv_scores, best_gammas, intercept
    else:
        return deltas, refit_weights, cv_scores, best_gammas


#: Dictionary with all group ridge solvers
GROUP_RIDGE_SOLVERS = {
    "random_search": solve_group_ridge_random_search,
    "deterministic": solve_group_ridge_deterministic,
}
