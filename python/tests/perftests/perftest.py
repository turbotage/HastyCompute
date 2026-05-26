import torch


def theta_grad_orthogonal(
    Theta,
    Lambda,
    g_generator,
    k_batch_size=128,
):
    """
    Gradient assuming orthonormal columns of Theta.

    Assumes:
        sum_i Theta[i,p].conj() * Theta[i,q] = delta_pq

    Computes:
        dL / dTheta*

    Parameters
    ----------
    Theta : complex tensor, shape (Ni, P)

    Lambda : complex tensor, shape (Nk, P)

    g_generator : callable
        g_generator(k_indices) -> tensor shape (Kb, Ni)

        returns:
            g[k,i] = exp(Phi(r_i, t_k))

    k_batch_size : int

    Returns
    -------
    grad : complex tensor, shape (Ni, P)
    """

    device = Theta.device
    dtype = Theta.dtype

    Ni, P = Theta.shape
    Nk = Lambda.shape[0]

    grad = torch.zeros_like(Theta)

    for k0 in range(0, Nk, k_batch_size):

        k1 = min(k0 + k_batch_size, Nk)

        ks = torch.arange(k0, k1, device=device)

        # shape: (Kb, Ni)
        g_batch = g_generator(ks)

        # alpha[k,p] = sum_i g*[k,i] Theta[i,p]
        #
        # shape: (Kb, P)
        alpha = g_batch.conj() @ Theta

        # first term:
        #
        # sum_k |Lambda_kp|^2 Theta_jp
        #
        coeff = torch.sum(torch.abs(Lambda[ks]) ** 2, dim=0)

        grad += Theta * coeff[None, :]

        # second term:
        #
        # sum_k Lambda*_kp g_jk alpha_kp
        #
        term = (
            g_batch[:, :, None]
            * alpha[:, None, :]
            * Lambda[ks].conj()[:, None, :]
        )

        grad -= torch.sum(term, dim=0)

    return grad


def theta_grad_nonorthogonal(
    Theta,
    Lambda,
    g_generator,
    k_batch_size=128,
):
    """
    Gradient without orthogonality assumption.

    Computes:
        dL / dTheta*

    Parameters
    ----------
    Theta : complex tensor, shape (Ni, P)

    Lambda : complex tensor, shape (Nk, P)

    g_generator : callable
        g_generator(k_indices) -> tensor shape (Kb, Ni)

    Returns
    -------
    grad : complex tensor, shape (Ni, P)
    """
    device = Theta.device

    Ni, P = Theta.shape
    Nk = Lambda.shape[0]

    grad = torch.zeros_like(Theta)

    #
    # Gram matrix:
    #
    # G[p',p] = sum_i Theta*[i,p'] Theta[i,p]
    #
    G = Theta.conj().T @ Theta

    for k0 in range(0, Nk, k_batch_size):

        k1 = min(k0 + k_batch_size, Nk)

        ks = torch.arange(k0, k1, device=device)

        # shape: (Kb, Ni)
        g_batch = g_generator(ks)

        # shape: (Kb, P)
        alpha = g_batch.conj() @ Theta

        #
        # first term:
        #
        # sum_p' Lambda_kp' Lambda*_kp G[p',p] Theta_jp'
        #
        # Build:
        #
        # C[k,p',p]
        #
        C = (
            Lambda[ks][:, :, None]
            * Lambda[ks].conj()[:, None, :]
            * G[None, :, :]
        )

        #
        # Sum over p'
        #
        # tmp[k,p']
        #
        tmp = torch.sum(C, dim=2)

        #sudo udevadm control --reload-rules && sudo udevadm trigger
        # sum_k Theta_jp' tmp_kp'
        #
        grad += Theta @ torch.sum(tmp, dim=0).diag()

        #
        # second term
        #
        term = (
            g_batch[:, :, None]
            * alpha[:, None, :]
            * Lambda[ks].conj()[:, None, :]
        )

        grad -= torch.sum(term, dim=0)

    return grad


def orthonormalize_theta(Theta, Lambda=None):
    """
    Orthonormalize columns of Theta using QR.

    If Lambda is supplied, rescales Lambda so that
    the represented operator stays approximately invariant.

    Original:
        sum_p Lambda_kp Theta_p Theta_p^

    After QR:
        Theta = Q R

    We absorb R into Lambda approximately.

    Parameters
    ----------
    Theta : complex tensor, shape (Ni, P)

    Lambda : complex tensor, optional
        shape (Nk, P)

    Returns
    -------
    Theta_orth : complex tensor

    Lambda_new : complex tensor or None
    """

    #
    # QR decomposition
    #
    # Theta = Q R
    #
    Q, R = torch.linalg.qr(Theta)

    if Lambda is None:
        return Q

    #
    # We approximately absorb diagonal scaling into Lambda.
    #
    # Exact absorption is impossible because:
    #
    # Theta_p Theta_p^H
    #
    # is quadratic in Theta.
    #
    # But diagonal scaling works well in practice.
    #
    diagR = torch.diagonal(R)

    Lambda_new = Lambda * (torch.abs(diagR)[None, :] ** 2)

    return Q, Lambda_new



p = 10
Nk = 4e5
Nv = 1e5

Lpk = torch.randn(p, Nk)
Oip = torch.randn(p, Nv)


