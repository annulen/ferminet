# Copyright 2023 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tools for density matrix calculation."""

import functools
from typing import Tuple

from ferminet import constants
from ferminet import mcmc
from ferminet import networks
from ferminet.utils import scf
import itertools
import jax
from jax import numpy as jnp
import jax.scipy.special as jss
import numpy as np


def _eval_mos(pos: jnp.ndarray, scf_approx: scf.Scf,
              nspins: Tuple[int, int]) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Evaluates molecular orbitals.

  Args:
    pos: Electron positions of shape (M, 3), where M can be anything.
    scf_approx: SCF object with information about the Hartree-Fock calculation.
    nspins: Number of spin-up and spin-down electrons.

  Returns:
    all_mos: All MOs evaluated at positions `pos`.
    occ_mos: Just the occupied MOs evaluated at those positions.
  """

  if scf_approx.restricted:
    all_mos = jnp.asarray(scf_approx.eval_mos(pos)[0])
    occ_mos = jnp.asarray(all_mos[:, :nspins[0]])
  else:
    all_mos = list(scf_approx.eval_mos(pos))
    occ_mos = jnp.concatenate(
        [mo[:, :nspin] for mo, nspin in zip(all_mos, nspins)], axis=-1)
    all_mos = jnp.concatenate(all_mos, axis=-1)

  return all_mos, occ_mos


def calc_hf_prob(pos: jnp.ndarray, scf_approx: scf.Scf,
                 nspins: Tuple[int, int]) -> jnp.ndarray:
  """Calculates the probability of the current configuration based on Hartree-Fock.
  """

  # evaluate occupied phi's
  _, occ_mos = _eval_mos(
      pos=pos.reshape(-1, 3), scf_approx=scf_approx, nspins=nspins)

  # The probability density of finding a single electron at position r, given a
  # HF wavefunction, is the mean of |phi_i|^2 for each occupied orbital phi_i.
  # See Eq. (2.19) here for proof (after dividing by the number of electrons
  # to convert the electron density to the one-electron probability):
  # https://www.home.uni-osnabrueck.de/apostnik/Lectures/DFT-2.pdf.
  # Intuitively this result says that an electron has the same probability of
  # being in any of the orbitals (electrons are indistinguishable), and the
  # probability density associated with an orbital is |phi_i|^2. So the overall
  # probability is mean(|phi_i|^2).

  # For a restricted calculation we have N / 2 occupied MOs for N electrons,
  # and for an unrestricted calculation we have N (see `_eval_mos`
  # above). So occ_mos has size (batch * N, nocc) where nocc = the number of
  # occupied orbitals = N / 2 for restricted, and N for unrestricted. Then
  # taking jnp.mean(occ_mos ** 2, axis=-1) takes the average over the occupied
  # orbitals, evaluated at the (batch * N) electron positions.

  prob = jnp.mean(occ_mos**2, axis=-1).reshape(pos.shape[:-1])

  return prob


def make_effective_batch_network(
    scf_approx: scf.Scf,
    nspins: Tuple[int, int],
) -> networks.LogFermiNetLike:
  """Makes function to compute 1/2 * log(HF prob) for use in mcmc.make_mcmc_step.

  Args:
    scf_approx: SCF object with information about the Hartree-Fock calculation.
    nspins: Number of spin-up and spin-down electrons.

  Returns:
    eff_batch_network: Function that can be called in the same way as FermiNet
      and return 1/2 * log(probability). This can be used in existing functions
      built for FermiNet, like MCMC steps, but will instead use the probability
      of finding a single electron at a position using the Hartree-Fock
      density.
  """

  def eff_batch_network(params, pos, spins, atoms, charges):
    """Function that can be called like FermiNet, and returns 1/2 * log(HF prob).
    """
    del params, spins, atoms, charges
    prob = calc_hf_prob(pos=pos, scf_approx=scf_approx, nspins=nspins)

    # Since a normal batched network outputs log|psi|, an MCMC step multiplies
    # it by 2 so that it's equal to log|psi|^2 = log(probability). Since we
    # already have the probability here, we need to divide log(prob) by 2 to
    # counteract the multiplication in the MCMC step.

    half_log_prob = 1 / 2 * jnp.log(jnp.abs(prob))
    return half_log_prob

  return eff_batch_network


def make_rprime_mcmc_step(
    steps: int,
    ndim: int,
    blocks: int,
    nspins: Tuple[int, int],
    device_batch_size: int,
    scf_approx: scf.Scf,
) ->...:
  """Makes an MCMC step function for the r' electron positions.

  Args:
    steps: Number of MCMC moves to attempt in a single call to the MCMC step
      function.
    ndim: dimensionality of system.
    blocks: number of blocks to split electron updates into.
    nspins: Number of spin-up and spin-down electrons.
    device_batch_size: Batch size on each device.
    scf_approx: SCF object with information about the Hartree-Fock calculation.

  Returns:
    mcmc_step: A callable function that takes the same arguments as an MCMC
      step created in mcmc.py. The main change is that it also returns the
      probability associated with the last step. This is needed so that we can
      divide by the HF probability when computing the density rho(r, r').
  """
  eff_batch_network = make_effective_batch_network(
      scf_approx=scf_approx, nspins=nspins)
  base_mcmc_step = mcmc.make_mcmc_step(
      eff_batch_network,
      device_batch_size,
      steps=steps,
      blocks=blocks,
      ndim=ndim,
  )

  @functools.partial(constants.pmap)
  def mcmc_step(params, data, mcmc_key, mcmc_width):
    """Regular MCMC step, but with the final probability also returned."""
    # Do the same thing as a regular MCMC step, but also compute the probability
    # at the last step
    data, pmove = base_mcmc_step(params, data, mcmc_key, mcmc_width)
    prob = calc_hf_prob(
        pos=data.positions,
        scf_approx=scf_approx,
        nspins=nspins,
    )
    return data, prob, pmove

  return mcmc_step


def get_rho(
    batch_network: networks.FermiNetLike,
    params: networks.ParamTree,
    dim: int,
    pos: jnp.ndarray,
    spins: jnp.ndarray,
    charges: jnp.ndarray,
    nspins: Tuple[int, int],
    batch_atoms: jnp.ndarray,
    rj_pos: jnp.ndarray,
    probs: jnp.ndarray,
    scf_approx: scf.Scf,
) -> jnp.ndarray:
  r"""Gets a sample of the density matrix from r' and (r1, ..., rN) positions.

  The general approach is to construct the density matrix rho(r, r') in a basis
  of MOs {\phi_i}, giving us the matrix rho_ij. We construct samples of rho_ij
  for each MCMC step (after decorrelation) of the positions (r1, ..., rN) from
  the wavefunction, and the position r' from the marginal (one-electron)
  probability from the Hartree-Fock wavefunction. We use the HF wavefunction
  instead of the real wavefunction because the marginal distribution is known
  analytically.

  Explicitly, we have

  rho_ij = N \int dr' dr1 ... drN \phi_i(r1) \phi_j(r') * \psi(r1, r2, ..., rN)
           * \psi(r', r2, ..., rN).

  = N * expectation_{r' ~ p_HF(r'), {r1, ..., rN} ~ |psi(r1, ..., rN)|^2 } (
    \psi(r', ..., rN) \phi_i(r1) \phi_j(r')
    / [\psi(r1, ..., rN) * p_HF(r')] ),

  where
    p_HF(r') = \int dr2, ..., drN |\psi_HF(r', r2, ..., rN)|^2
  is the one-electron probability from Hartree-Fock.

  Args:
    batch_network: vmapped network giving the sign of psi and log|psi|.
    params: Network parameters.
    dim: System dimension.
    pos: (r1, ..., rN) electron positions.
    spins: Spin of each electron.
    charges: Atom charges.
    nspins: Number of spin-up and spin-down electrons.
    batch_atoms: Atom coordinates, replicated along batch dimensions.
    rj_pos: Sampled positions of r' (used in phi_j calculations).
    probs: Probability of sampling the current values of r'.
    scf_approx: Scf object with Hartree-Fock information about the current atom
      configuration.

  Returns:
    rho_mat: Estimate of the one-body density matrix, expressed in a basis of
    Hartree-Fock molecular orbitals.

  Raises:
    ValueError: if system dimension is not 3 or if the number of spin-up
    electrons is not equal to the number of spin-down electrons.
  """

  if dim != 3:
    raise ValueError('Only implemented for 3D systems')

  # Treat spins separately by default
  idx = (0, nspins[0]) if nspins[1] > 0 else (0,)
  rho_mats = []

  denom_signs, denom_logs = batch_network(
      params,
      pos,
      spins,
      batch_atoms,
      charges,
  )

  phi_j, _ = _eval_mos(
      pos=rj_pos.reshape(-1, dim), scf_approx=scf_approx, nspins=nspins)

  use_excited = denom_signs.ndim == 3  # only true for excited states
  if use_excited:
    nstates = denom_signs.shape[-1]
    # Reshape pos to be (batch, states, num_el * dim)
    pos = pos.reshape(pos.shape[0], nstates, -1)
    rj_pos = rj_pos.reshape(-1, nstates, dim)
    probs = probs.reshape(-1, nstates)
    phi_j = phi_j.reshape(-1, nstates, phi_j.shape[-1])

  norb = phi_j.shape[-1] // len(idx)  # number of orbitals per spin

  for spin, i in enumerate(idx):
    sampled_pos = pos.at[..., dim*i:dim*(i+1)].set(rj_pos)
    numer_signs, numer_logs = batch_network(
        params,
        sampled_pos,
        spins,
        batch_atoms,
        charges,
    )

    r1 = pos[..., dim*i:dim*(i+1)].reshape(-1, dim)
    phi_i, _ = _eval_mos(pos=r1, scf_approx=scf_approx, nspins=nspins)
    if use_excited:
      phi_i = phi_i.reshape(-1, nstates, phi_i.shape[-1])

      # subtract off log probs *before* computing log_max for stability
      numer_logs -= jnp.expand_dims(jnp.log(probs), -1)
      log_max = jnp.maximum(jnp.max(denom_logs, axis=[1, 2], keepdims=True),
                            jnp.max(numer_logs, axis=[1, 2], keepdims=True))
      denom = denom_signs * jnp.exp(denom_logs - log_max)
      numer = numer_signs * jnp.exp(numer_logs - log_max)

      phi_i_ = jnp.transpose(phi_i, (0, 2, 1))
      phi_i_ = phi_i_[:, spin*norb:(spin+1)*norb, None, :, None]

      phi_j_ = jnp.transpose(phi_j, (0, 2, 1))
      phi_j_ = phi_j_[:, None, spin*norb:(spin+1)*norb, :, None]

      numer_ = numer[:, None, None] * phi_i_ * phi_j_

      frac = jnp.linalg.solve(denom[:, None, None], numer_)
      rho_mat = jnp.mean(frac, axis=0) * nspins[spin] * (2 // len(idx))
    else:
      frac = numer_signs * denom_signs * jnp.exp(numer_logs - denom_logs)
      norm_frac = frac / probs
      rho_mat = jnp.mean(
          phi_j[:, None, spin*norb:(spin+1)*norb] *
          phi_i[:, spin*norb:(spin+1)*norb, None] *
          norm_frac[:, None, None],
          axis=0) * nspins[spin] * (2 // len(idx))

    rho_mats.append(rho_mat)

  return jnp.stack(rho_mats)


def get_rho_2(
    batch_network: networks.FermiNetLike,
    params: networks.ParamTree,
    dim: int,
    pos: jnp.ndarray,
    spins: jnp.ndarray,
    charges: jnp.ndarray,
    nspins: Tuple[int, int],
    batch_atoms: jnp.ndarray,
    scf_approx: scf.Scf,
) -> jnp.ndarray:
  if dim != 3:
    raise ValueError('Only implemented for 3D systems')

  def analytical_1s_density(pos):
   # r_ae: Shape (nelectrons, natoms). r_ae[i, j] gives the distance between
   #   electron i and atom j.
   # _, _, r_ae, _ = networks.construct_input_features(pos, batch_atoms.reshape(-1, dim), ndim=dim)

   vmap_features = jax.vmap(networks.construct_input_features, (0, 0))
   _, _, r_ae, _ = vmap_features(pos, batch_atoms)

   # FIXME
   r_ae = r_ae.reshape(-1)
   return jnp.exp(-2 * r_ae) / jnp.pi

  def hf_hydrogen_density(pos):
    _, occ_mos = _eval_mos(
        pos=pos.reshape(-1, dim), scf_approx=scf_approx, nspins=nspins)
    return occ_mos[:, 0] ** 2

  def phi_log(p):
    _, phi_log = scf_approx.eval_slater(p, nspins)
    return phi_log

  _, psi_full_logs = batch_network(
      params,
      pos,
      spins,
      batch_atoms,
      charges,
  )
  numer_value = jnp.zeros(2)

  # Treat spins separately by default
  idx = (0, nspins[0]) if nspins[1] > 0 else (0,)
  # idx = (0,)
  # return f"{nspins[0] = }, {nspins[1] = }, {len(idx) = }"
  for spin, i in enumerate(idx):
    zeroed_pos = pos.at[..., dim*i:dim*(i+1)].set(jnp.zeros(dim))
    _, psi_zero_logs = batch_network(
        params,
        zeroed_pos,
        spins,
        batch_atoms,
        charges,
    )

    def mean_r(q_left, q_right, R):
      batch_size = R.shape[0]
      Q_left = jnp.tile(q_left, (batch_size, 1))
      Q_right = jnp.tile(q_right, (batch_size, 1))
      M = jnp.hstack((Q_left, R, Q_right))
      rho_r = calc_hf_prob(pos=R, scf_approx=scf_approx, nspins=nspins)
      # return f"{pos.shape = }, {rho_r.shape = }, {Q_left.shape = }, {Q_right.shape = }, {R.shape = }, {M.shape = }"
      return jnp.mean(
        jnp.exp(2 * phi_log(M)) / rho_r,
        axis=0
      )

    def mean_q():
      # FIXME: i==0
      # jnp.hsplit(pos, (dim*i, dim*(i+1)))
      split_res = jnp.hsplit(pos, (0, dim))
      Q_left = split_res[0]
      R = split_res[1]
      Q_right = split_res[2]
      # return f"{Q_left.shape = }, {Q_right.shape = }, {R.shape = }"
      return jnp.mean(
        jnp.exp(2 * psi_zero_logs) / jax.vmap(mean_r, in_axes=(0, 0, None))(Q_left, Q_right, R),
        axis=0
      )

    numer_value = numer_value.at[spin].set(
        mean_q()
    #   jnp.mean(
    #       jnp.exp(2 * psi_zero_logs) / rho_r,
    #       axis=0
    #     )
      )

  # denom_value = jnp.mean(jnp.exp(2 * psi_full_logs) / probs, axis=0)
  denom_value = jnp.mean(jnp.exp(2 * (psi_full_logs - phi_log(pos))), axis=0)
  spin_rho = jnp.sum(numer_value) / denom_value

  #if nspins[1] > 0:
  #  spin_rho = jnp.abs(rhos[0] - rhos[1])
  #else:
  #  spin_rho = rhos[0]

  # return f"{occ_mos.shape = }"
  # results = diagonal_elements_of_density_matrix(dim, pos, nspins, scf_approx)
  # return f"{len(results) = }, {results[0].shape = }"
  # return jnp.array([numer_value, numer_value_2, denom_value, denom_value_2, spin_rho])
  # return jnp.array([spin_rho])

  return jnp.array([numer_value[0], numer_value[1], jnp.sum(numer_value), denom_value, spin_rho])


def phi_log(positions: jnp.ndarray, scf_approx: scf.Scf, nspins: Tuple[int, int]):
  _, phi_log = scf_approx.eval_slater(positions, nspins)
  return phi_log


def get_rho_3(
    batch_network: networks.FermiNetLike,
    params: networks.ParamTree,
    dim: int,
    pos: jnp.ndarray,
    spins: jnp.ndarray,
    charges: jnp.ndarray,
    nspins: Tuple[int, int],
    batch_atoms: jnp.ndarray,
    rj_pos: jnp.ndarray,
    probs: jnp.ndarray,
    scf_approx: scf.Scf,
) -> jnp.ndarray:
  if dim != 3:
    raise ValueError('Only implemented for 3D systems')

  _, psi_full_logs = batch_network(
      params,
      pos,
      spins,
      batch_atoms,
      charges,
  )

  # Treat spins separately by default
  idx = (0, nspins[0]) if nspins[1] > 0 else (0,)
  numer_value = jnp.zeros(2)

  for spin, i in enumerate(idx):
    def mean_r(q_left, q_right, R):
      R_size = R.shape[0]
      Q_left = jnp.tile(q_left, (R_size, 1))
      Q_right = jnp.tile(q_right, (R_size, 1))
      M = jnp.hstack((Q_left, R, Q_right))  #.reshape(-1, (nspins[0] + nspins[1]) * dim)
      # return f"{pos.shape = }, {rho_r.shape = }, {Q_left.shape = }, {Q_right.shape = }, {R.shape = }, {M.shape = }"
      probs_for_R = calc_hf_prob(pos=R, scf_approx=scf_approx, nspins=nspins)
      return jnp.mean(
        jnp.exp(2 * phi_log(M, scf_approx, nspins)) / probs_for_R,
        axis=0
      )

    zeroed_pos = pos.at[..., dim*i:dim*(i+1)].set(jnp.zeros(dim))
    _, psi_zero_logs = batch_network(
        params,
        zeroed_pos,
        spins,
        batch_atoms,
        charges,
    )

    split_res = jnp.hsplit(pos, (0, dim))
    Q_left = split_res[0]
    Q_right = split_res[2]

    # radial_positions = jnp.array([(0.2 * i, 0, 0) for i in range(10)])
    # mean_rs = jax.vmap(mean_r, in_axes=(0, 0, None))(Q_left, Q_right, radial_positions)

    mean_rs = jax.vmap(mean_r, in_axes=(0, 0, None))(Q_left, Q_right, rj_pos)
    numer_value = numer_value.at[spin].set(
      jnp.mean(
        jnp.exp(2 * psi_zero_logs) / mean_rs,
        axis=0
      )
    )
    #return f"{pos.shape = }, {Q_left.shape = }, {Q_right.shape = }, {rj_pos.shape = }, {mean_rs.shape = }"

  denom_value = jnp.mean(jnp.exp(2 * (psi_full_logs - phi_log(pos, scf_approx, nspins))), axis=0)
  spin_rho = jnp.sum(numer_value) / denom_value

  return jnp.array([numer_value[0], numer_value[1], jnp.sum(numer_value), denom_value, spin_rho])

  return mean_rs


def get_rho_He(
    batch_network: networks.FermiNetLike,
    params: networks.ParamTree,
    dim: int,
    pos: jnp.ndarray,
    spins: jnp.ndarray,
    charges: jnp.ndarray,
    nspins: Tuple[int, int],
    batch_atoms: jnp.ndarray,
    scf_approx: scf.Scf,
) -> jnp.ndarray:
  if dim != 3:
    raise ValueError('Only implemented for 3D systems')

  _, psi_full_logs = batch_network(
      params,
      pos,
      spins,
      batch_atoms,
      charges,
  )

  # Treat spins separately by default
  idx = (0, nspins[0]) if nspins[1] > 0 else (0,)
  numer_value = jnp.zeros(2)

  for spin, i in enumerate(idx):
    zeroed_pos = pos.at[..., dim*i:dim*(i+1)].set(jnp.zeros(dim))
    _, psi_zero_logs = batch_network(
        params,
        zeroed_pos,
        spins,
        batch_atoms,
        charges,
    )
    probs = calc_hf_prob(pos=pos.reshape(-1, dim), scf_approx=scf_approx, nspins=nspins)
    # For He only:
    probs = probs.reshape(-1, 2)[:, 1 - i]
    numer_value = numer_value.at[spin].set(jnp.mean(jnp.exp(2 * psi_zero_logs) / probs, axis=0))

  denom_value = jnp.mean(jnp.exp(2 * (psi_full_logs - phi_log(pos, scf_approx, nspins))), axis=0)
  spin_rho = jnp.sum(numer_value) / denom_value

  # return jnp.array([numer_value[0], numer_value[1], jnp.sum(numer_value), denom_value, spin_rho])
  return jnp.array([spin_rho])


NDArray = jnp.ndarray

class MOs:
  def __init__(self, mos: NDArray):
    self._mos = mos
    # self._mo_squares = jax.vmap(jax.vmap(...))

  def mo(self, norb: int, nelec: int):
    # checkify?
    # assert norb > 0, "norb must be positive"
    # assert nelec > 0, "nelec must be positive"
    return self._mos[..., nelec - 1, norb - 1]

  def mo_square(self, norb: int, nelec: int):
    return self.mo(norb, nelec) ** 2


def eval_orbitals2(self: scf.Scf,
                  pos: NDArray,
                  nspins: Tuple[int, int]) -> MOs:
    """Evaluates SCF orbitals at a set of positions.

    Args:
      pos: an array of electron positions to evaluate the orbitals at, of shape
        (..., nelec*3), where the leading dimensions are arbitrary, nelec is the
        number of electrons and the spin up electrons are ordered before the
        spin down electrons.
      nspins: tuple with number of spin up and spin down electrons.

    Returns:
      ...
    """
    leading_dims = pos.shape[:-1]
    # split into separate electrons
    pos = jnp.reshape(pos, [-1, 3])  # (batch*nelec, 3)
    mos = self.eval_mos(pos)  # (batch*nelec, nbasis), (batch*nelec, nbasis)
    # Reshape into (batch, nelec, nbasis) for each spin channel.
    mos = [jnp.reshape(mo, leading_dims + (sum(nspins), -1)) for mo in mos]
    # Return (using Aufbau principle) the matrices for the occupied alpha and
    # beta orbitals. Number of alpha electrons given by nspins[0].
    alpha_spin = mos[0][..., :, :nspins[0]]
    beta_spin = mos[1][..., :, :nspins[1]]
    return MOs(jnp.concatenate((alpha_spin, beta_spin), axis=-1))


def eval_orbitals2_alpha(self: scf.Scf,
                         pos: NDArray,
                         nspins: Tuple[int, int]) -> MOs:
    """Evaluates SCF orbitals at a set of positions.

    Args:
      pos: an array of electron positions to evaluate the orbitals at, of shape
        (..., nelec*3), where the leading dimensions are arbitrary, nelec is the
        number of electrons and the spin up electrons are ordered before the
        spin down electrons.
      nspins: tuple with number of spin up and spin down electrons.

    Returns:
      ...
    """
    leading_dims = pos.shape[:-1]
    # split into separate electrons
    pos = jnp.reshape(pos, [-1, 3])  # (batch*nelec, 3)
    mos = self.eval_mos(pos)  # (batch*nelec, nbasis), (batch*nelec, nbasis)
    # Reshape into (batch, nelec, nbasis) for each spin channel.
    mos = [jnp.reshape(mo, leading_dims + (sum(nspins), -1)) for mo in mos]
    # Return (using Aufbau principle) the matrices for the occupied alpha and
    # beta orbitals. Number of alpha electrons given by nspins[0].
    alpha_spin = mos[0][..., :, :nspins[0]]
    return MOs(alpha_spin)


def probs_He(m: MOs, nelec: int):
  # For He: use squared orbital as probability
  return m.mo_square(1 - nelec, 1 - nelec)


def probs_He_2(m: MOs, nelec: int):
  return 0.5 * (m.mo_square(1, 1 - nelec) + m.mo_square(2, 1 - nelec))


# def probs_Li_alpha(m: MOs):
#   return (m.mo_square(1, 2) * m.mo_square(2, 3)
#         + m.mo_square(1, 2) * m.mo_square(3, 3)
#         + m.mo_square(2, 2) * m.mo_square(1, 3)
#         + m.mo_square(2, 2) * m.mo_square(3, 3)
#         + m.mo_square(3, 2) * m.mo_square(1, 3)
#         + m.mo_square(3, 2) * m.mo_square(2, 3)
#         - 2/3 * m.mo(1, 2) * m.mo(2, 3) * m.mo(1, 2) * m.mo(2, 3)
#         - 2/3 * m.mo(2, 2) * m.mo(3, 3) * m.mo(2, 2) * m.mo(3, 3)
#         - 2/3 * m.mo(1, 2) * m.mo(3, 3) * m.mo(1, 2) * m.mo(3, 3)
#         )
# 
# def probs_Li_beta(m: MOs):
#   return (m.mo_square(1, 1) * m.mo_square(2, 2)
#         + m.mo_square(1, 1) * m.mo_square(3, 2)
#         + m.mo_square(2, 1) * m.mo_square(1, 2)
#         + m.mo_square(2, 1) * m.mo_square(3, 2)
#         + m.mo_square(3, 1) * m.mo_square(1, 2)
#         + m.mo_square(3, 1) * m.mo_square(2, 2)
#         - 2/3 * m.mo(1, 1) * m.mo(2, 2) * m.mo(1, 1) * m.mo(2, 2)
#         - 2/3 * m.mo(2, 1) * m.mo(3, 2) * m.mo(2, 1) * m.mo(3, 2)
#         - 2/3 * m.mo(1, 1) * m.mo(3, 2) * m.mo(1, 1) * m.mo(3, 2)
#         )
# def probs_Li_alpha(m: MOs):
#   return ( 
#         m.mo_square(1, 2) * m.mo_square(3, 3)
#         + m.mo_square(2, 2) * m.mo_square(3, 3)
#         + m.mo_square(3, 2) * m.mo_square(1, 3)
#         + m.mo_square(3, 2) * m.mo_square(2, 3)
#         )

# def probs_Li_beta(m: MOs):
#   return (m.mo_square(1, 2) * m.mo_square(2, 3)
#         + m.mo_square(2, 2) * m.mo_square(1, 3)
#         - 2 * m.mo(1, 2) * m.mo(2, 3) * m.mo(1, 3) * m.mo(2, 2)
#         )

def probs_Li(m: MOs):
  return (1/6) * (  # 1/(3!)
          m.mo_square(2, 2) * m.mo_square(3, 3)
        + m.mo_square(1, 2) * m.mo_square(2, 3)
        + m.mo_square(3, 2) * m.mo_square(1, 3)
        + m.mo_square(2, 2) * m.mo_square(1, 3)
        + m.mo_square(1, 2) * m.mo_square(3, 3)
        + m.mo_square(3, 2) * m.mo_square(2, 3)
        - 2 * m.mo(1, 2) * m.mo(2, 3) * m.mo(2, 2) * m.mo(1, 3)
  )


def probs_Li_rohf(m: MOs, el_i: int, el_j: int):
  return (1/3) * (  # 2/(3!)
          m.mo_square(2, el_i) * m.mo_square(1, el_j)
        + m.mo_square(1, el_i) * m.mo_square(2, el_j)
        + m.mo_square(1, el_i) * m.mo_square(1, el_j)
        - m.mo(1, el_i) * m.mo(2, el_j)
        * m.mo(2, el_i) * m.mo(1, el_j)
  )


def probs_Li_rohf_v2(m: MOs, els: NDArray, nelec_factorial: int):
  el_i = els[0]
  el_j = els[1]
  return (2 / nelec_factorial) * (
          m.mo_square(2, el_i) * m.mo_square(1, el_j)
        + m.mo_square(1, el_i) * m.mo_square(2, el_j)
        + m.mo_square(1, el_i) * m.mo_square(1, el_j)
        - m.mo(1, el_i) * m.mo(2, el_j)
        * m.mo(2, el_i) * m.mo(1, el_j)
  )


def probs_Li_squares(m: MOs, orb_pairs: NDArray, elecs: NDArray, nelec_factorial: int):
  def prod_squares(orbs: NDArray):
    squares = jax.vmap(m.mo_square, in_axes=(0, 0))(orbs, elecs)
    return jnp.prod(squares, axis=0)

  prods = jax.vmap(prod_squares, in_axes=(0))(orb_pairs)
  return jnp.sum(prods, axis=0)


def probs_Li_nondiag(m: MOs, orb_pair, elecs: NDArray):
  return ((-1)
    * m.mo(orb_pair[0], elecs[0])
    * m.mo(orb_pair[1], elecs[0])
    * m.mo(orb_pair[0], elecs[1])
    * m.mo(orb_pair[1], elecs[1])
  )


def probs_Li_rohf_v3(m: MOs, orb_pairs: NDArray, elecs: NDArray, nelec_factorial: int):
  return (2 / nelec_factorial) * (
      probs_Li_squares(m, orb_pairs, elecs, nelec_factorial)
    + probs_Li_nondiag(m, (1, 2), elecs)
  )


def probs_Li_uhf(m: MOs, orb_pairs: NDArray, elecs: NDArray, nelec_factorial: int):
  return (1 / nelec_factorial) * (
      probs_Li_squares(m, orb_pairs, elecs, nelec_factorial)
    + 2 * probs_Li_nondiag(m, (1, 2), elecs)
  )


def probs_Li_factorized(m: MOs):
  return (1/2) * (
          m.mo_square(2, 2) * m.mo_square(1, 3)
        + m.mo_square(1, 2) * m.mo_square(1, 3)
  )


def get_rho_He_2(
    batch_network: networks.FermiNetLike,
    params: networks.ParamTree,
    dim: int,
    pos: jnp.ndarray,
    spins: jnp.ndarray,
    charges: jnp.ndarray,
    nspins: Tuple[int, int],
    batch_atoms: jnp.ndarray,
    scf_approx: scf.Scf,
) -> jnp.ndarray:
  if dim != 3:
    raise ValueError('Only implemented for 3D systems')

  _, psi_full_logs = batch_network(
      params,
      pos,
      spins,
      batch_atoms,
      charges,
  )

  # Treat spins separately by default
  idx = (0, nspins[0]) if nspins[1] > 0 else (0,)
  # idx = (0, 1)
  numer_value = jnp.zeros(2)
  mos = eval_orbitals2(scf_approx, pos, nspins)

  for spin, i in enumerate(idx):
    zeroed_pos = pos.at[..., dim*i:dim*(i+1)].set(jnp.zeros(dim))
    _, psi_zero_logs = batch_network(
        params,
        zeroed_pos,
        spins,
        batch_atoms,
        charges,
    )
    # probs = probs_He(mos, i+1)
    # probs = probs_He_2(mos, i+1)
    # numer_value = numer_value.at[spin].set(jnp.mean(jnp.exp(2 * psi_zero_logs) / probs, axis=0))
    if spin == 0:
      probs = probs_Li_rohf(mos)
      # probs = jnp.exp(2 * phi_log(zeroed_pos, scf_approx, nspins))
      numer_value = numer_value.at[spin].set(jnp.mean(jnp.exp(2 * psi_zero_logs) / probs, axis=0))
    elif spin == 1:
      # probs = probs_Li(mos)
      numer_value = numer_value.at[spin].set(0)

  denom_value = jnp.mean(jnp.exp(2 * (psi_full_logs - phi_log(pos, scf_approx, nspins))), axis=0)
  spin_rho = 3 * jnp.sum(numer_value) / denom_value

  return jnp.array([numer_value[0], numer_value[1], denom_value, spin_rho])
  return jnp.array([spin_rho])


def irange(stop: int):
  return range(1, 1 + stop)


def int_factorial(x: int):
  return jnp.round(jss.factorial(x))


def get_rho_Li_all_zero(
    batch_network: networks.FermiNetLike,
    params: networks.ParamTree,
    dim: int,
    pos: jnp.ndarray,
    spins: jnp.ndarray,
    charges: jnp.ndarray,
    nspins: Tuple[int, int],
    batch_atoms: jnp.ndarray,
    scf_approx: scf.Scf,
) -> jnp.ndarray:
  if dim != 3:
    raise ValueError('Only implemented for 3D systems')

  _, psi_full_logs = batch_network(
      params,
      pos,
      spins,
      batch_atoms,
      charges,
  )

  nelec = nspins[0] + nspins[1]
  if scf_approx.restricted:
    orb_pairs_iter = itertools.combinations(itertools.chain(irange(nspins[0]), irange(nspins[1])), 2)
    mos = eval_orbitals2_alpha(scf_approx, pos, nspins)
    probs_fun = probs_Li_rohf_v3
  else:
    orb_pairs_iter = itertools.permutations(irange(nelec), 2)
    mos = eval_orbitals2(scf_approx, pos, nspins)
    probs_fun = probs_Li_uhf

  orb_pairs = jnp.array(tuple(orb_pairs_iter))
  nelec_factorial = int_factorial(nelec)
  numer_value = jnp.zeros(nelec)

  # for spin, i in enumerate(idx):
  el_numbers = range(nelec)
  for i in el_numbers:
    zeroed_pos = pos.at[..., dim*i:dim*(i+1)].set(jnp.zeros(dim))
    _, psi_zero_logs = batch_network(
        params,
        zeroed_pos,
        spins,
        batch_atoms,
        charges,
    )
    el_numbers_without_i = jnp.array([n + 1 for n in el_numbers if n != i])
    probs = probs_fun(mos, orb_pairs, el_numbers_without_i, nelec_factorial)
    numer_value = numer_value.at[i].set(jnp.mean(jnp.exp(2 * psi_zero_logs) / probs, axis=0))

  denom_value = jnp.mean(jnp.exp(2 * (psi_full_logs - phi_log(pos, scf_approx, nspins))), axis=0)
  spin_rho = jnp.sum(numer_value) / (nelec * denom_value)

  return jnp.array([*numer_value, denom_value, spin_rho])
