.. _models:

Physical Models
===============

All models live in the :mod:`qens.models` subpackage and are importable
directly from the top-level ``qens`` namespace.

Three Built-in Forward Models
------------------------------

Pass these names to :func:`~qens.fitting.find_map` and
:func:`~qens.sampling.run_mcmc`.

.. list-table::
   :header-rows: 1
   :widths: 25 15 25 35

   * - Name
     - Parameters
     - Parameter names
     - When to use
   * - ``translation_only``
     - 2
     - D*, ⟨u²⟩
     - Atoms, monatomic ions — no rotation present.
   * - ``isotropic_rotor``
     - 3
     - D*, ⟨u²⟩, D\ :sub:`r`
     - Rotation present but no anisotropy expected (spherical molecules).
   * - ``anisotropic_rotor``
     - 4
     - D*, ⟨u²⟩, D\ :sub:`t`, D\ :sub:`s`
     - Oblate axial rotors — e.g. benzene, toluene, naphthalene.

Translational HWHM Functions
------------------------------

These return the quasi-elastic linewidth Γ(*Q*) in meV and are used
directly for the classical HWHM workflow.

.. autofunction:: qens.models.fickian_hwhm

.. autofunction:: qens.models.ce_hwhm

.. autofunction:: qens.models.ss_hwhm

Rotational Structure Factor
----------------------------

The full incoherent scattering function (Sears/Bée formalism):

.. math::

   S_\text{inc}(Q,\omega) = e^{-Q^2\langle u^2\rangle/3}
   \left[j_0^2(QR)\,L_T(\omega)
   + \sum_{l=1}^{2}(2l+1)\,j_l^2(QR)\,\bigl(L_T \ast L_R^{(l)}\bigr)(\omega)
   \right] + b(Q)

where :math:`L_T` is a Lorentzian of width :math:`\hbar D^* Q^2`,
:math:`L_R^{(l)}` are rotational Lorentzians, and :math:`j_l(QR)` are
spherical Bessel functions evaluated at molecular radius *R*
(default 2.48 Å for benzene).

.. autofunction:: qens.models.bessel_weights

.. autofunction:: qens.models.rot_widths_isotropic

.. autofunction:: qens.models.rot_widths_anisotropic

For the anisotropic rotor the widths are:

.. math::

   \Gamma_1 = \hbar(D_s + D_t), \quad
   \Gamma_2 = 6\hbar D_t, \quad
   \Gamma_3 = \hbar(4D_s + 2D_t)

.. autofunction:: qens.models.predict_sqw

Lineshape Primitives
---------------------

.. autofunction:: qens.models.lorentz

.. autofunction:: qens.models.gnorm

.. autofunction:: qens.models.lorentz_sum

Model Registry
---------------

.. autofunction:: qens.models.available_models

.. autofunction:: qens.models.get_model

.. autofunction:: qens.models.register_model

For how to write and register a custom model, see :doc:`custom_models`.
