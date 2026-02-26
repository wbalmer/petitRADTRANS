import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jax import jit, lax

class LogLikelihood:
    def __init__(self,
                 model_spectrum: ArrayLike,
                 data_spectrum: ArrayLike,
                 data_uncertainties: ArrayLike = None,
                 data_covariance: ArrayLike = None,
                 n_free_parameters: int = 0):
        """
        Initializes the LogLikelihood calculator.

        Args:
            model_spectrum: The model spectrum, already convolved and rebinned to match data_spectrum.
            data_spectrum: The observed data spectrum, already appropriately scaled.
            n_free_parameters: Number of free parameters in the model for DoF calculation.
            data_uncertainties: 1D array of uncertainties for the data points.
                                Required if data_covariance is None.
            data_covariance: 2D covariance matrix for the data points.
                             Required if data_uncertainties is None.
        """
        if data_uncertainties is None and data_covariance is None:
            raise ValueError("Either data_uncertainties or data_covariance must be provided.")
        if data_spectrum.shape != model_spectrum.shape:
            raise ValueError("The data and model model must have the same shape.")

        self.model_spectrum = jnp.asarray(model_spectrum)
        self.data_spectrum = jnp.asarray(data_spectrum)

        self.n_data_points = self.data_spectrum.shape[0]

        self.n_free_parameters = n_free_parameters

        if data_uncertainties is not None:
            self.data_uncertainties = jnp.asarray(data_uncertainties)
            if self.data_uncertainties.shape != self.model_spectrum.shape:
                raise ValueError(f"data_uncertainties must have the same shape as the model_spectrum!)")
            #if jnp.any(self.data_uncertainties <= 0): TODO: move to data.py
            #    raise ValueError("Uncertainties must be positive.")
        else:
            self.data_uncertainties = None

        if data_covariance is not None:
            self.data_covariance = jnp.asarray(data_covariance)
            if self.data_covariance.shape != (self.n_data_points, self.n_data_points):
                raise ValueError(f"Shape of data_covariance ({self.data_covariance.shape}) "
                                 f"is not ({self.n_data_points}, {self.n_data_points})")


        # Precompute terms for likelihood calculation
        if data_covariance is not None:
            #try:
            # TODO: This assumes 1D spectrum. Need to extend to ND data.
            self.inv_covariance = jnp.linalg.inv(self.data_covariance)
            _sign, self.log_det_covariance = jnp.linalg.slogdet(self.data_covariance)
            if _sign <= 0:
                raise ValueError("Covariance matrix determinant is not positive. Log determinant is undefined.")
            self._log_norm_constant = -0.5 * (self.n_data_points * jnp.log(2.0 * jnp.pi) +
                                              self.log_det_covariance)
            # TODO: re-enable exception handling for JAX
            # except jnp.linalg.LinAlgError:
            #    raise ValueError("Covariance matrix is singular and cannot be inverted.")
        else:
            self.data_covariance = jnp.diag(self.data_uncertainties**2)
            self.inv_covariance = jnp.linalg.inv(self.data_covariance)
            _sign, self.log_det_covariance = jnp.linalg.slogdet(self.data_covariance)
            self._log_norm_constant = -0.5 * jnp.sum(jnp.log(2.0 * jnp.pi * self.data_uncertainties**2))

    def get_chi_squared(self) -> float:
        chi_squared = lax.cond(
            jnp.sum(self.data_covariance) != 0.0,
            self._calculate_chi_squared_covariance,
            self._calculate_chi_squared_uncertainties
            )
        return chi_squared

    def _calculate_chi_squared_covariance(self):
        residual = self.data_spectrum - self.model_spectrum
        return jnp.dot(residual, jnp.dot(self.inv_covariance, residual))

    def _calculate_chi_squared_uncertainties(self):
        residual = self.data_spectrum - self.model_spectrum
        return jnp.sum((residual / self.data_uncertainties)**2)

    def calculate_log_likelihood(self) -> float:
        chi_squared = self.get_chi_squared()
        # self._log_norm_constant was pre-calculated in __init__
        # It already contains the -0.5 * (N*log(2pi) + log(det(Cov))) part for covariance case
        # or -0.5 * sum(log(2pi*sigma_i^2)) for uncertainties case.
        # Standard Gaussian log-likelihood:
        # log_L = -0.5 * sum_i((x_i - mu_i)^2 / sigma_i^2) - 0.5 * sum_i(log(2*pi*sigma_i^2))
        # or for covariance: log_L = -0.5 * (x-mu)^T Cov^-1 (x-mu) - 0.5 * log(det(2*pi*Cov))
        # The chi-squared term is (x-mu)^T Cov^-1 (x-mu) or sum((x-mu)^2/sigma^2).
        # So, log_L = -0.5 * chi_squared + self._log_norm_constant
        log_l = self._log_norm_constant - 0.5 * chi_squared
        return log_l

    def get_reduced_chi_squared(self) -> float:
        """
        Calculates the reduced chi-squared value.

        Returns:
            The reduced chi-squared value. Returns jnp.nan if degrees of freedom is <= 0.
        """
        dof = self.n_data_points - self.n_free_parameters
        if dof <= 0:
            # Returning NaN for invalid degrees of freedom
            return jnp.nan

        chi_squared = self.get_chi_squared()
        return chi_squared / dof

    def get_noise_normalization_constant(self) -> float:
        """
        Returns the noise-related normalization constant of the log-likelihood.
        This is -0.5 * (N*log(2pi) + log(det(Cov))) or -0.5 * sum(log(2pi*sigma_i^2)).
        This term does not depend on the model-data residuals.

        Returns:
            The pre-computed noise normalization constant.
        """
        return self._log_norm_constant

    def log_likelihood(self, beta=None, beta_mode='multiply'):
        return self._log_likelihood(
            self.model_spectrum,
            self.data_spectrum,
            self.data_uncertainties,
            beta=beta,
            beta_mode=beta_mode)

    @staticmethod
    def _log_likelihood(model, data, uncertainties, beta=None, beta_mode='multiply'):
        """Calculate the log-likelihood between the model and the data.

        The spectrum model must be on the same wavelength grid than the data.

        From Gibson et al. 2020 (https://doi.org/10.1093/mnras/staa228). Constant terms are dropped and constant
        coefficients are set to 1.
        The 'add' beta mode comes from Line et al. 2015 (DOI 10.1088/0004-637X/807/2/183).

        Set:
            chi2(A, B) = sum(((f_i - A * m_i) / (B * sig_i)) ** 2),  implicit sum on i
        with f_i the data, m_i the model, and sig_i the uncertainty; where "i" denotes wavelength/time variation.
        Starting from (implicit product on i):
            L        = prod(1 / sqrt(2 * pi * B * sig_i ** 2) * exp(-1/2 * sum(((f_i - A * m_i) / (B * sig_i)) ** 2))),
            => L     = prod( 1 / sqrt(2 * pi * B * sig_i ** 2) * exp(-1/2 * chi2(A, B)) ),
            => ln(L) = -N/2 * ln(2 * pi) - N * ln(B) - sum(ln(sig_i)) - 1/2 * chi2(A, B).
        Dropping constant terms:
            ln(L)^* = - N * ln(B) - 1/2 * chi2(A, B).

        B can be automatically optimised by nulling the ln(L) partial derivative with respect to B.
        Using the best estimator of B instead of the true value:
            d_ln(L) / d_B = - N / B + 1 / B ** 3 * chi2(A, B=1) = 0,
            => B = sqrt( 1/N *  chi2(A, B=1)).
        Replacing:
            ln(L)^**= - N * ln(B) - 1/2 * chi2(A, B),
                    = - N * ln(sqrt(1/N * chi2(A, B=1))) - 1/2 * sum(((f_i - A * m_i) / (B * sig_i)) ** 2),
                    = - N/2 * ln(1/N * chi2(A, B=1)) - 1/2 / B ** 2 * sum(((f_i - A * m_i) / sig_i) ** 2),  B cst with i
                    = - N/2 * ln(1/N * chi2(A, B=1)) - 1/2 * N / chi2(A, B=1) * chi2(A, B=1),
                    = - N/2 * ln(1/N * chi2(A, B=1)) - N/2.
        Dropping constant terms:
            ln(L)^*** = - N/2 * ln(1/N * chi2(A, B=1)).

        Args:
            model: numpy.ndarray
                The model flux in the same units as the data.
            data: numpy.ndarray
                The data.
            uncertainties: numpy.ndarray
                The uncertainties on the data.
            beta: float, optional
                Noise scaling coefficient. If None, the noise scaling is "automatically optimised".
            beta_mode: string, optional


        Returns:
            logL : float
                The log likelihood of the model given the data.
        """
        if beta is None:
            # "Automatically optimise" for beta
            chi2 = data - model
            chi2 /= uncertainties
            chi2 *= chi2
            chi2 = chi2.sum()

            return - 0.5 * data.size * jnp.log(chi2 / data.size)
        else:
            # Classical log-likelihood
            if beta_mode == 'multiply':
                uncertainties = uncertainties * beta
                penalty_term = - data.size * jnp.log(beta)
            elif beta_mode == 'add':
                uncertainties = uncertainties + beta
                penalty_term = - jnp.sum(jnp.log(uncertainties))
            else:
                raise ValueError(f"beta mode must be 'multiply'|'add', but was '{beta_mode}'")

            chi2 = data - model
            chi2 /= uncertainties
            chi2 *= chi2
            chi2 = chi2.sum()

            return - 0.5 * chi2 + penalty_term

    @staticmethod
    def log_likelihood2chi2(log_likelihood: float) -> float:
        return -2 * log_likelihood
