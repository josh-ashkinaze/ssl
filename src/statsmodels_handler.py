import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

class StatsmodelsHandler:
    """
    A wrapper for summarizing and visualizing fitted `statsmodels` regression models.

    This class provides clean LaTeX summaries and coefficient plots with 95% confidence intervals,
    optionally applying custom cleaning to variable names and supporting categorical encodings.

    Attributes:
        model (statsmodels model): A fitted statsmodels model (e.g., from OLS, Logit).

    Example:
        >>> import statsmodels.formula.api as smf
        >>> from statsmodels_handler import StatsmodelsHandler, clean_var_name_from_formula
        >>> mod = smf.ols("y ~ C(condition, Treatment(reference='control'))", data=df).fit()
        >>> handler = StatsmodelsHandler(mod)
        >>> handler.to_latex()
        >>> handler.plot(clean_var_name=clean_var_name_from_formula)
        >>> plt.xlabel("Coefficient")
        >>> plt.ylabel("Condition")
        >>> plt.title("Estimated Effects with 95% CIs")
        >>> plt.show()
    """

    def __init__(self, model):
        """
        Initialize with a fitted statsmodels model.

        Args:
            model (statsmodels model): A fitted model from statsmodels.
        """
        self.model = model
        self.params = model.params
        self.errors = model.bse
        self.pvalues = model.pvalues
        self.conf_int = model.conf_int()

    def to_latex(self, beta_digits=2, se_digits=2, p_digits=3, ci_digits=2, print_sci_not=False):
        """
        Print LaTeX-formatted summaries of all parameters in the model.

        Args:
            beta_digits (int): Decimal places for coefficients.
            se_digits (int): Decimal places for standard errors.
            p_digits (int): Decimal places for p-values.
            ci_digits (int): Decimal places for confidence intervals.
            print_sci_not (bool): Use scientific notation for very small p-values.
        """
        for param_name, beta in self.params.items():
            safe_param_name = param_name.replace('_', '\\_')
            se = self.errors[param_name]
            p = self.pvalues[param_name]
            ci_lower, ci_upper = self.conf_int.loc[param_name]

            if p < 0.001:
                p_formatted = f"= {p:.2e}" if print_sci_not else "<0.001"
            else:
                p_formatted = f"= {p:.{p_digits}f}"

            summary = (f"{safe_param_name}: $\\beta = {beta:.{beta_digits}f}$, "
                       f"$SE = {se:.{se_digits}f}$, $p {p_formatted}$, "
                       f"$95\\% CI = [{ci_lower:.{ci_digits}f}, {ci_upper:.{ci_digits}f}]$")
            print(summary)

    def plot(self, exp=False, figsize=(10, 6), color_dict=None, drop_intercept=True, clean_var_name=None):
        """
        Plot model coefficients with confidence intervals and optional color coding.

        Args:
            exp (bool): If True, exponentiate coefficients (e.g., for odds ratios).
            figsize (tuple): Size of the plot figure.
            color_dict (dict): Colors for significance: {'ns', 'neg', 'pos'}.
            drop_intercept (bool): If True, omit the intercept term from the plot.
            clean_var_name (function): Optional function to clean variable names for plotting.

        Returns:
            matplotlib.pyplot object: The resulting plot.
        """
        color_dict = color_dict or {'ns': 'gray', 'neg': '#D41876', 'pos': '#89DAFF'}
        sum_res = pd.read_html(self.model.summary().tables[1].as_html(), header=0, index_col=0)[0].reset_index()

        if "P>|t|" in sum_res.columns:
            stat_type = "t"
        elif "P>|z|" in sum_res.columns:
            stat_type = "z"
        else:
            raise ValueError("Could not identify p-value column.")

        if exp:
            threshold = 1
            for v in ['coef', '[0.025', '0.975]']:
                sum_res[v] = np.exp(sum_res[v])
        else:
            threshold = 0

        sum_res = sum_res.rename(columns={'index': 'var'})
        if drop_intercept:
            sum_res = sum_res[sum_res['var'] != 'Intercept']

        if clean_var_name:
            sum_res['var'] = sum_res['var'].apply(clean_var_name)

        sum_res = sum_res.sort_values(by='coef', ascending=False)
        sum_res['var'] = sum_res['var'].str.replace("T.", "")
        sum_res['error_lower'] = sum_res['coef'] - sum_res['[0.025']
        sum_res['error_upper'] = sum_res['0.975]'] - sum_res['coef']

        sum_res['color'] = sum_res.apply(
            lambda row: color_dict['pos'] if (row[f'P>|{stat_type}|'] < 0.05 and row['coef'] > threshold)
            else color_dict['neg'] if (row[f'P>|{stat_type}|'] < 0.05 and row['coef'] < threshold)
            else color_dict['ns'], axis=1)

        plt.figure(figsize=figsize)
        for _, row in sum_res.iterrows():
            plt.errorbar(row['coef'], row['var'],
                         xerr=[[row['error_lower']], [row['error_upper']]],
                         fmt='o', capsize=5, color=row['color'])
        ax = sns.pointplot(y='var', x='coef', data=sum_res, capsize=0.1,
                           palette=sum_res['color'].tolist())
        plt.axvline(x=threshold, color='gray', linestyle='dashed')
        return plt


def clean_var_name_from_formula(term_str):
    """
    Converts a statsmodels formula term like:
    'C(var_name, Treatment(reference="baseline"))[level]'
    into a clean label for plotting.

    Args:
        term_str (str): Variable name from the model output.

    Returns:
        str: Readable string such as "var_name = level\n(reference = baseline)"
    """
    pattern = r'C\(([^,]+),\s*Treatment\(reference="([^"]+)"\)\)\[([^\]]+)\]'
    match = re.match(pattern, term_str.strip())
    if not match:
        return term_str
    question, reference, coef = match.groups()
    clean = lambda x: x.replace('_', ' ')
    return f"{clean(question)} = {clean(coef)}\n(reference = {clean(reference)})"
