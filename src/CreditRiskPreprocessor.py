import numpy as np

class CreditRiskPreprocessor():

    def _get_imputation_values(self, df):
        """
        Calculates imputation values for missing data.

        The mean is calculated for 'Income', while the mode is calculated
        for 'No.Dep'. The resulting values are stored as instance attributes.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame used to calculate the imputation values.

        Returns
        -------
        None
        """

        self._income_imputation = df["Income"].mean()
        self._nodep_imputation = df["No.Dep"].mode()[0]

    def _impute_missing_values(self):
        """
        Imputes missing values in the 'Income' and 'No.Dep' columns using
        previously calculated imputation values.

        The imputation values are stored in the instance attributes
        '_income_imputation' and '_nodep_imputation'.

        Returns
        -------
        pandas.DataFrame
            DataFrame with missing values imputed.
        """

        self._df["Income"] = self._df["Income"].fillna(
            self._income_imputation
        )
        self._df["No.Dep"] = self._df["No.Dep"].fillna(
            self._nodep_imputation
        )

        return self._df


    def _set_features(self):
        """
        Creates additional features from existing columns.

        Features created
        ----------------
        AnyLate : int
            Indicates whether the customer has any late payments.
        TotalLate : numeric
            Total number of late payments across the three delinquency periods.
        LateScore : numeric
            Weighted score of late payments, assigning higher weights to
            longer delinquency periods.
        UtilAbove100 : int
            Indicates whether credit utilization exceeds 1 (100%).

        Returns
        -------
        pandas.DataFrame
            DataFrame with the additional features.
        """

        self._df["AnyLate"] = (
            (self._df["Late30_59"] > 0)
            | (self._df["Late60_89"] > 0)
            | (self._df["Late90"] > 0)
        ).astype(int)

        self._df["TotalLate"] = (
            self._df["Late30_59"]
            + self._df["Late60_89"]
            + self._df["Late90"]
        )

        self._df["LateScore"] = (
            self._df["Late30_59"]
            + 2 * self._df["Late60_89"]
            + 3 * self._df["Late90"]
        )

        self._df["UtilAbove100"] = np.where(
            self._df["Util"] > 1, 1, 0
        )

        return self._df

    def fit(self, df):
        """
        Learns imputation values from the provided training data.

        Parameters
        ----------
        df : pandas.DataFrame
            Training data used to calculate the imputation values.

        Returns
        -------
        CreditRiskPreprocessor
            Fitted preprocessor instance.
        """

        self._get_imputation_values(df)

        return self

    def transform(self, df):
        """
        Imputes missing values and creates additional features.

        Uses the imputation values previously learned by the 'fit' method.
        The input DataFrame is copied before being modified.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame to preprocess.

        Returns
        -------
        pandas.DataFrame
            Preprocessed DataFrame with missing values imputed and
            additional features created.
        """

        self._df = df.copy()

        self._df = self._impute_missing_values()
        self._df = self._set_features()

        return self._df
