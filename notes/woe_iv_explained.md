# Weight of Evidence (WoE) and Information Value (IV)

## The core idea

WoE and IV answer one question: **does this variable help separate good payers from defaulters?**

Suppose clients are split into age ranges. If defaulters concentrate in some ranges and good payers in others, age carries information about risk. If both groups spread the same way across every range, age adds nothing.

WoE and IV measure exactly this. WoE looks at **each bin**; IV summarises **the whole variable** in a single number.

## WoE (Weight of Evidence): the weight of each bin

For each bin, two proportions are compared:

- **% goods:** of all good payers in the data, how many fall in this bin?
- **% bads:** of all defaulters in the data, how many fall in this bin?

$$\text{WoE} = \ln\left(\frac{\%\ \text{goods}}{\%\ \text{bads}}\right)$$

Example with round numbers. The data has 1,000 goods and 100 bads, and the bin "1 late payment" has 50 goods and 20 bads:

- % goods = 50 / 1,000 = **5%**
- % bads = 20 / 100 = **20%**
- WoE = ln(5% / 20%) = ln(0.25) ≈ **−1.39**

How to read it:

| WoE | Meaning |
|---|---|
| **0** | The bin has the same mix of goods and bads as the whole data. It says nothing. |
| **Negative** | Proportionally more bads here. The bin is **riskier** than average. |
| **Positive** | Proportionally more goods. The bin is **safer** than average. |

The logarithm makes "4 times more bads" and "4 times more goods" the same size, only with opposite signs (−1.39 and +1.39).

## IV (Information Value): the power of the variable

IV adds up the contribution of each bin:

$$\text{IV} = \sum_{\text{bins}} (\%\ \text{goods} - \%\ \text{bads}) \times \text{WoE}$$

Both factors of the product always have the same sign, so every bin contributes a **positive** value. A bin contributes a lot when it is very different from the average (large WoE) **and** holds a relevant share of the population (large difference in proportions). A very risky bin with only 2 clients weighs little.

Usual interpretation ranges:

| IV | Predictive power |
|---|---|
| < 0.02 | not useful |
| 0.02 – 0.10 | weak |
| 0.10 – 0.30 | medium |
| 0.30 – 0.50 | strong |
| > 0.50 | suspicious: check for leakage or overly fine bins |

## Why it is useful in this project

**1. It compares very different variables.** Age in years, utilisation as a ratio, income in currency, counts of late payments: correlation does not compare such different scales well. IV puts them all on the same ruler.

**2. It captures non-linear relationships.** Risk falls with age until about 65 and then changes behaviour, and utilisation shows a jump near 0.3. Spearman correlation only captures monotonic relationships. IV works by bins and captures any shape.

**3. It handles outliers well.** A utilisation of 50,000 simply falls into the last bin ("> 5") without distorting anything. This matters while the outliers in Util and Debt are still to be confirmed.

**4. It treats missing values as information.** At this stage missing income is still `NaN`, so it becomes its own "Missing" bin with its own WoE. This shows whether **not reporting income** already says something about risk, which connects with a pattern seen earlier: clients with Debt > 10 almost always have missing income and a lower default rate.

**5. It is standard practice in credit risk.** WoE and IV are the basis of traditional scorecards. Using them places this analysis within the usual tools of the credit-risk field, not only those of general machine learning.

## Limitations

- **It is univariate.** IV measures each variable in isolation. If Late30_59, Late60_89 and Late90 all have high IV, that does not mean the three together add information, since they may be redundant. This is why the correlation analysis is still needed.
- **It depends on the bins.** Bins that are too fine inflate IV; bins that are too wide hide patterns. Bins with few clients produce unstable WoE values.
- **It is not causality.** IV shows that a variable separates goods from bads, not why it does so.
- **A very high IV needs checking.** An IV above 0.5 may reflect real predictive power (as expected for the late-payment variables), but it may also signal leakage. The question to ask is: would this information be available at the time the credit decision is made?