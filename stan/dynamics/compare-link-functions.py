import numpy as np
import pandas as pd
from plotnine import ggplot, aes, geom_line, labs, theme_minimal

x_vals = np.arange(201) / 40
df = pd.DataFrame({
    'x': x_vals,
    'exp(-x)': np.exp(-x_vals),
    '1/(1+x**2)': 1 / (1 + x_vals**2)
})
df_long = pd.melt(df, id_vars='x', value_vars=['exp(-x)', '1/(1+x**2)'],
                      var_name='Function', value_name='y')
plot = (
    ggplot(df_long, aes(x='x', y='y', color='Function')) +
    geom_line() +
    labs(title='Comparison of exp(-x) and 1/(1+x**2)', x='x', y='y') +
    theme_minimal()
)
plot.save('compare-link-functions.jpeg')
plot.show()

