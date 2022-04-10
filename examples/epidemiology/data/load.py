import pkg_resources

import pandas as pd


def load_epidemiology():
  """Epidemiological data.

    Epidemiological data relating Human Papilloma Virus prevalence and Cervical
    Cancer incidence.
    A common dataset analised in the literature of modular inference.

    References:
        Plummer, M. (2015).
        Cuts in Bayesian graphical models.
        Statistics and Computing,
        25(1), 37-43.

        Maucort-Boulch, D., Franceschi, S., & Plummer, M. (2008).
        International correlation between human papillomavirus prevalence and cervical cancer incidence.
        Cancer Epidemiology and Prevention Biomarkers,
        17(3), 717-720.
    """
  filename = 'epidemiology_plummer.csv'
  stream = pkg_resources.resource_stream(__name__, filename)
  dataset = pd.read_csv(stream)
  return dataset
