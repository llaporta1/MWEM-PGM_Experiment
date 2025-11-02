# minimal workload holder compatible with your current call: Marginals(domain, marginals)
class Marginals:
    def __init__(self, domain, marginals):
        self.domain = domain
        # list of tuples like [("ZIP_reduced",), ("ZIP_reduced","Age_bin"), ...]
        self.marginals = list(marginals)
