class LambdaTuner:
    def __init__(self, train_one_lambda_pair, logger, type1_error_lowerbound, type1_error_upperbound, max_tuning_tries, initial_increment_factor, lambda_min, lambda_max):
        self.logger = logger
        self.type1_error_lowerbound = type1_error_lowerbound
        self.type1_error_upperbound = type1_error_upperbound
        self.max_tuning_tries = max_tuning_tries
        self.initial_increment_factor = initial_increment_factor
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.train_one_lambda_pair = train_one_lambda_pair

    def fine_tune_lambda(self, lambda_source, lambda_normal, prior_type1_error=None, tune_lambda='normal'):
        if not prior_type1_error:
            prior_type1_error = (self.type1_error_lowerbound + self.type1_error_upperbound) / 2

        if tune_lambda not in ['normal', 'source']:
            raise ValueError("tune_lambda must be either 'normal' or 'source'")

        if tune_lambda == 'normal':
            self.logger.log_lambda_updates(f"Fixing lambda_source at {lambda_source} and fine tuning lambda_normal (starting with {lambda_normal}).")
        else:
            self.logger.log_lambda_updates(f"Fixing lambda_normal at {lambda_normal} and Fine tuning lambda_source (starting with {lambda_source}).")

        increment_factor = self.initial_increment_factor
        tries, tries_no_overshooting = 0, 0

        while tries < self.max_tuning_tries:
            evaluation_error_rates = self.train_one_lambda_pair(lambda_source, lambda_normal)
            type1_error, type2_error = evaluation_error_rates[0], evaluation_error_rates[1]

            if self._is_within_error_bounds(type1_error):
                return lambda_source, lambda_normal, type1_error, type2_error

            increment_factor, tries_no_overshooting = self._adjust_increment_factor(
                prior_type1_error, type1_error, increment_factor, tries_no_overshooting
            )

            lambda_source, lambda_normal = self._adjust_lambdas(
                lambda_source, lambda_normal, increment_factor, type1_error, tune_lambda
            )

            if self._is_lambda_out_of_bounds(lambda_source, lambda_normal, tune_lambda):
                self.logger.log_lambda_updates(f"Lambda_{tune_lambda} of {lambda_normal if tune_lambda == 'normal' else lambda_source} too small or too large. Lambda did not converge.")
                break

            prior_type1_error = type1_error
            tries += 1

        if tries == self.max_tuning_tries:
            self.logger.log_lambda_updates(f"Max tries of {self.max_tuning_tries} reached. Lambda did not converge.")
        return None

    def _adjust_lambdas(self, lambda_source, lambda_normal, increment_factor, type1_error, tune_lambda):
        if tune_lambda == 'normal':
            lambda_normal = lambda_normal * (1 + increment_factor) if type1_error > self.type1_error_upperbound else lambda_normal * (1 - increment_factor)
        else:
            lambda_source = lambda_source * (1 + increment_factor) if type1_error <= self.type1_error_upperbound else lambda_source * (1 - increment_factor)

        self.logger.log_lambda_updates(f"Applying increment_factor of {1 + increment_factor if type1_error > self.type1_error_upperbound else 1 - increment_factor} to lambda_{tune_lambda}.")

        return lambda_source, lambda_normal

    def _adjust_increment_factor(self, prior_type1_error, type1_error, increment_factor, tries_no_overshooting):
        if (prior_type1_error <= self.type1_error_lowerbound and type1_error >= self.type1_error_upperbound) or \
          (prior_type1_error >= self.type1_error_upperbound and type1_error <= self.type1_error_lowerbound):
            increment_factor /= 2
            tries_no_overshooting = 0
            self.logger.log_lambda_updates(f"Over/undershooting. Halving increment_factor to {increment_factor}.")
        else:
            tries_no_overshooting += 1
            if tries_no_overshooting >= 5:
                increment_factor = (1 - increment_factor) / 2 + increment_factor
                self.logger.log_lambda_updates(f"5+ attempts without over/undershooting. Increasing increment_factor to {increment_factor}.")

        return increment_factor, tries_no_overshooting

    def _is_within_error_bounds(self, type1_error):
        return self.type1_error_lowerbound <= type1_error <= self.type1_error_upperbound

    def _is_lambda_out_of_bounds(self, lambda_source, lambda_normal, tune_lambda):
        return (tune_lambda == 'normal' and (lambda_normal < self.lambda_min or lambda_normal > self.lambda_max)) or \
              (tune_lambda == 'source' and (lambda_source < self.lambda_min or lambda_source > self.lambda_max))
