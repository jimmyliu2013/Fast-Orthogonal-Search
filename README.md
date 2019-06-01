# Fast-Orthogonal-Search
Implementation of Fast Orthogonal Search algorithm

FOS is used to identify an ARMA model. reference: [Applications of fast orthogonal search: Time-series analysis and resolution of signals in noise](https://link.springer.com/article/10.1007/BF02368043)

usage:

 1.model = FOSModel(max_delay_in_input, max_delay_in_output, max_order, max_m, mse_reduction_threshold), where max_m is the max number of  terms, mse_reduction_threshold is the early-stop criteria
 
 2.model.fit(x, y), x (input) and y (output) are 1-dimentional series
 
 3.model.predict(x, y), all x points and a few initial points of y, predicted y series will be returned
