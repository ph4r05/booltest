# pvalues
 - pvalue = probability (in the null hypothesis distribution) to be observed as a value equal to or more extreme than the value observed
 
## computation 
 - Derive CDF -> find 0 regions = extremes
 - Integrate from 0 regions towards region of increasing integral value. 
 - Once sum of all integrations is alpha, stop. Integrated area is a critical region
 - Computation for x: integrate until the first integral boundary hits x. pvalue = sum of integrals
 - Tabulation: for each desired pvalue compute boundaries (4 values) where critical region starts. 
 - pvalue(x): need to do the integration OR function table (\forall zscores: P(zscore) > 0).
 - In our case 4 extremes, integrate: 
   - -\inf towards 0
   - +\inf towards 0
   - 0 towards +\inf
   - 0 towards -\inf
   - 10000 samples, pvalue = 0 -> 1/10000. 
 - absolutize -> we have a new distribution -> 2x more datapoints, 2 tails.  
  
  